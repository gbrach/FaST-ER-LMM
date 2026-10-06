"""
Run record for reproducibility: one json (run_info.json, run_info.shardX.json for shard tasks) with the command line,
every resolved argument, the input files (path, size, mtime), versions, host and outcome, plus a text log
(run.log, run.shardX.log) that tees stderr and stdout.  Nothing here touches scan math
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SLURM_VARS = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST",
              "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK")


def record_path(outdir: str | Path, shard_i: int | None) -> Path:
    return Path(outdir) / (f"run_info.shard{shard_i}.json" if shard_i is not None else "run_info.json")


def log_path(outdir: str | Path, shard_i: int | None) -> Path:
    return Path(outdir) / (f"run.shard{shard_i}.log" if shard_i is not None else "run.log")


class _Tee:
    """
    Stream wrapper that copies every write into the log file and passes everything else through
    """

    def __init__(self, stream, fh) -> None:
        self._stream = stream
        self._fh = fh

    def write(self, text):
        n = self._stream.write(text)
        try:
            self._fh.write(text)
            self._fh.flush()
        except (OSError, ValueError):
            pass  # a full disk or a closed log must never kill the scan
        return n

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _file_entry(role: str, path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    entry: dict = {"role": role, "path": str(path), "abs_path": str(p.resolve())}
    try:
        st = p.stat()
        entry["size_bytes"] = st.st_size
        entry["mtime"] = datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec = "seconds")
    except OSError:
        entry["missing"] = True
    return entry


def _input_files(args) -> list[dict]:
    out: list[dict] = []
    for role in ("geno", "kinship_geno", "grm"):
        prefix = getattr(args, role, None)
        if prefix:
            out += [_file_entry(f"{role}{ext}", f"{prefix}{ext}") for ext in (".bed", ".bim", ".fam")]
    for role in ("pheno", "covar", "chrom_sizes"):
        out.append(_file_entry(role, getattr(args, role, None)))
    return [e for e in out if e is not None]


def _git_commit() -> str | None:
    here = Path(__file__).resolve().parent
    try:
        res = subprocess.run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"], capture_output = True,
                             text = True, timeout = 3)
        return res.stdout.strip() or None if res.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _versions() -> dict:
    import numpy
    import pyarrow
    import torch

    import fasterlmm
    return {"fasterlmm": fasterlmm.__version__, "git_commit": _git_commit(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": numpy.__version__,
            "pyarrow": pyarrow.__version__}


def _host() -> dict:
    import torch
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # not linux
        cores = os.cpu_count()
    # device_count only, no device queries: the dispatch parent must not touch cuda before its workers pin a GPU
    return {"hostname": socket.gethostname(), "platform": platform.platform(), "cpus_visible": cores,
            "cuda_devices_visible": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm": {k: os.environ[k] for k in SLURM_VARS if k in os.environ}}


def _write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(tmp, "w") as f:
        json.dump(payload, f, indent = 2, default = str)
        f.write("\n")
    tmp.replace(path)


def _tee_to_log(outdir: str | Path, shard_i: int | None, header: str) -> None:
    path = log_path(outdir, shard_i)
    path.parent.mkdir(parents = True, exist_ok = True)
    fh = open(path, "a", buffering = 1)  # append so a rerun into the same outdir keeps the earlier log
    fh.write(f"\n# {header}\n")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


def begin(args, command: str, shard_i: int | None = None, *, started_by: str = "main") -> Path:
    """
    Write the run record and start the log tee for this process
    Call once per process, after argument validation.  shard_i picks the per-shard file names
    """
    outdir = Path(args.outdir)
    outdir.mkdir(parents = True, exist_ok = True)
    cmd = " ".join([sys.argv[0]] + [shlex.quote(a) for a in sys.argv[1:]])
    now = datetime.now().astimezone().isoformat(timespec = "seconds")
    _tee_to_log(outdir, shard_i, f"{now}  {cmd}")
    payload = {"state": "running", "command": command, "command_line": cmd, "cwd": os.getcwd(),
               "started_at": now, "shard_index": shard_i, "pid": os.getpid(), "started_by": started_by,
               "args": vars(args), "inputs": _input_files(args), "versions": _versions(), "host": _host()}
    path = record_path(outdir, shard_i)
    _write(path, payload)
    return path


def begin_worker(args, command: str, rank: int) -> Path:
    """
    Per-GPU worker of an auto-dispatched run: own log and record, the parent keeps the full run_info.json
    """
    outdir = Path(args.outdir)
    now = datetime.now().astimezone().isoformat(timespec = "seconds")
    _tee_to_log(outdir, rank, f"{now}  worker {rank} of {command}")
    path = record_path(outdir, rank)
    _write(path, {"state": "running", "command": command, "started_at": now, "shard_index": rank,
                  "pid": os.getpid(), "started_by": "dispatch", "args": vars(args)})
    return path


def update(outdir: str | Path, shard_i: int | None, **fields) -> None:
    """
    Merge fields into the record, creating it when missing.  Never raises, a record must not fail a scan
    """
    path = record_path(outdir, shard_i)
    try:
        cur = json.loads(path.read_text()) if path.exists() else {}
        cur.update(fields)
        cur["updated_at"] = datetime.now().astimezone().isoformat(timespec = "seconds")
        _write(path, cur)
    except (OSError, ValueError):
        pass


def gpu_name(device: str) -> str | None:
    import torch
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(torch.device(device))
        except (RuntimeError, AssertionError):
            return None
    return None


def outcome(started_at: float, **fields) -> dict:
    return {"wall_s": round(time.time() - started_at, 1), **fields}
