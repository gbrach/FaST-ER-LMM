"""LUX progress snapshots and local stage heartbeats.

These maintain cumulative scan state for epi-watch, independently of the
core progress format. None of these helpers changes numerical scan results.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# guards write_status against the heartbeat thread racing the main thread.
# read-merge-write isn't atomic; without this lock a heartbeat can briefly
# rewind chunks_done to a stale value
_WRITE_LOCK = threading.Lock()


_RECENT_CAP = 32


_HOSTNAME = socket.gethostname()  # process-local; doesn't change
_SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID")  # None when not under slurm. baked at module load so the watcher can offer copy-pasteable scancel ids without poking environ on the worker every heartbeat


def _sample_memory() -> dict[str, Any]:
    """current + peak RSS (linux /proc) and torch GPU mem if cuda's loaded.
    cheap, best-effort. VmHWM and torch.max_memory_* track peaks
    continuously between calls so sampling at write time still catches a
    transient spike that happened mid-chunk. also stamps hostname so the
    hub heartbeat poller knows which physical node the slurm task landed on,
    and slurm_job_id when present so the watcher can show scancel-ready ids"""
    out: dict[str, Any] = {"host": _HOSTNAME}
    if _SLURM_JOB_ID:
        out["slurm_job_id"] = _SLURM_JOB_ID
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["rss_mb"] = int(line.split()[1]) / 1024.0
                elif line.startswith("VmHWM:"):
                    out["peak_rss_mb"] = int(line.split()[1]) / 1024.0
                if "rss_mb" in out and "peak_rss_mb" in out:
                    break
    except (FileNotFoundError, OSError, ValueError):
        pass
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                d = torch.cuda.current_device()
                out["gpu_index"] = int(d)
                out["gpu_name"] = torch.cuda.get_device_name(d)
                out["gpu_alloc_mb"] = torch.cuda.memory_allocated(d) / 1024**2
                out["gpu_peak_alloc_mb"] = torch.cuda.max_memory_allocated(d) / 1024**2
                out["gpu_reserved_mb"] = torch.cuda.memory_reserved(d) / 1024**2
                out["gpu_peak_reserved_mb"] = torch.cuda.max_memory_reserved(d) / 1024**2
                free, total = torch.cuda.mem_get_info(d)
                out["gpu_free_mb"] = free / 1024**2
                out["gpu_total_mb"] = total / 1024**2
        except Exception:
            pass
    return out


def _read_state(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _atomic_write_json(p: Path, data: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, p)


def _merge_event(state: dict[str, Any], payload: dict[str, Any]) -> None:
    """folding one event's keys into the snapshot. counters like
    chunks_done are the emitter's job, just plain dict.update here"""
    state.update(payload)


def write_status(
    path: str | Path | None, payload: dict[str, Any],
    *,
    append: bool = False, _heartbeat: bool = False) -> None:
    """recording a progress event into the snapshot + .jsonl timeline.
    append=True keeps the legacy "jsonl only" mode some callers use.
    _heartbeat=True suppresses the .jsonl row + the recent-buffer push
    (otherwise heartbeats would flood the timeline)"""
    if path is None:
        return
    payload = {"ts": time.time(), **_sample_memory(), **payload}
    p = Path(path)
    if append:
        with open(p, "a") as f:
            f.write(json.dumps(payload) + "\n")
        return

    with _WRITE_LOCK:
        state = _read_state(p)
        if "started_at" not in state:
            state["started_at"] = payload["ts"]
        _merge_event(state, payload)
        state["updated_at"] = payload["ts"]
        state["elapsed_s"] = payload["ts"] - state["started_at"]

        if not _heartbeat:
            recent = state.setdefault("recent", [])
            recent.append({k: v for k, v in payload.items() if k != "recent"})
            state["recent"] = recent[-_RECENT_CAP:]

        _atomic_write_json(p, state)

    jsonl = p.with_suffix(p.suffix + ".jsonl")
    try:
        with open(jsonl, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass


# ── vendored from dev fasterlmm.progress: the `stage` context manager + its
# heartbeat plumbing. the epi tier-2 scan wraps long inner kernels in `stage`
# (minutes can pass between tile boundaries) and the watcher relies on the
# re-touched updated_at to not show stalled progress. absent from the public
# core's progress, so carried here alongside write_status. cosmetic / progress
# side-effect only — outside the parity surface.

_HEARTBEAT_S = 10.0  # how often a long-running stage refreshes updated_at


_HEARTBEAT_POST_FN: list = []  # process-global: appended once by cli on startup


def set_heartbeat_post_fn(fn) -> None:
    """register a callable(payload: dict) -> None that the stage heartbeat
    thread will invoke alongside write_status. caller is responsible for
    swallowing transport errors so a dead hub doesn't crash the run"""
    _HEARTBEAT_POST_FN.clear()
    if fn is not None:
        _HEARTBEAT_POST_FN.append(fn)


@contextmanager
def stage(name: str, status_file: str | Path | None = None,
          *, heartbeat_s: float = _HEARTBEAT_S, **fields):
    """timing one named stage: prints start/end on stderr and writes
    running/done transitions to the status file if one's configured.
    spawns a daemon heartbeat thread that re-touches updated_at every
    heartbeat_s while the stage runs, so the watcher doesn't show
    stuck progress during long inner kernels (epi tier-2 scan can run
    minutes between tile boundaries). if a heartbeat post fn was
    registered via set_heartbeat_post_fn, the thread also pushes a
    payload to the hub on each tick"""
    print(f"[{name}] start", file=sys.stderr, flush=True)
    write_status(status_file, {"stage": name, "stage_state": "running", **fields})
    t0 = time.time()

    stop = threading.Event()
    have_post = bool(_HEARTBEAT_POST_FN)
    if status_file is not None and heartbeat_s > 0 or have_post:
        def _beat():
            while not stop.wait(heartbeat_s):
                try:
                    snap = {"stage": name, "stage_state": "running",
                            "stage_elapsed_s": time.time() - t0, **fields}
                    write_status(status_file, snap, _heartbeat=True)
                except Exception:
                    pass  # never let a heartbeat crash the run
                if _HEARTBEAT_POST_FN:
                    try:
                        _HEARTBEAT_POST_FN[0]({"stage": name,
                                               "stage_state": "running",
                                               "stage_elapsed_s": time.time() - t0,
                                               **_sample_memory(),
                                               **fields})
                    except Exception:
                        pass
        thread = threading.Thread(target=_beat, daemon=True,
                                  name=f"progress-heartbeat-{name}")
        thread.start()
    else:
        thread = None

    try:
        yield
    finally:
        stop.set()
        elapsed = time.time() - t0
        print(f"[{name}] done in {elapsed:.1f}s", file=sys.stderr, flush=True)
        write_status(status_file, {"stage": name, "stage_state": "done",
                                   "stage_elapsed_s": elapsed, **fields})
        if _HEARTBEAT_POST_FN:
            try:
                _HEARTBEAT_POST_FN[0]({"stage": name, "stage_state": "done",
                                       "stage_elapsed_s": elapsed,
                                       **_sample_memory(), **fields})
            except Exception:
                pass
