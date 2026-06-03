"""
fasterlmm extreme command-line entry: big-N LOCO GWAS that streams the genotype off disk
The exact low-rank engine (lowrank + extreme_scan) for the case where N gets large enough that forming the N-by-N kernel and holding the whole N-by-M genotype both fall over.  the kinship comes from a capped pruned marker set (auto-strided, or a --grm bed), the test variants stream past it a block at a time, phenos go in chunks that re-stream the genotype per chunk
Builds on the frozen core -- it reuses the gwas writer and the bundle dataset as-is so the output schema is identical, only the scan path underneath is the streamed low-rank one.  single device for now, the multi-gpu / multinode dispatch is the adaptive layer on top
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pyarrow as pa
import scipy.stats as ss
import torch

from fasterlmm.bundle import (
    BUNDLE_FILENAME, BUNDLE_PARTS_DIRNAME, BundleWriter, merge_bundle_parts,
)
from fasterlmm.cli import (
    _default_write_workers, _drain_done, _parse_shard, _resource_stats, _write_pheno,
)
from fasterlmm.extreme_scan import loco_scan_resident, loco_scan_streamed
from fasterlmm.io import read_covar, read_phen
from fasterlmm.io_stream import (
    open_aligned_bed, read_all_standardised, read_grm_factor, select_grm_markers,
    stream_genotype_var,
)
from fasterlmm.normalize import rint_columns
from fasterlmm.progress import write_status


def _pack_perms(Y_real: torch.Tensor,
                pheno_idx: list[int],
                n_perm: int,
                seed: int) -> torch.Tensor:
    """
    Pack the B real phenos and their B * n_perm row-shuffles into one (N, B + B*n_perm) tensor
    Same independant per-pheno permutation scheme as perms.perm_threshold -- pheno b's shuffles are seeded on its own colum index so they're stable regardless of how the phenos get chunked.  kept as local code so the parity-locked perms.py stays frozen
    """
    N, B = Y_real.shape
    orders = np.empty((N, B, n_perm), dtype=np.int64)
    for b_pos, p in enumerate(pheno_idx):
        rng = np.random.default_rng([seed, int(p)])
        orders[:, b_pos, :] = rng.random((n_perm, N)).argsort(axis=1).T
    gather_idx = torch.from_numpy(orders).to(Y_real.device)
    y_perms = torch.gather(Y_real.unsqueeze(2).expand(N, B, n_perm), 0, gather_idx)
    y_perms = y_perms.reshape(N, B * n_perm)  # pheno-major: [b0 perms, b1 perms, ...]
    return torch.cat([Y_real, y_perms], dim=1)  # (N, B + B*n_perm)


def _meminfo_available_bytes() -> int | None:
    """
    MemAvailable from /proc/meminfo, or None when it can't be read
    The kernel's own estimate of what a fresh alocation can claim without swapping.  this is the NODE's free ram though, not what a slurm cgroup actualy lets the job touch, so _ram_headroom_bytes caps it with the cgroup limit
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # the value is in kB
    except OSError:
        pass
    return None


def _cgroup_headroom_bytes() -> int | None:
    """
    Whats left of this job's cgroup memory limit (limit - current usage), v2 then v1, or None when uncapped / unreadable
    Slurm caps a --mem job trough a cgroup, and /proc/meminfo dosn't see that cap -- so without this the probe reads the whole node's free ram and happily over-commits a 40GB slab into a job that only got 16GB, wich the oom killer then ends.  a "max" limit means uncapped
    """
    pairs = [("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),  # cgroup v2
             ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
              "/sys/fs/cgroup/memory/memory.usage_in_bytes")]  # cgroup v1
    for limit_path, usage_path in pairs:
        try:
            raw = Path(limit_path).read_text().strip()
            if raw == "max":
                return None
            limit = int(raw)
            if limit > (1 << 62):  # v1 writes a sentinel near INT64_MAX when uncapped
                return None
            usage = int(Path(usage_path).read_text().strip())
            return max(limit - usage, 0)
        except (OSError, ValueError):
            continue
    return None


def _ram_headroom_bytes() -> int | None:
    """The safe headroom for a new slab: node MemAvailable capped by the cgroup limit, None if niether reads"""
    avail = _meminfo_available_bytes()
    cg = _cgroup_headroom_bytes()
    vals = [v for v in (avail, cg) if v is not None]
    return min(vals) if vals else None


def _want_resident(mode: str, need_bytes: int) -> bool:
    """
    Decide whether to hold the standardised genotype resident insted of re-streaming it per batch
    auto probes the free ram (cgroup-capped) and goes resident only when the N x M slab leaves room to spare for the working set + perms + writers (half the headroom, a deliberately roomy margin).  on forces it, off always streams -- the streaming fallback is what keeps the path alive past the point N x M stops fiting, the march toward a million strains
    """
    if mode == "off":
        return False
    if mode == "on":
        return True
    head = _ram_headroom_bytes()
    if head is None:  # can't probe, play safe and stream
        return False
    return need_bytes < 0.5 * head


def _run_extreme(args: argparse.Namespace, shard_i: int | None,
                 shard_n: int | None, device: str) -> None:
    """Load inputs, build the pruned kinship factor, stream the LOCO scan per pheno chunk, write the bundle

    shard_i / shard_n slice the pheno list round-robin so concurent workers (one per GPU, or a slurm-array task) split the phenos evenly.  status file is per-shard so the workers dont stomp each other
    """
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    status_file = str(outdir / (f"status.shard{shard_i}.json" if shard_i is not None else "status.json"))
    log_prefix = f"[shard {shard_i}] " if shard_i is not None else ""
    write_status(status_file, {"state": "loading", "geno": args.geno, "pheno": args.pheno,
                               "shard": f"{shard_i}/{shard_n}" if shard_n else None})
    dtype = torch.float64 if args.float64 else torch.float32

    pheno = read_phen(args.pheno)
    if args.rint:
        pheno.Y = rint_columns(pheno.Y)  # Blom RINT before alignment, rank-then-qnorm is row-order invariant
    covar = read_covar(args.covar) if args.covar else None

    common = set(pheno.iid)
    if covar is not None:
        common &= set(covar.iid)
    handle = open_aligned_bed(args.geno, list(common))  # bed-order rows filtered to the common strains
    iid = handle.iid
    N = len(iid)
    if N == 0:
        raise ValueError("no strain overlap between geno / pheno / covar")

    p_pos = {s: i for i, s in enumerate(pheno.iid)}
    p_idx = [p_pos[s] for s in iid]
    Y = torch.from_numpy(pheno.Y[p_idx, :]).to(dtype)
    if covar is not None:
        c_pos = {s: i for i, s in enumerate(covar.iid)}
        c_idx = [c_pos[s] for s in iid]
        X = np.concatenate([np.ones((N, 1)), covar.C[c_idx, :]], axis=1)
    else:
        X = np.ones((N, 1))
    X = torch.from_numpy(X).to(dtype)

    # kinship factor: a separate pre-pruned --grm bed, else a strided auto-prune of the test bed
    if args.grm:
        grm_handle = open_aligned_bed(args.grm, iid)
        if grm_handle.iid != iid:
            raise ValueError("--grm strains don't cover the geno/pheno strain set")
        G = read_grm_factor(grm_handle, np.arange(len(grm_handle.sid)), dtype=dtype)
        g_chrom = grm_handle.chrom
    else:
        grm_cols = select_grm_markers(handle, args.grm_k)
        G = read_grm_factor(handle, grm_cols, dtype=dtype)
        g_chrom = handle.chrom[grm_cols]

    print(f"{log_prefix}loaded N={N} M={len(handle.sid)} P_total={Y.shape[1]} k_grm={G.shape[1]} "
          f"device={device} dtype={str(dtype).replace('torch.', '')}", file=sys.stderr, flush=True)

    g_var = stream_genotype_var(handle, block_size=args.block_size)  # raw per-SNP var for EffectSize, one pass
    if device != "cpu":
        G = G.to(device)
        X = X.to(device)

    # resident-vs-stream: the per-batch stream is ~2/3 of the wall and it's pure repeat work (the standardise
    # dosn't depend on the phenos), so when the standardised N x M slab fits ram we decode it once here and slice
    # it per batch insted of re-reading the bed every time.  past the point it stops fiting we fall back to streaming
    M_test = len(handle.sid)
    elem = 8 if dtype == torch.float64 else 4
    resident = _want_resident(args.resident, N * M_test * elem)
    Zres = None
    if resident:
        write_status(status_file, {"state": "preloading", "mode": "extreme", "N": N, "M": M_test,
                                   "shard": f"{shard_i}/{shard_n}" if shard_n else None, "device": device,
                                   "resident": True})
        t_pre = time.time()
        Zres = read_all_standardised(handle, block_size=args.block_size, dtype=dtype)
        print(f"{log_prefix}resident genotype {N}x{M_test} ({elem * N * M_test / 1e9:.0f}GB) decoded once in "
              f"{time.time() - t_pre:.0f}s, no per-batch re-stream", file=sys.stderr, flush=True)
    else:
        print(f"{log_prefix}streaming genotype per batch (slab too big for ram, or --resident off)",
              file=sys.stderr, flush=True)

    P_total = Y.shape[1]
    if args.pheno_idx is not None:
        pheno_list = [args.pheno_idx]
    else:
        start = args.pheno_start or 0
        end = args.pheno_end if args.pheno_end is not None else P_total
        pheno_list = list(range(start, end))
    # round-robin slice so each shard carries a balanced spread of phenos, display_offset tiles the printed ranges
    display_offset = 0
    if shard_n is not None:
        display_offset = sum(len(pheno_list[k::shard_n]) for k in range(shard_i))
        pheno_list = pheno_list[shard_i::shard_n]

    M = len(handle.sid)
    C = X.shape[1]
    df2 = N - C - 1
    if args.write_workers is None:
        args.write_workers = _default_write_workers(shard_n or 1)

    writer_ctx = {"snp_id": pa.array(handle.sid, type=pa.string()),
                  "chrom": handle.chrom, "pos": handle.pos,
                  "g_var": g_var,
                  "sid_index": np.arange(M, dtype=np.int64),
                  "gendist": pa.nulls(M, pa.float64()),
                  "mixing": np.zeros(M),
                  "phenocount": np.full(M, 1 + args.n_perm),
                  "pheno_idx": pa.array(np.zeros(M, dtype=np.int32))}
    if args.bundle:
        if shard_i is not None:
            bundle_path = outdir / BUNDLE_PARTS_DIRNAME / f"shard{shard_i}.parquet"
        else:
            bundle_path = outdir / BUNDLE_FILENAME
        bundle_writer = BundleWriter(bundle_path)
    else:
        bundle_writer = None
    writer_pool = ThreadPoolExecutor(max_workers=args.write_workers)

    n_chroms = len(set(handle.chrom.tolist()))
    shard_str = f"{shard_i}/{shard_n}" if shard_n else None
    started_at = time.time()
    status_base = {"state": "scanning", "mode": "extreme", "N": N, "M": M, "n_perm": args.n_perm,
                   "shard": shard_str, "device": device, "dtype": str(dtype).replace("torch.", ""),
                   "loco": True, "phenos_total": len(pheno_list), "pid": os.getpid(),
                   "started_at": started_at, "chroms_total": n_chroms, "k_grm": int(G.shape[1]),
                   "resident": resident}
    write_status(status_file, {**status_base, "phenos_done": 0, "elapsed_s": 0.0, **_resource_stats(device)})

    done = 0
    write_futures: list = []
    try:
        for b_start in range(0, len(pheno_list), args.phenos_per_job):
            batch = pheno_list[b_start:b_start + args.phenos_per_job]
            B = len(batch)
            Y_real = Y[:, batch].to(device)
            Y_all = _pack_perms(Y_real, batch, args.n_perm, args.seed)  # (N, B + B*n_perm)

            def _on_chrom(k, n, _done=done, _b=B):
                pending = sum(1 for f in write_futures if not f.done())
                write_status(status_file, {**status_base, "phenos_done": _done + _b * k / n,
                                           "elapsed_s": time.time() - started_at,
                                           "chroms_done": k, "writes_pending": pending,
                                           **_resource_stats(device)})

            t_batch = time.time()
            if Zres is not None:
                res = loco_scan_resident(Zres, handle.chrom, G, g_chrom, X, Y_all, n_real=B,
                                         block_size=args.block_size, dtype=dtype, on_chrom=_on_chrom)
            else:
                res = loco_scan_streamed(handle, G, g_chrom, X, Y_all, n_real=B,
                                         block_size=args.block_size, dtype=dtype, on_chrom=_on_chrom)
            perm_max_F = res.max_F[B:].reshape(B, args.n_perm)
            p_real = ss.f.sf(res.f.cpu().numpy(), 1, df2)  # (M, B)
            perm_min_p = ss.f.sf(perm_max_F.cpu().numpy(), 1, df2)  # (B, n_perm)
            beta_np = res.beta.cpu().numpy()
            se_np = res.se.cpu().numpy()
            sfve_np = res.sfve.cpu().numpy()
            nullh2_np = res.nullh2.cpu().numpy()
            p_var = Y_real.cpu().numpy().var(axis=0)  # (B,)

            for j, p in enumerate(batch):
                write_futures.append(writer_pool.submit(
                    _write_pheno, writer_ctx, str(outdir), pheno.names[p],
                    p_col=p_real[:, j], beta_col=beta_np[:, j], se_col=se_np[:, j],
                    sfve_col=sfve_np[:, j], nullh2_col=nullh2_np[:, j],
                    p_var=float(p_var[j]), perm_min_p=perm_min_p[j],
                    perm_quantile=args.perm_quantile, per_pheno_dirs=args.per_pheno_dirs,
                    bundle_writer=bundle_writer))
            write_futures = _drain_done(write_futures)
            done += B
            print(f"{log_prefix}batch {display_offset + b_start}..{display_offset + b_start + B}: "
                  f"{B} phenos in {time.time() - t_batch:.1f}s",
                  file=sys.stderr, flush=True)
            write_status(status_file, {**status_base, "phenos_done": done,
                                       "elapsed_s": time.time() - started_at, "chroms_done": n_chroms,
                                       "writes_pending": len(write_futures), **_resource_stats(device)})

        for f in as_completed(write_futures):
            f.result()
    finally:
        writer_pool.shutdown(wait=True)
    if bundle_writer is not None:
        bundle_writer.close()

    print(f"{log_prefix}done: {len(pheno_list)} phenos in {time.time() - started_at:.1f}s",
          file=sys.stderr, flush=True)
    write_status(status_file, {**status_base, "state": "done", "phenos_done": len(pheno_list),
                               "elapsed_s": time.time() - started_at, **_resource_stats(device)})


def _shard_entrypoint(rank: int, n_gpu: int, args_dict: dict) -> None:
    """
    Multiprocessing worker, one per GPU
    Pins the process to its own device with CUDA_VISIBLE_DEVICES before any cuda call so the worker sees one GPU as cuda:0 -- seperate processes each get their own cuda context, wich also sidesteps the eigh lazy-init thread race.  args arrives as a dict so the parent never imports torch before the children set the env var
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    args = argparse.Namespace(**args_dict)
    _run_extreme(args, shard_i=rank, shard_n=n_gpu, device="cuda:0")


def main() -> None:
    """
    Parse args and run the extreme scan, same dispatch + bundle-merge shape as the gwas entry
    """
    parser = argparse.ArgumentParser(
        prog="fasterlmm extreme",
        description="big-N LOCO GWAS, exact low-rank scan with a streamed genotype and a capped kinship")
    parser.add_argument("--geno", required=True, help="plink BED prefix for the test variants")
    parser.add_argument("--pheno", required=True, help="wide phen tsv with Strain column")
    parser.add_argument("--covar", default=None, help="plink-style .cov (optional)")
    parser.add_argument("--grm", default=None,
                        help="plink BED prefix for a pre-pruned kinship marker set (optional, else auto-strided from --geno)")
    parser.add_argument("--grm-k", type=int, default=5000,
                        help="target kinship marker count for the auto-stride when --grm is not given (default 5000)")
    parser.add_argument("--outdir", required=True, help="output dir, created if missing")
    parser.add_argument("--pheno-idx", type=int, default=None, help="0-based pheno column for a single-pheno scan")
    parser.add_argument("--pheno-start", type=int, default=None, help="0-based start of a pheno range (inclusive)")
    parser.add_argument("--pheno-end", type=int, default=None, help="0-based end of a pheno range (exclusive)")
    parser.add_argument("--n-perm", type=int, default=100, help="permutation count for the threshold")
    parser.add_argument("--perm-quantile", type=float, default=0.05,
                        help="quantile of per-perm min-p used as the genome-wide threshold (default 0.05)")
    parser.add_argument("--phenos-per-job", type=int, default=64,
                        help="real phenos packed into one streamed scan, the genotype re-streams once per chunk. gpu cols = this x (1 + n_perm)")
    parser.add_argument("--block-size", type=int, default=8192,
                        help="test variants held resident per streamed block, the lever that keeps the N x M genotype off the heap")
    parser.add_argument("--resident", choices=["auto", "on", "off"], default="auto",
                        help="hold the standardised genotype resident when N x M fits ram (auto, the default), force on, or always stream (off). resident decodes once insted of re-streaming every pheno batch")
    parser.add_argument("--write-workers", type=int, default=None, help="writer threads, unset fills the core allocation")
    parser.add_argument("--rint", action=argparse.BooleanOptionalAction, default=True,
                        help="Blom rank-based inverse normal transform per pheno, ON by default")
    parser.add_argument("--float64", action="store_true", help="run in float64 (default float32, the scale setting)")
    parser.add_argument("--seed", type=int, default=19930909)
    parser.add_argument("--device", default="cuda",
                        help="cuda (auto-dispatch across visible GPUs), cuda:N (single device), cpu, or mps")
    parser.add_argument("--shard", default=None,
                        help="X/N to process only the X-th of N pheno shards (explicit, e.g. slurm-array)")
    parser.add_argument("--no-multi-gpu", action="store_true",
                        help="opt out of auto-dispatch when --device cuda sees more than one GPU")
    parser.add_argument("--bundle", action="store_true", help="stream the per-pheno tables into one gwas_bundle.parquet")
    parser.add_argument("--no-per-pheno-dirs", dest="per_pheno_dirs", action="store_false", default=True,
                        help="skip the per-pheno output tree, write only the bundle (needs --bundle)")
    args = parser.parse_args()
    if not args.per_pheno_dirs and not args.bundle:
        parser.error("--no-per-pheno-dirs needs --bundle, otherwise nothing gets written")

    # clear a stale parts dir up front, only the orchestrator does this (a --shard array task would race)
    if args.bundle and args.shard is None:
        shutil.rmtree(Path(args.outdir) / BUNDLE_PARTS_DIRNAME, ignore_errors=True)

    # point them at the live watcher -- it takes the outdir and autodiscovers single-pane vs the per-shard dashboard
    if args.shard is None:
        print(f"watch live:  fasterlmm watch {args.outdir}", file=sys.stderr, flush=True)

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    auto_dispatch = (args.device == "cuda" and args.shard is None
                     and not args.no_multi_gpu and n_gpu > 1)

    if auto_dispatch:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        write_status(str(outdir / "status.json"),
                     {"state": "dispatch", "n_gpu": n_gpu, "shards": list(range(n_gpu))})
        # plain multiprocessing -- each child sets CUDA_VISIBLE_DEVICES before touching cuda
        ctx = multiprocessing.get_context("spawn")
        if args.write_workers is None:
            args.write_workers = _default_write_workers(n_gpu)
        args_dict = vars(args).copy()
        procs = [ctx.Process(target=_shard_entrypoint, args=(r, n_gpu, args_dict)) for r in range(n_gpu)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        if any(p.exitcode != 0 for p in procs):
            raise SystemExit("a gpu shard worker failed, see the traceback above")
    else:
        if args.shard:
            shard_i, shard_n = _parse_shard(args.shard)
        else:
            shard_i, shard_n = None, None
        device = args.device
        if device == "cuda" and n_gpu == 1:
            device = "cuda:0"
        _run_extreme(args, shard_i=shard_i, shard_n=shard_n, device=device)

    # fold the per-shard bundle parts into the final dataset, same as the gwas path (a --shard array uses fasterlmm concat)
    if args.shard is None:
        final: dict = {"state": "done", "n_gpu": n_gpu if auto_dispatch else 1}
        if args.bundle:
            outdir = Path(args.outdir)
            parts = outdir / BUNDLE_PARTS_DIRNAME
            shards = sorted(parts.glob("shard*.parquet")) if parts.is_dir() else []
            if shards:
                bundle_path = merge_bundle_parts(outdir)
                final["bundle"] = str(bundle_path)
                print(f"\n{len(shards)} shards gathered into {bundle_path}", flush=True)
            else:
                final["bundle"] = str(outdir / BUNDLE_FILENAME)
        # merge over the rich done payload the single-device worker already wrote (N, M, phenos, peak RAM, elapsed)
        # so the watcher pointed at a finished run still shows the final summary insted of an empty 0% pane
        status_path = Path(args.outdir) / "status.json"
        existing: dict = {}
        try:
            existing = json.loads(status_path.read_text())
        except (OSError, ValueError):
            pass
        write_status(str(status_path), {**existing, **final})


if __name__ == "__main__":
    main()
