"""
fasterlmm gwas command-line entry: plink + phen (+ optional covar) -> per-pheno LOCO scan + perm threshold
--pheno-start/--pheno-end carves out a pheno range, --shard X/N is the explicit slurm-array slice. bare --device cuda fans out one worker per visble GPU.  --bundle collapses the per-pheno tsvs into a single parquet at the end
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# metal has no float64 and a fistful of linalg ops (eigh above all) just arent
# wired up for mps, so letting torch quietly bounce those back to cpu instead
# of hard-erroring partway through a scan.  inert on cuda / cpu boxes so it
# costs nothing to set always, just has to land before torch gets imported
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import scipy.stats as ss
import torch

from fasterlmm.bundle import bundle_outdir
from fasterlmm.io import align_inputs, read_covar, read_phen, read_plink
from fasterlmm.normalize import rint_columns
from fasterlmm.perms import perm_threshold
from fasterlmm.progress import write_status


def _parse_shard(shard: str) -> tuple[int, int]:
    """X/N -> (X, N) with bounds check.  Empty / None means no sharding"""
    i, n = shard.split("/")
    i, n = int(i), int(n)
    if not (0 <= i < n):
        raise ValueError(f"shard index {i} out of range for n={n}")
    return i, n


def _resource_stats(device: str) -> dict:
    """
    Current + peak host RSS and, on cuda, torch's per-process gpu memory, all in MB
    The watcher only ever sees the shared filesystem, so the worker has to read its own /proc and torch counters and ship them inside the status snapshot
    """
    out: dict = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["rss_mb"] = int(line.split()[1]) / 1024
                elif line.startswith("VmHWM:"):
                    out["peak_rss_mb"] = int(line.split()[1]) / 1024
    except OSError:
        pass
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            dev = torch.device(device)
            out["gpu_alloc_mb"] = torch.cuda.memory_allocated(dev) / 1e6
            out["gpu_peak_alloc_mb"] = torch.cuda.max_memory_allocated(dev) / 1e6
            out["gpu_total_mb"] = torch.cuda.get_device_properties(dev).total_memory / 1e6
        except (RuntimeError, AssertionError):
            pass
    return out


# snp_id / chrom / pos are the same for every pheno, so they go through this module global
# instead of being pickled into each write task -- the writer pool forks, so the workers
# inherit whatever is in here at fork time for free
_WRITER_CTX: dict = {}


def _write_pheno(outdir_str: str, pheno_name: str, p_col, beta_col, se_col,
                 sfve_col, nullh2_col, p_var: float, perm_min_p,
                 perm_quantile: float, n_perm: int) -> None:
    """
    Writer-pool worker: one pheno's gwas.tsv + perms.tsv + threshold.txt
    gwas.tsv carries the fastlmm single_snp column set (sid_index .. PhenoCount, sorted by PValue) so it drops straight into a fastlmm-shaped pipeline.  Runs in a forked process so the csv writing never blocks the gpu -- the next batch scans while this one writes
    """
    snp_id = _WRITER_CTX["snp_id"]
    sub = Path(outdir_str) / pheno_name
    sub.mkdir(parents=True, exist_ok=True)
    thresh = float(np.quantile(perm_min_p, perm_quantile))
    # EffectSize = beta^2 * var(genotype) / var(pheno), fastlmm single_snp.py:1454
    effect_size = beta_col * beta_col * _WRITER_CTX["g_var"] / p_var
    pd.DataFrame({"sid_index": np.arange(len(snp_id)),
                  "SNP": snp_id,
                  "Chr": _WRITER_CTX["chrom"],
                  "GenDist": np.nan,
                  "ChrPos": _WRITER_CTX["pos"],
                  "PValue": p_col,
                  "SnpWeight": beta_col,
                  "SnpWeightSE": se_col,
                  "EffectSize": effect_size,
                  "SnpFractVarExpl": sfve_col,
                  "Mixing": 0.0,
                  "Nullh2": nullh2_col,
                  "Pheno": pheno_name,
                  "PhenoCount": 1 + n_perm,
                  }).sort_values("PValue").reset_index(drop=True).to_csv(
                      sub / "gwas.tsv", sep="\t", index=False)
    pd.DataFrame({"perm_min_p": perm_min_p}).to_csv(sub / "perms.tsv", sep="\t", index=False)
    (sub / "threshold.txt").write_text(f"{thresh:.6e}\n")


def _drain_done(futures: list) -> list:
    """dropping finished write futures (re-raising any that failed), returns the still-pending ones so the in-flight list and the pheno arrays it pins stay bounded"""
    pending = []
    for f in futures:
        if f.done():
            f.result()  # re-raises if the writer worker hit an error
        else:
            pending.append(f)
    return pending


def _run_scan(args: argparse.Namespace, shard_i: int | None,
              shard_n: int | None, device: str) -> None:
    """
    sLoading inputs, slicing the pheno list down to this shards chunk, loop, write per-pheno outputs
    status file is status.shard{i}.json when shard_i is set so concurrent workers dont stomp the same file
    """
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    status_file = str(outdir / (f"status.shard{shard_i}.json" if shard_i is not None else "status.json"))
    write_status(status_file, {"state": "loading", "geno": args.geno, "pheno": args.pheno,
                               "shard": f"{shard_i}/{shard_n}" if shard_n else None})

    log_prefix = f"[shard {shard_i}] " if shard_i is not None else ""
    print(f"{log_prefix}device {device}, loading inputs", file=sys.stderr, flush=True)

    geno = read_plink(args.geno)
    pheno = read_phen(args.pheno)
    if args.rint:
        # Blom RINT before alignment so the strain order doesn't matter -- rank-then-qnorm is invariant to row permutation but applying here keeps the pipeline short
        pheno.Y = rint_columns(pheno.Y)
    covar = read_covar(args.covar) if args.covar else None
    # mps cant touch float64 at all, so an apple-gpu run drops to float32 -- not
    # bit-for-bit with fastlmm anymore but well inside float noise.  cuda / cpu
    # stay float64 so the parity path is left exactly as it was
    if device.startswith("mps") and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps but this torch build has no working mps backend, "
                           "need a recent torch on apple silicon")
    dtype = torch.float32 if device.startswith("mps") else torch.float64
    data = align_inputs(geno, pheno, covar, dtype=dtype)

    # spinning up the writer pool here, before the first cuda call -- the workers fork from a
    # process that hasnt touched cuda yet, so they inherit a clean state (forking after cuda
    # init is unsafe).  snp_id / chrom / pos ride a module global the forked workers inherit
    writer_pool = None
    if not args.dry_run:
        _WRITER_CTX["snp_id"] = data.snp_id
        _WRITER_CTX["chrom"] = data.chrom
        _WRITER_CTX["pos"] = data.pos
        # raw per-SNP genotype variance for the EffectSize column, nan-aware since raw Z keeps missing calls
        _WRITER_CTX["g_var"] = np.nanvar(data.Z.cpu().numpy(), axis=0)
        writer_pool = ProcessPoolExecutor(max_workers=args.write_workers,
                                          mp_context=multiprocessing.get_context("fork"))

    if device != "cpu":
        data.Z = data.Z.to(device)
        data.X = data.X.to(device)
        data.Y = data.Y.to(device)
    gpu_name = (f", {torch.cuda.get_device_name(0)}"
                if device.startswith("cuda") and torch.cuda.is_available() else "")
    print(f"{log_prefix}loaded N={data.Y.shape[0]} M={data.Z.shape[1]} "
          f"P_total={data.Y.shape[1]}{gpu_name}", file=sys.stderr, flush=True)

    P_total = data.Y.shape[1]
    if args.pheno_idx is not None:
        pheno_list = [args.pheno_idx]
    else:
        start = args.pheno_start or 0
        end = args.pheno_end if args.pheno_end is not None else P_total
        pheno_list = list(range(start, end))
    if shard_n is not None:
        pheno_list = pheno_list[shard_i::shard_n]  # round-robin slice so progress stays balanced acros shards

    shard_str = f"{shard_i}/{shard_n}" if shard_n else None
    started_at = time.time()
    # status_base carries every field the watcher needs. write_status overwrites the file on each call, so each write has to be self-contained -- spreading status_base back in keeps that true
    status_base = {"state": "scanning", "N": data.Y.shape[0], "M": data.Z.shape[1],
                   "n_perm": args.n_perm, "shard": shard_str, "device": device,
                   "dtype": str(dtype).replace("torch.", ""), "loco": args.loco,
                   "phenos_total": len(pheno_list), "pid": os.getpid(),
                   "started_at": started_at}
    write_status(status_file, {**status_base, "phenos_done": 0, "elapsed_s": 0.0,
                               **_resource_stats(device)})

    if args.dry_run:
        # dry-run lives here (after load + slice, before per-pheno work) so the printed numbers reflect what would actually run -- N/M after intersection, P after pheno slicing + sharding
        shard_str = f"{shard_i}/{shard_n}" if shard_n is not None else "single"
        print(f"[dry-run] N={data.Y.shape[0]} M={data.Z.shape[1]} P={len(pheno_list)} "
              f"n_perm={args.n_perm} loco={args.loco} rint={args.rint} "
              f"device={device} dtype={str(dtype).replace('torch.', '')} "
              f"shard={shard_str} perm_quantile={args.perm_quantile}",
              flush=True)
        write_status(status_file, {"state": "dry-run", "phenos_total": len(pheno_list),
                                   "shard": shard_str})
        return

    N, C = data.X.shape
    df2 = N - C - 1
    # phenos go through the scan in batches -- a whole batch (reals + every perm column) shares one
    # per-chromosome eigendecomposition, so the eigh is paid once per batch instead of once per pheno
    done = 0
    write_futures: list = []
    try:
        for b_start in range(0, len(pheno_list), args.phenos_per_job):
            batch = pheno_list[b_start:b_start + args.phenos_per_job]
            B = len(batch)

            def _on_chrom(k, n, _done=done, _b=B):
                # mid-batch heartbeat -- credit the in-flight batch fraction by fraction so the
                # watcher bar still crawls forward through a long scan instead of jumping per batch
                write_status(status_file, {**status_base, "phenos_done": _done + _b * k / n,
                                           "elapsed_s": time.time() - started_at,
                                           "chroms_done": k, "chroms_total": n,
                                           **_resource_stats(device)})

            t_batch = time.time()
            res, perm_max_F = perm_threshold(data, batch, n_perm=args.n_perm,
                                             seed=args.seed, loco=args.loco,
                                             on_chrom=_on_chrom if args.loco else None)
            # F -> p once for the whole batch, scipy on cpu is fine.  F itself is not written --
            # fastlmm's schema carries SnpWeight / SnpWeightSE and F is recoverable from them
            p_real = ss.f.sf(res.f.cpu().numpy(), 1, df2)  # (M, B)
            perm_min_p = ss.f.sf(perm_max_F.cpu().numpy(), 1, df2)  # (B, n_perm)
            beta_np = res.beta.cpu().numpy()
            se_np = res.se.cpu().numpy()
            sfve_np = res.sfve.cpu().numpy()
            nullh2_np = res.nullh2.cpu().numpy()
            # raw pheno variance, post-RINT as the lmm saw it, for the EffectSize column (ddof=0)
            p_var = data.Y[:, batch].cpu().numpy().var(axis=0)  # (B,)

            # handing each pheno's output files to the writer pool -- the gpu starts the next batch's
            # scan while these csv writes run, and within a batch the writes spread over workers
            for j, p in enumerate(batch):
                write_futures.append(writer_pool.submit(
                    _write_pheno, str(outdir), data.pheno_names[p],
                    p_real[:, j], beta_np[:, j], se_np[:, j], sfve_np[:, j],
                    nullh2_np[:, j], float(p_var[j]), perm_min_p[j],
                    args.perm_quantile, args.n_perm))
            write_futures = _drain_done(write_futures)

            done += B
            print(f"{log_prefix}batch {b_start}..{b_start + B}: {B} phenos scanned in "
                  f"{time.time() - t_batch:.1f}s, {len(write_futures)} writes in flight",
                  file=sys.stderr, flush=True)
            # per-shard progress for the watcher -- reporting only, the scan numbers above are untouched
            write_status(status_file, {**status_base, "phenos_done": done,
                                       "elapsed_s": time.time() - started_at,
                                       "writes_pending": len(write_futures),
                                       **_resource_stats(device)})

        # gpu scan done, wait out the trailing per-pheno writes
        for f in write_futures:
            f.result()
    finally:
        writer_pool.shutdown(wait=True)

    print(f"{log_prefix}done: {len(pheno_list)} phenos in {time.time() - started_at:.1f}s",
          file=sys.stderr, flush=True)
    write_status(status_file, {**status_base, "state": "done",
                               "phenos_done": len(pheno_list),
                               "elapsed_s": time.time() - started_at,
                               **_resource_stats(device)})


def _shard_entrypoint(rank: int, n_gpu: int, args_dict: dict) -> None:
    """
    Multiprocessing worker, one per GPU
    Pins the process to its own device by setting CUDA_VISIBLE_DEVICES before any cuda call, so the worker sees exactly one GPU and that GPU is cuda:0.  Letting every worker see the whole 2-GPU set instead makes the cuda runtime init contend across processes -- the workers then come up ragged, one lagging the other by ten-plus seconds.  Pinning each to its own device inits them independently and they start together
    args arrives as a plain dict because the parent must not import torch / touch cuda before the children get to set the env var
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    args = argparse.Namespace(**args_dict)
    _run_scan(args, shard_i=rank, shard_n=n_gpu, device="cuda:0")


def main() -> None:
    parser = argparse.ArgumentParser(prog="fasterlmm gwas",
                                     description="torch port of fastlmm GWAS with LOCO + perm threshold")
    parser.add_argument("--geno", required=True, help="plink BED prefix")
    parser.add_argument("--pheno", required=True, help="wide phen tsv with Strain column")
    parser.add_argument("--covar", default=None, help="plink-style .cov (optional)")
    parser.add_argument("--outdir", required=True, help="output dir, will be created if missing")
    parser.add_argument("--pheno-idx", type=int, default=None, help="0-based pheno column for a single-pheno scan")
    parser.add_argument("--pheno-start", type=int, default=None, help="0-based start of a pheno range (inclusive)")
    parser.add_argument("--pheno-end", type=int, default=None, help="0-based end of a pheno range (exclusive)")
    parser.add_argument("--loco", action=argparse.BooleanOptionalAction, default=True,
                        help="leave-one-chromosome-out, ON by default (use --no-loco to fit one K over all SNPs)")
    parser.add_argument("--n-perm", type=int, default=100, help="permutation count for the threshold")
    parser.add_argument("--perm-quantile", type=float, default=0.05,
                        help="quantile of per-perm min-p used as the genome-wide threshold (default 0.05)")
    parser.add_argument("--phenos-per-job", type=int, default=256,
                        help="real phenos packed into one gpu scan, gpu cols = this x (1 + n_perm). bigger amortizes the per-chromosome eigendecomposition over more phenos, smaller trims gpu memory")
    parser.add_argument("--write-workers", type=int, default=8,
                        help="worker processes for the per-pheno output writing, which runs off the gpu thread so the next batch can scan while this one writes")
    parser.add_argument("--rint", action=argparse.BooleanOptionalAction, default=True,
                        help="Blom rank-based inverse normal transform on each pheno column, ON by default (use --no-rint to disable)")
    parser.add_argument("--seed", type=int, default=19930909)
    parser.add_argument("--device", default="cuda", help="cuda (auto-dispatch across visible GPUs), cuda:N (single device), mps (apple silicon gpu, runs float32), or cpu")
    parser.add_argument("--shard", default=None, help="X/N to process only the X-th of N pheno shards (explicit, e.g. slurm-array)")
    parser.add_argument("--no-multi-gpu", action="store_true", help="opt out of auto-dispatch when --device cuda sees more than one GPU")
    parser.add_argument("--bundle", action="store_true", help="after scanning, bundle per-pheno gwas.tsv into one parquet")
    parser.add_argument("--dry-run", action="store_true",
                        help="load inputs, print the planned work (N/M/P, n_perm, shards, device), and exit before scanning")
    args = parser.parse_args()

    # auto-dispatch fires when --device cuda is bare, no --shard, no opt-out, and theres >1 visible GPU
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    auto_dispatch = (args.device == "cuda"
                     and args.shard is None
                     and not args.no_multi_gpu
                     and n_gpu > 1)

    if auto_dispatch:
        # parent manifest up front so the watcher pointed at outdir knows how many shards to wait for, even before any worker has writen its first status
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        write_status(str(outdir / "status.json"),
                     {"state": "dispatch", "n_gpu": n_gpu, "shards": list(range(n_gpu))})
        # plain multiprocessing, not torch.multiprocessing -- each child sets CUDA_VISIBLE_DEVICES
        # before it ever touches cuda (see _shard_entrypoint), which torch.mp.spawn doesnt leave room for
        ctx = multiprocessing.get_context("spawn")
        args_dict = vars(args).copy()
        procs = [ctx.Process(target=_shard_entrypoint, args=(r, n_gpu, args_dict))
                 for r in range(n_gpu)]
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
        # collapsing bare cuda with 1 GPU to cuda:0, otherwise passing through whatever the user asked for
        device = args.device
        if device == "cuda" and n_gpu == 1:
            device = "cuda:0"
        _run_scan(args, shard_i=shard_i, shard_n=shard_n, device=device)

    if args.shard is None:
        final: dict = {"state": "done", "n_gpu": n_gpu if auto_dispatch else 1}
        if args.bundle:
            # bundle once from the parent (after spawn join) or after the single-shard scan, so the workers dont race the parquet write
            final["bundle"] = str(bundle_outdir(Path(args.outdir)))
        write_status(str(Path(args.outdir) / "status.json"), final)


if __name__ == "__main__":
    main()
