"""
fasterlmm gwas command-line entry: plink + phen (+ optional covar) -> per-pheno LOCO scan + perm threshold
--pheno-start/--pheno-end carves out a pheno range, --shard X/N is the explicit slurm-array slice. bare --device cuda dispatches one worker per visble GPU.  --bundle streams the per-pheno tables into a gwas_bundle.parquet dataset, --no-per-pheno-dirs skips the tree and keeps only that bundle
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# metal has no float64 and a fistful of linalg ops (eigh above all) just arent
# wired up for mps, so letting torch quietly bounce those back to cpu instead
# of hard-erroring partway through a scan.  inert on cuda / cpu boxes so it
# costs nothing to set always, just has to land before torch gets imported
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import scipy.stats as ss
import torch

from fasterlmm.bundle import BUNDLE_FILENAME, BUNDLE_PARTS_DIRNAME, BundleWriter, merge_bundle_parts
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


def _default_write_workers(n_procs: int) -> int:
    """
    Pick a writer-pool size that fills the core allocation when --write-workers is left unset
    Writer threads compress with the gil released, so they scale with cores -- but every scan process sharing the node runs its own pool, so the cores have to be split n_procs ways or the pools just oversubscribe each other.  sched_getaffinity is the cgroup-allocated core set under srun, the real budget rather than the whole node.  n_procs is the gpu worker count for an auto-dispatch run, 1 for a single device or a slurm-array task that owns its own srun allocation
    """
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # not linux, sched_getaffinity is missing on mac
        cores = os.cpu_count() or 1
    return max(1, cores // max(1, n_procs))


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


# csv writes go through pyarrow, not pandas -- write_csv is a C++ routine that drops the GIL,
# so the writer pool can be plain threads.  tab delimiter + no quoting to match fastlmm's bare
# tsv (genomics ids never carry a tab, so quoting-none is safe and keeps the file byte-clean)
_TSV_WRITE_OPTS = pacsv.WriteOptions(delimiter="\t", include_header=False, quoting_style="none")


def _write_tsv(table: pa.Table, path) -> None:
    """
    Bare unquoted tsv, the way fastlmm writes it
    pyarrow quotes the header line even at quoting_style none, so the header is written by hand and the table body streamed through write_csv right underneath it
    """
    with open(path, "wb") as fh:
        fh.write(("\t".join(table.column_names) + "\n").encode())
        pacsv.write_csv(table, fh, write_options=_TSV_WRITE_OPTS)


def _write_pheno(ctx: dict, outdir_str: str, pheno_name: str, p_col,
                 beta_col, se_col, sfve_col, nullh2_col, p_var: float,
                 perm_min_p, perm_quantile: float, per_pheno_dirs: bool,
                 bundle_writer) -> None:
    """
    Writer-pool worker: one pheno's output
    Builds the 16-column table once (the 14 fastlmm single_snp columns plus threshold + significant) and routes it two ways, either or both can be on at once.  With per_pheno_dirs the pheno gets its own dir -- gwas.tsv (the bare 14-column fastlmm schema, so it drops straight into a fastlmm-shaped pipeline), perms.tsv and threshold.txt.  With a bundle_writer the full 16-column table is appended to the streaming parquet, one row group per pheno
    Table build + sort + write all go trough pyarrow, its csv writer runs in C++ and drops the GIL so the writer threads genuinely overlap the next batch's gpu scan.  ctx carries everything identical across phenos (the id columns, the filler columns, the dictionary index for Pheno) so each call only assembles the handful of columns that change
    """
    thresh = float(np.quantile(perm_min_p, perm_quantile))
    # EffectSize = beta^2 * var(genotype) / var(pheno), fastlmm single_snp.py:1454
    effect_size = beta_col * beta_col * ctx["g_var"] / p_var
    # Pheno is one repeated string -- dictionary-encode it so the table build stays O(1) on it
    # instead of materializing M python strings on the writer thread
    cols = {"sid_index": ctx["sid_index"],
            "SNP": ctx["snp_id"],
            "Chr": ctx["chrom"],
            "GenDist": ctx["gendist"],
            "ChrPos": ctx["pos"],
            "PValue": p_col,
            "SnpWeight": beta_col,
            "SnpWeightSE": se_col,
            "EffectSize": effect_size,
            "SnpFractVarExpl": sfve_col,
            "Mixing": ctx["mixing"],
            "Nullh2": nullh2_col,
            "Pheno": pa.DictionaryArray.from_arrays(ctx["pheno_idx"], [pheno_name]),
            "PhenoCount": ctx["phenocount"],
            "threshold": np.full(len(p_col), thresh),
            "significant": p_col < thresh}
    table = pa.table(cols).sort_by([("PValue", "ascending")])
    if per_pheno_dirs:
        sub = Path(outdir_str) / pheno_name
        sub.mkdir(parents=True, exist_ok=True)
        # gwas.tsv stays the bare 14-column fastlmm schema -- threshold + significance keep out of it,
        # they live in threshold.txt next to it instead
        _write_tsv(table.select(list(cols)[:-2]), sub / "gwas.tsv")
        _write_tsv(pa.table({"perm_min_p": perm_min_p}), sub / "perms.tsv")
        (sub / "threshold.txt").write_text(f"{thresh:.6e}\n")
    if bundle_writer is not None:
        # appended straight into the streaming bundle, one row group, no per-pheno parquet on disk
        bundle_writer.append(table)


def _drain_done(futures: list) -> list:
    """dropping finished write futures (re-raising any that failed), returns the still-pending ones so the pending list and the pheno arrays it pins stay bounded"""
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

    # writer pool is plain threads -- pyarrow's csv writer drops the GIL, so the threads overlap
    # the gpu scan without the fork-a-clean-process dance a ProcessPoolExecutor would need.  the
    # per-run constants ride writer_ctx, shared straight by reference (no pickling between threads)
    writer_ctx: dict = {}
    writer_pool = None
    bundle_writer = None
    if not args.dry_run:
        M = data.Z.shape[1]
        # writer_ctx holds everything that's identical across phenos -- the id columns, the filler
        # columns (GenDist all-null, Mixing all-zero, PhenoCount constant), the dictionary index
        # for Pheno -- all built once so each _write_pheno only assembles the few columns that change
        # g_var comes off cpu Z here, before Z moves to the device, nan-aware since raw Z keeps missing calls
        writer_ctx = {"snp_id": pa.array(data.snp_id, type=pa.string()),
                      "chrom": data.chrom, "pos": data.pos,
                      "g_var": np.nanvar(data.Z.cpu().numpy(), axis=0),
                      "sid_index": np.arange(M, dtype=np.int64),
                      "gendist": pa.nulls(M, pa.float64()),
                      "mixing": np.zeros(M),
                      "phenocount": np.full(M, 1 + args.n_perm),
                      "pheno_idx": pa.array(np.zeros(M, dtype=np.int32))}
        # one bundle writer per scan process -- a sharded run streams to .bundle_parts/shard{i}.parquet
        # and the parent concats them, an unsharded run streams straight to the final bundle
        if args.bundle:
            if shard_i is not None:
                bundle_path = outdir / BUNDLE_PARTS_DIRNAME / f"shard{shard_i}.parquet"
            else:
                bundle_path = outdir / BUNDLE_FILENAME
            bundle_writer = BundleWriter(bundle_path)
        writer_pool = ThreadPoolExecutor(max_workers=args.write_workers)

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
    # each shard counts its own batches from a local 0, so without an offset both workers print
    # "batch 0..256".  display_offset is the count of phenos the lower-ranked shards carry, so the
    # printed ranges tile [0, P) -- a runing global count, not the real interleaved pheno indices
    display_offset = 0
    if shard_n is not None:
        display_offset = sum(len(pheno_list[k::shard_n]) for k in range(shard_i))
        pheno_list = pheno_list[shard_i::shard_n]  # round-robin slice so progress stays balanced acros shards

    shard_str = f"{shard_i}/{shard_n}" if shard_n else None
    started_at = time.time()
    # one leave-one-out fold per distinct chromosome -- the watcher draws that sweep as a dot strip,
    # so chroms_total has to ride status_base and reach every snapshot.  None when loco is off
    n_chroms = int(np.unique(data.chrom).size)
    # status_base carries every field the watcher needs. write_status overwrites the file on each call, so each write has to be self-contained -- spreading status_base back in keeps that true
    status_base = {"state": "scanning", "N": data.Y.shape[0], "M": data.Z.shape[1],
                   "n_perm": args.n_perm, "shard": shard_str, "device": device,
                   "dtype": str(dtype).replace("torch.", ""), "loco": args.loco,
                   "phenos_total": len(pheno_list), "pid": os.getpid(),
                   "started_at": started_at,
                   "chroms_total": n_chroms if args.loco else None}
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
                # mid-batch heartbeat -- credit the running batch fraction by fraction so the
                # watcher bar still crawls forward through a long scan instead of jumping per
                # batch.  chroms_done drives the loco dot strip, chroms_total rides status_base.
                # writes_pending counts the not-yet-done futures, not the raw list length -- the
                # list only gets pruned by _drain_done at batch end, so a plain len() would sit
                # frozen for a whole batch and the watcher would look like the writes had stalled
                pending = sum(1 for f in write_futures if not f.done())
                write_status(status_file, {**status_base, "phenos_done": _done + _b * k / n,
                                           "elapsed_s": time.time() - started_at,
                                           "chroms_done": k, "writes_pending": pending,
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
            # scan while these csv writes run, and within a batch the writes spread over workers.
            # the result columns go by keyword, five look-alike float arrays are easy to transpose
            for j, p in enumerate(batch):
                write_futures.append(writer_pool.submit(
                    _write_pheno, writer_ctx, str(outdir), data.pheno_names[p],
                    p_col=p_real[:, j], beta_col=beta_np[:, j], se_col=se_np[:, j],
                    sfve_col=sfve_np[:, j], nullh2_col=nullh2_np[:, j],
                    p_var=float(p_var[j]), perm_min_p=perm_min_p[j],
                    perm_quantile=args.perm_quantile, per_pheno_dirs=args.per_pheno_dirs,
                    bundle_writer=bundle_writer))
            write_futures = _drain_done(write_futures)

            done += B
            print(f"{log_prefix}batch {display_offset + b_start}.."
                  f"{display_offset + b_start + B}: {B} phenos scanned in "
                  f"{time.time() - t_batch:.1f}s, {len(write_futures)} writes pending",
                  file=sys.stderr, flush=True)
            # per-shard progress for the watcher -- reporting only, the scan numbers above are untouched.
            # chroms_done = n_chroms so between batches the loco strip reads full rather than blinking
            # out, the batch just ran every fold and the next one hasnt picked up yet
            write_status(status_file, {**status_base, "phenos_done": done,
                                       "elapsed_s": time.time() - started_at,
                                       "chroms_done": n_chroms,
                                       "writes_pending": len(write_futures),
                                       **_resource_stats(device)})

        # gpu scan done -- now wait out the trailing per-pheno writes.  with a few thousand phenos
        # still draining trough the writer pool this runs a good while past the last batch, and with
        # no heartbeat here the log and the watcher both look frozen.  so tick an update every couple
        # seconds as the write futures land -- no fresh scan line, but a sign the writes still move.
        # each beat reports cpu-time and bundle bytes averaged over the whole drain so far: cpu-time
        # over wall-time under one full core, with writes still pending, means the drain is parked
        # waiting on the filesystem rather than compute-bound on the snappy compression
        n_writes = len(write_futures)
        drained = 0
        mb_s = cpu_frac = 0.0
        drain_t0 = last_beat = time.time()
        drain_cpu0 = sum(os.times()[:2])
        drain_bytes0 = bundle_writer.bytes_on_disk() if bundle_writer is not None else 0
        for f in as_completed(write_futures):
            f.result()  # surfaces any exception a writer thread hit
            drained += 1
            now = time.time()
            if now - last_beat >= 2.0 or drained == n_writes:
                last_beat = now
                # rates run cumulative from the drain start, not a 2s window -- the writer pool keeps
                # pace now so the drain is usually just the last batch, and a windowed sample over a
                # sub-2s drain is pure noise.  averaging the whole drain gives one honest figure
                # however short it runs, and the final beat can never reprint a stale window
                wall = now - drain_t0
                if wall >= 0.5:
                    cpu_frac = (sum(os.times()[:2]) - drain_cpu0) / wall
                    if bundle_writer is not None:
                        mb_s = (bundle_writer.bytes_on_disk() - drain_bytes0) / 1e6 / wall
                beat = {**status_base, "state": "writing", "phenos_done": done,
                        "elapsed_s": now - started_at,
                        "writes_pending": n_writes - drained,
                        "write_cpu_frac": cpu_frac, **_resource_stats(device)}
                if bundle_writer is not None:
                    beat["write_mb_s"] = mb_s
                write_status(status_file, beat)
                rate_str = f", {mb_s:.0f} MB/s" if bundle_writer is not None else ""
                print(f"{log_prefix}draining writes: {drained}/{n_writes} done, "
                      f"{n_writes - drained} still pending, cpu {cpu_frac * 100:.0f}%{rate_str}",
                      file=sys.stderr, flush=True)
    finally:
        writer_pool.shutdown(wait=True)
    # every pheno is written -- close the streamed bundle so its .tmp gets renamed into place
    if bundle_writer is not None:
        bundle_writer.close()

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
    parser.add_argument("--write-workers", type=int, default=None,
                        help="writer threads for the per-pheno output, which run off the gpu thread so the next batch can scan while this one writes. left unset it fills the core allocation, the cores this job can see split across the gpu workers sharing the node")
    parser.add_argument("--rint", action=argparse.BooleanOptionalAction, default=True,
                        help="Blom rank-based inverse normal transform on each pheno column, ON by default (use --no-rint to disable)")
    parser.add_argument("--seed", type=int, default=19930909)
    parser.add_argument("--device", default="cuda", help="cuda (auto-dispatch across visible GPUs), cuda:N (single device), mps (apple silicon gpu, runs float32), or cpu")
    parser.add_argument("--shard", default=None, help="X/N to process only the X-th of N pheno shards (explicit, e.g. slurm-array)")
    parser.add_argument("--no-multi-gpu", action="store_true", help="opt out of auto-dispatch when --device cuda sees more than one GPU")
    parser.add_argument("--bundle", action="store_true", help="after scanning, bundle the per-pheno results into one parquet")
    parser.add_argument("--no-per-pheno-dirs", dest="per_pheno_dirs", action="store_false", default=True,
                        help="skip the per-pheno output tree, write only the bundle parquet (needs --bundle)")
    parser.add_argument("--dry-run", action="store_true",
                        help="load inputs, print the planned work (N/M/P, n_perm, shards, device), and exit before scanning")
    args = parser.parse_args()
    if not args.per_pheno_dirs and not args.bundle:
        parser.error("--no-per-pheno-dirs needs --bundle, otherwise nothing gets written")

    # a stale .bundle_parts from an earlier run would get swept into this run's bundle, so clear it
    # up front -- only the orchestrator does this, never a --shard array task (they'd race)
    if args.bundle and args.shard is None:
        shutil.rmtree(Path(args.outdir) / BUNDLE_PARTS_DIRNAME, ignore_errors=True)

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
        # resolve the writer pool here so every shard inherits the same size -- n_gpu workers share
        # this srun task's cores, so each pool gets a 1/n_gpu slice
        if args.write_workers is None:
            args.write_workers = _default_write_workers(n_gpu)
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
        # one scan process here -- a single device or a slurm-array task owning its own srun cores,
        # so the writer pool gets the whole allocation, no divisor
        if args.write_workers is None:
            args.write_workers = _default_write_workers(1)
        _run_scan(args, shard_i=shard_i, shard_n=shard_n, device=device)

    if args.shard is None:
        final: dict = {"state": "done", "n_gpu": n_gpu if auto_dispatch else 1}
        if args.bundle:
            outdir = Path(args.outdir)
            parts = outdir / BUNDLE_PARTS_DIRNAME
            shards = sorted(parts.glob("shard*.parquet")) if parts.is_dir() else []
            if shards:
                # every worker has exited by the time we land here, so the gpus are already free --
                # and merge_bundle_parts is just a handful of same-filesystem renames, instant, no
                # gpu touched.  so the dispatch parent folds the merge in right here, no second
                # command to run.  a --shard slurm array has no parent process to do this, those
                # still finish with a separate fasterlmm concat (see the README slurm-array example)
                try:
                    bundle_path = merge_bundle_parts(outdir)
                except Exception as exc:
                    # the scan is done and every shard part is safe under .bundle_parts/, only the
                    # final gather tripped -- so dont dump a traceback over a run that actually
                    # worked, write the status and say how to finish the gather by hand
                    write_status(str(outdir / "status.json"),
                                 {"state": "done", "n_gpu": n_gpu, "bundle": None,
                                  "merge_error": str(exc)})
                    raise SystemExit(
                        f"scan finished, but gathering the {len(shards)} bundle shards failed: "
                        f"{exc}\nthe parts are intact under {parts}, run "
                        f"`fasterlmm concat {args.outdir}` to gather them") from None
                final["bundle"] = str(bundle_path)
                print(f"\n{len(shards)} shards gathered into {bundle_path}", flush=True)
            else:
                # single-worker run streamed straight to the final bundle, nothing left to do
                final["bundle"] = str(outdir / BUNDLE_FILENAME)
        write_status(str(Path(args.outdir) / "status.json"), final)


if __name__ == "__main__":
    main()
