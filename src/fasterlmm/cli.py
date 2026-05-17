"""
fasterlmm gwas command-line entry: plink + phen (+ optional covar) -> per-pheno LOCO scan + perm threshold
--pheno-start/--pheno-end carves out a pheno range, --shard X/N is the explicit slurm-array slice. bare --device cuda fans out one worker per visble GPU.  --bundle collapses the per-pheno tsvs into a single parquet at the end
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as ss
import torch

from fasterlmm.bundle import bundle_outdir
from fasterlmm.io import align_inputs, read_covar, read_phen, read_plink
from fasterlmm.perms import perm_threshold
from fasterlmm.progress import write_status


def _parse_shard(shard: str) -> tuple[int, int]:
    """X/N -> (X, N) with bounds check.  Empty / None means no sharding"""
    i, n = shard.split("/")
    i, n = int(i), int(n)
    if not (0 <= i < n):
        raise ValueError(f"shard index {i} out of range for n={n}")
    return i, n


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

    geno = read_plink(args.geno)
    pheno = read_phen(args.pheno)
    covar = read_covar(args.covar) if args.covar else None
    data = align_inputs(geno, pheno, covar, dtype=torch.float64)
    if device != "cpu":
        data.Z = data.Z.to(device)
        data.X = data.X.to(device)
        data.Y = data.Y.to(device)

    P_total = data.Y.shape[1]
    if args.pheno_idx is not None:
        pheno_list = [args.pheno_idx]
    else:
        start = args.pheno_start or 0
        end = args.pheno_end if args.pheno_end is not None else P_total
        pheno_list = list(range(start, end))
    if shard_n is not None:
        pheno_list = pheno_list[shard_i::shard_n]  # round-robin slice so progress stays balanced acros shards

    write_status(status_file, {"state": "scanning", "N": data.Y.shape[0],
                               "M": data.Z.shape[1], "n_pheno": len(pheno_list),
                               "n_perm": args.n_perm, 
                               "shard": f"{shard_i}/{shard_n}" if shard_n else None,
                               "device": device})

    N, C = data.X.shape
    df2 = N - C - 1
    for p in pheno_list:
        pheno_name = data.pheno_names[p]
        sub = outdir / pheno_name
        sub.mkdir(parents=True, exist_ok=True)
        real_F, perm_max_F = perm_threshold(data, p=p, n_perm=args.n_perm,
                                            seed=args.seed,
                                            status_file=str(sub / "status.json"))
        real_p = ss.f.sf(real_F.cpu().numpy(), 1, df2)  # F -> p once per pheno, scipy on cpu is fine
        perm_min_p = ss.f.sf(perm_max_F.cpu().numpy(), 1, df2)
        thresh_05 = float(np.quantile(perm_min_p, 0.05))
        pd.DataFrame({"SNP": data.snp_id,
                      "Chr": data.chrom,
                      "Pos": data.pos,
                      "F": real_F.cpu().numpy(),
                      "PValue": real_p}).to_csv(sub / "gwas.tsv", sep="\t", index=False)
        pd.DataFrame({"perm_min_p": perm_min_p}).to_csv(sub / "perms.tsv", sep="\t", index=False)
        (sub / "threshold.txt").write_text(f"{thresh_05:.6e}\n")

    write_status(status_file, {"state": "done", "n_pheno": len(pheno_list),
                               "shard": f"{shard_i}/{shard_n}" if shard_n else None})


def _spawn_entry(rank: int, args: argparse.Namespace, n_gpu: int) -> None:
    """mp.spawn target, grabbing shard rank/n_gpu on cuda:rank for the rank-th worker"""
    _run_scan(args, shard_i=rank, shard_n=n_gpu, device=f"cuda:{rank}")


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
    parser.add_argument("--n-perm", type=int, default=100, help="permutation count for the threshold")
    parser.add_argument("--seed", type=int, default=19930909)
    parser.add_argument("--device", default="cuda", help="cuda (auto-dispatch across visible GPUs), cuda:N (single device), or cpu")
    parser.add_argument("--shard", default=None, help="X/N to process only the X-th of N pheno shards (explicit, e.g. slurm-array)")
    parser.add_argument("--no-multi-gpu", action="store_true", help="opt out of auto-dispatch when --device cuda sees more than one GPU")
    parser.add_argument("--bundle", action="store_true", help="after scanning, bundle per-pheno gwas.tsv into one parquet")
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
        import torch.multiprocessing as mp
        mp.spawn(_spawn_entry, args=(args, n_gpu), nprocs=n_gpu, join=True)
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
