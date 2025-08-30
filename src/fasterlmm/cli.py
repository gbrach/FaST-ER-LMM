"""
fasterlmm gwas command-line entry: plink + phen (+ optional covar) -> per-pheno LOCO scan + perm threshold
Supports a pheno range (--pheno-start / --pheno-end) as well as multi-GPU sharding (--shard X/N), with an optional one-file parquet output (--bundle)
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
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu (cpu is mostly for tiny sanity checks)")
    parser.add_argument("--shard", default=None, help="X/N to process only the X-th of N pheno shards (for slurm-style multi-GPU)")
    parser.add_argument("--bundle", action="store_true", help="after scanning, bundle per-pheno gwas.tsv into one parquet")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    status_file = str(outdir / "status.json")
    write_status(status_file, {"state": "loading", "geno": args.geno, "pheno": args.pheno})

    geno = read_plink(args.geno)
    pheno = read_phen(args.pheno)
    covar = read_covar(args.covar) if args.covar else None
    data = align_inputs(geno, pheno, covar, dtype=torch.float64)
    if args.device != "cpu":
        data.Z = data.Z.to(args.device)
        data.X = data.X.to(args.device)
        data.Y = data.Y.to(args.device)

    P_total = data.Y.shape[1]
    if args.pheno_idx is not None:
        pheno_list = [args.pheno_idx]
    else:
        start = args.pheno_start or 0
        end = args.pheno_end if args.pheno_end is not None else P_total
        pheno_list = list(range(start, end))
    if args.shard:
        shard_i, shard_n = _parse_shard(args.shard)
        pheno_list = pheno_list[shard_i::shard_n]  # round-robin slice so progress stays balanced acros shards

    write_status(status_file, {"state": "scanning",
                               "N": data.Y.shape[0],
                               "M": data.Z.shape[1],
                               "n_pheno": len(pheno_list),
                               "n_perm": args.n_perm,
                               "shard": args.shard})

    N, C = data.X.shape
    df2 = N - C - 1
    for p in pheno_list:
        pheno_name = data.pheno_names[p]
        sub = outdir / pheno_name
        sub.mkdir(parents=True, exist_ok=True)
        real_F, perm_max_F = perm_threshold(data,
                                            p=p,
                                            n_perm=args.n_perm,
                                            seed=args.seed,
                                            status_file=str(sub / "status.json"))
        real_p = ss.f.sf(real_F.cpu().numpy(), 1, df2)  # F -> p once per pheno, scpiy on cpu is fine
        perm_min_p = ss.f.sf(perm_max_F.cpu().numpy(), 1, df2)
        thresh_05 = float(np.quantile(perm_min_p, 0.05))
        pd.DataFrame({"SNP": data.snp_id,
                      "Chr": data.chrom,
                      "Pos": data.pos,
                      "F": real_F.cpu().numpy(),
                      "PValue": real_p}).to_csv(sub / "gwas.tsv", sep="\t", index=False)
        pd.DataFrame({"perm_min_p": perm_min_p}).to_csv(sub / "perms.tsv", sep="\t", index=False)
        (sub / "threshold.txt").write_text(f"{thresh_05:.6e}\n")

    if args.bundle:
        path = bundle_outdir(outdir)
        write_status(status_file, {"state": "done", "bundle": str(path)})
    else:
        write_status(status_file, {"state": "done", "n_pheno": len(pheno_list)})


if __name__ == "__main__":
    main()
