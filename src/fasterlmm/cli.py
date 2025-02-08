"""
fasterlmm-gwas command-line entry: plink + phen (+ optional covar) -> LOCO scan + perm threshold
One pheno per call for now, batchin across phenos will come in a later commit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as ss
import torch

from fasterlmm.io import align_inputs, read_covar, read_phen, read_plink
from fasterlmm.perms import perm_threshold
from fasterlmm.progress import write_status


def main() -> None:
    parser = argparse.ArgumentParser(prog="fasterlmm gwas", description="torch port of fastlmm GWAS with LOCO + perm threshold")
    parser.add_argument("--geno", required=True, help="plink BED prefix")
    parser.add_argument("--pheno", required=True, help="wide phen tsv with Strain column")
    parser.add_argument("--covar", default=None, help="plink-style .cov (optional)")
    parser.add_argument("--outdir", required=True, help="output dir, will be created if missing")
    parser.add_argument("--pheno-idx", type=int, default=0, help="0-based pheno column to scan")
    parser.add_argument("--n-perm", type=int, default=100, help="permutation count for the threshold")
    parser.add_argument("--seed", type=int, default=19930909)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu (cpu is mostly for tiny sanity checks)")
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

    write_status(status_file, {"state": "scanning", "N": data.Y.shape[0],
                               "M": data.Z.shape[1], "pheno_idx": args.pheno_idx,
                               "n_perm": args.n_perm})
    real_F, perm_max_F = perm_threshold(data,
                                        p=args.pheno_idx,
                                        n_perm=args.n_perm,
                                        seed=args.seed,
                                        status_file=status_file)

    # turning F-stats into p-values once at the very end, scpiy on cpu is fine for a single batch
    N, C = data.X.shape
    df2 = N - C - 1
    real_p = ss.f.sf(real_F.cpu().numpy(), 1, df2)
    perm_min_p = ss.f.sf(perm_max_F.cpu().numpy(), 1, df2)
    thresh_05 = float(np.quantile(perm_min_p, 0.05))

    df = pd.DataFrame({"SNP": data.snp_id, "Chr": data.chrom, "Pos": data.pos,
                       "F": real_F.cpu().numpy(), "PValue": real_p})
    df.to_csv(outdir / "gwas.tsv", sep="\t", index=False)
    pd.DataFrame({"perm_min_p": perm_min_p}).to_csv(outdir / "perms.tsv", sep="\t", index=False)
    (outdir / "threshold.txt").write_text(f"{thresh_05:.6e}\n")

    n_sig = int((real_p < thresh_05).sum())
    write_status(status_file, {"state": "done", "thresh_05": thresh_05, "n_signif": n_sig})


if __name__ == "__main__":
    main()
