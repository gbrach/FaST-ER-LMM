"""
Genomic inflation factor (lambda GC) per phenotype, same definition as HaploTeam/1086YeastGenomes GWAS/src/calc_GIF.R
NaN p-values dropped, chisq = qchisq(1 - p, 1), lambda = median(chisq) / qchisq(0.5, 1)
Shard tasks write lambda_gc.shardX.tsv, the dispatch parent (or fasterlmm concat) folds them into lambda_gc.tsv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

LAMBDA_FILENAME = "lambda_gc.tsv"
LAMBDA_COLUMNS = ["Pheno", "PhenoIndex", "LambdaGC", "NVariants"]


def lambda_gc(p: np.ndarray) -> tuple[float, int]:
    """
    Lambda GC from a vector of association p-values, plus the number of p-values used
    isf(p) is qchisq(1 - p) without the 1 - p rounding, the median sits mid-distribution so both agree
    """
    p = np.asarray(p, dtype = np.float64)
    p = p[~np.isnan(p)]
    if p.size == 0:
        return float("nan"), 0
    return float(np.median(chi2.isf(p, 1)) / chi2.ppf(0.5, 1)), int(p.size)


def write_shard_table(outdir: str | Path, shard_i: int | None, rows: list, name_to_idx: dict) -> Path:
    """
    Write this process's rows (pheno, lambda, n variants) in pheno column order
    lambda_gc.tsv for an unsharded run, lambda_gc.shard{i}.tsv otherwise
    """
    outdir = Path(outdir)
    path = outdir / (LAMBDA_FILENAME if shard_i is None else f"lambda_gc.shard{shard_i}.tsv")
    df = pd.DataFrame([(n, name_to_idx[n], lam, nv) for n, lam, nv in rows], columns = LAMBDA_COLUMNS)
    df.sort_values("PhenoIndex").to_csv(path, sep = "\t", index = False, float_format = "%.6f")
    return path


def merge_shard_tables(outdir: str | Path) -> Path | None:
    """
    Fold lambda_gc.shard*.tsv into lambda_gc.tsv in pheno column order and remove the parts
    Returns None when there are no parts, so a second call leaves the merged file alone
    """
    outdir = Path(outdir)
    parts = sorted(outdir.glob("lambda_gc.shard*.tsv"))
    if not parts:
        return None
    df = pd.concat([pd.read_csv(p, sep = "\t") for p in parts]).sort_values("PhenoIndex")
    path = outdir / LAMBDA_FILENAME
    df.to_csv(path, sep = "\t", index = False, float_format = "%.6f")
    for p in parts:
        p.unlink()
    return path
