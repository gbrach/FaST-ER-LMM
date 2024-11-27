"""
Permutation thresholds for GWAS p-values
Shuffle y against the genotypes N times, rerun LOCO, keep min(p) per shuffle.
The empirical 5% quantile of those min p-values is the genome-wide significance threshold under "no association".
Putting perms into the same pipeline  so a single fasterlmm-gwas call gives real-Fs + perm threshold all together
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from fasterlmm.core import loco_scan
from fasterlmm.io import AlignedDataset, standardise_columns
from fasterlmm.progress import pbar, write_status


def perm_threshold(data: AlignedDataset,
                   p: int = 0,
                   *,
                   n_perm: int = 100,
                   seed: int = 19930909,
                   status_file: str | None = None) -> tuple[Tensor, Tensor]:
    """
    Min-F permutation null distribution for one pheno
    Returns (real_F (M,), perm_max_F (n_perm,)).  perm_max_F is the per-perm max F, wich corresponds to the per-perm min p (F and p are monotone-inverse).
    Compare real_F against the empirical quantile of perm_max_F to get the genome-wide threshold
    """
    Z_std = standardise_columns(data.Z)
    X = data.X
    Y = data.Y
    chrom = data.chrom
    y_real = Y[:, p:p+1]

    real_F = loco_scan(Z_std, X, y_real, chrom, p=0)

    perm_max_F = torch.zeros(n_perm, dtype=Z_std.dtype)
    rng = np.random.default_rng(seed)
    N = Y.shape[0]
    for k in pbar(range(n_perm), desc=f"perms pheno {p}", total=n_perm):
        idx = rng.permutation(N)  # shuffling the strain order of y, geno + covars stay in place
        y_perm = y_real[idx]
        F_perm = loco_scan(Z_std, X, y_perm, chrom, p=0)
        perm_max_F[k] = F_perm.max()
        if status_file is not None:
            write_status(status_file, {"pheno": p, "perm_done": k + 1, "n_perm": n_perm})
    return real_F, perm_max_F
