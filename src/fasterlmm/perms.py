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
from fasterlmm.progress import write_status


def perm_threshold(data: AlignedDataset,
                   p: int = 0,
                   *,
                   n_perm: int = 100,
                   seed: int = 19930909,
                   status_file: str | None = None) -> tuple[Tensor, Tensor]:
    """
    Min-F permutation null distribution for one pheno, all perms in one call
    Stack the real pheno + n_perm permutations as columns of a (N, 1+n_perm) matrix, run loco_scan once across all columns, take the per-column max F.
    Way faster than the per-perm loop because every perm gets to ride the same eigendecomp and the same GPU matmuls
    Returns (real_F (M,), perm_max_F (n_perm,)).  perm_max_F is the per-perm max F, wich corresponds to the per-perm min p (F and p are monotone-inverse).
    Compare real_F against the empirical quantile of perm_max_F to get the genome-wide threshold
    """
    Z_std = standardise_columns(data.Z)
    X = data.X
    chrom = data.chrom
    y_real = data.Y[:, p:p+1]  # (N, 1)
    N = y_real.shape[0]

    rng = np.random.default_rng(seed)
    perm_idx = np.stack([rng.permutation(N) for _ in range(n_perm)], axis=1)  # (N, n_perm)
    perm_idx_t = torch.from_numpy(perm_idx).long().to(y_real.device)
    y_perms = torch.gather(y_real.expand(N, n_perm), 0, perm_idx_t)  # (N, n_perm), shuffled y per column
    Y_all = torch.cat([y_real, y_perms], dim=1)  # (N, 1 + n_perm)

    if status_file is not None:
        write_status(status_file, {"pheno": p, "n_perm": n_perm, "state": "scanning all perms in one batch"})
    F_all = loco_scan(Z_std, X, Y_all, chrom)  # (M, 1 + n_perm)
    if status_file is not None:
        write_status(status_file, {"pheno": p, "perm_done": n_perm, "n_perm": n_perm})

    real_F = F_all[:, 0]
    perm_max_F = F_all[:, 1:].max(dim=0).values
    return real_F, perm_max_F
