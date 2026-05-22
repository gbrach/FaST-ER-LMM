"""
Permutation thresholds for GWAS p-values
Shuffle y against the genotypes n_perm times, rerun LOCO, keep min(p) per shuffle.
The empirical 5% quantile of those min p-values is the genome-wide significance threshold under "no association".
A batch of real phenos plus all their permutation columns ride one scan together, so the per-chromosome eigendecomposition is paid once for the whole batch rather than once per pheno -- that is where the speed comes from
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from fasterlmm.core import ScanResult, loco_scan_compat, single_k_scan_compat
from fasterlmm.io import AlignedDataset, standardise_columns


def perm_threshold(data: AlignedDataset,
                   pheno_idx: list[int],
                   *,
                   n_perm: int = 100,
                   seed: int = 19930909,
                   loco: bool = True,
                   on_chrom=None) -> tuple[ScanResult, Tensor]:
    """
    Min-F permutation null distribution for a whole batch of phenos in one scan
    Pack the B real phenos and their B * n_perm permutation columns into one (N, B + B*n_perm) matrix, run loco_scan_compat once, slice the per-pheno results back out.
    The eigendecomposition is genotype-only so it costs the same for one column or fifteen thousand -- batching is what amortizes it across phenos, not just across perms.
    Each pheno gets its own n_perm independent row-shuffles, the way fastlmm does it -- the rng is seeded per (seed, pheno index) so a pheno sees the same perms no matter which batch it lands in or how phenos-per-job is set.
    Returns (ScanResult for the B real phenos, perm_max_F (B, n_perm)).  perm_max_F is the per-perm genome max F, monotone-inverse to the per-perm min p.
    Compare the real F against the empirical quantile of perm_max_F to get the genome-wide threshold
    """
    Z_std = standardise_columns(data.Z)
    X = data.X
    chrom = data.chrom
    B = len(pheno_idx)
    y_real = data.Y[:, pheno_idx]  # (N, B)
    N = y_real.shape[0]

    # independent permutations: pheno b gets n_perm distinct shuffles of the N rows, rng seeded on
    # its own column index so its perms are stable regardless of batching.  orders[n, b, j] is the
    # row that lands in position n of pheno b's j-th shuffle
    orders = np.empty((N, B, n_perm), dtype=np.int64)
    for b_pos, p in enumerate(pheno_idx):
        rng = np.random.default_rng([seed, int(p)])
        orders[:, b_pos, :] = rng.random((n_perm, N)).argsort(axis=1).T
    gather_idx = torch.from_numpy(orders).to(y_real.device)  # (N, B, n_perm)
    y_perms = torch.gather(y_real.unsqueeze(2).expand(N, B, n_perm), 0, gather_idx)
    y_perms = y_perms.reshape(N, B * n_perm)  # pheno-major: [b0 j0..jP, b1 j0..jP, ...]
    Y_all = torch.cat([y_real, y_perms], dim=1)  # (N, B + B*n_perm)

    if loco:
        res = loco_scan_compat(Z_std, X, Y_all, chrom, n_real=B, on_chrom=on_chrom)
    else:
        res = single_k_scan_compat(Z_std, X, Y_all, n_real=B)  # one K over all SNPs

    perm_max_F = res.max_F[B:].reshape(B, n_perm)  # row b holds pheno b's per-perm genome max F
    return res, perm_max_F
