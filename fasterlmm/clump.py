"""
Post-GWAS LD clumping
Greedy per-trait grouping of the variants that pass a p-value cutoff, written as the LDGroup result column
"""

from __future__ import annotations

import numpy as np


def ld_clump(z_std: np.ndarray, chrom: np.ndarray,
             pos: np.ndarray, p: np.ndarray,
             cutoff: float, window_bp: float,
             r2_min: float) -> np.ndarray:
    """
    LD groups for one trait, returns an int32 (M,) with 0 for variants that are not grouped
    Candidates are the variants with p <= cutoff, nothing else gets a group
    Walks the candidates from the smallest p up, an unassigned variant becomes the index of a new group (ids count from 1)
    and takes every still unassigned candidate on its chromosome within window_bp with r2 >= r2_min
    r2 comes from z_std, the standardised genotypes over all strains (N, M), so it does not depend on the trait
    Ties in p resolve by variant order, so the grouping is deterministic
    """
    groups = np.zeros(len(p), dtype = np.int32)
    cand = np.flatnonzero(p <= cutoff)  # NaN p-values compare False and drop out here
    if cand.size == 0:
        return groups
    c_chrom = chrom[cand]
    c_pos = pos[cand]
    zc = z_std[:, cand]
    n = z_std.shape[0]
    free = np.ones(cand.size, dtype = bool)
    g = 0
    for i in np.argsort(p[cand], kind = "stable"):
        if not free[i]:
            continue
        g += 1
        near = np.flatnonzero(free & (c_chrom == c_chrom[i]) & (np.abs(c_pos - c_pos[i]) <= window_bp))
        near = near[near != i]
        members = near[((zc[:, near].T @ zc[:, i]) / n) ** 2 >= r2_min] if near.size else near
        free[i] = False
        free[members] = False
        groups[cand[i]] = g
        groups[cand[members]] = g
    return groups
