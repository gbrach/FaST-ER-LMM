"""
Post-GWAS LD grouping
Per-trait connected components of the variants that pass a p-value cutoff, written as the LDGroup result column
Same grouping as addLinkageGroups.py of HaploTeam/1086YeastGenomes, with r2 computed here instead of by plink --r2
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def ld_clump(z_std: np.ndarray, chrom: np.ndarray,
             pos: np.ndarray, p: np.ndarray,
             cutoff: float, window_bp: float,
             r2_min: float) -> np.ndarray:
    """
    LD groups for one trait, returns an int32 (M,) with 0 for variants that are not grouped
    Candidates are the variants with p <= cutoff, nothing else gets a group
    Two candidates are linked when they sit on the same chromosome, no more than window_bp apart, with r2 >= r2_min
    A group is a connected component of that graph, so it can chain past the window through intermediate variants
    Ids count from 1 in order of each group's smallest p-value, ties resolve by variant order
    r2 comes from z_std, the standardised genotypes over all strains (N, M), so it does not depend on the trait
    Missing calls sit at the mean there, plink --r2 drops them pairwise, the two differ a bit when calls are missing
    """
    groups = np.zeros(len(p), dtype = np.int32)
    cand = np.flatnonzero(p <= cutoff)  # NaN p-values compare False and drop out here
    if cand.size == 0:
        return groups
    # sort by (chrom, pos) so each variant's window is a contiguous run, one key keeps chromosomes apart
    _, c_rank = np.unique(chrom[cand], return_inverse = True)
    order = np.lexsort((pos[cand], c_rank))
    cand_s = cand[order]
    key = c_rank[order].astype(np.float64) * 1e13 + pos[cand_s]
    zc = z_std[:, cand_s]
    n = z_std.shape[0]
    hi = np.searchsorted(key, key + window_bp, side = "right")
    rows, cols = [], []
    for i in range(cand_s.size):
        js = np.arange(i + 1, hi[i])
        if js.size:
            keep = js[((zc[:, js].T @ zc[:, i]) / n) ** 2 >= r2_min]
            rows.append(np.full(keep.size, i))
            cols.append(keep)
    if rows:
        r, c = np.concatenate(rows), np.concatenate(cols)
    else:
        r = c = np.empty(0, dtype = np.int64)
    k = cand_s.size
    _, comp = connected_components(coo_matrix((np.ones(r.size), (r, c)), shape = (k, k)), directed = False)
    comp_of = np.empty(len(p), dtype = np.int64)
    comp_of[cand_s] = comp  # component label per variant index
    gid = {}
    for v in cand[np.argsort(p[cand], kind = "stable")]:  # smallest p first, so id 1 holds the top hit
        gid.setdefault(comp_of[v], len(gid) + 1)
        groups[v] = gid[comp_of[v]]
    return groups
