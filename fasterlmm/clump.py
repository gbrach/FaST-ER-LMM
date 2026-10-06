"""
Post-GWAS LD grouping
Per-trait connected components of the variants that pass a p-value cutoff, written as the LDGroup result column
Same grouping as addLinkageGroups.py of HaploTeam/1086YeastGenomes, with r2 computed here instead of by plink --r2
r2 is the squared Pearson correlation over the strains where both variants are called, what plink --r2 reports
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _pair_r2(x0: np.ndarray, ox: np.ndarray,
             y0: np.ndarray, oy: np.ndarray) -> np.ndarray:
    """
    Squared Pearson correlation of one variant against a block of others over the strains called in both
    x0 (N,) and y0 (N, J) hold the genotypes with missing calls set to 0, ox and oy are the called masks
    Pairs with no variance over their shared strains get 0, so they never link
    """
    mx = ox.astype(np.float64)
    my = oy.astype(np.float64)
    n = mx @ my
    sx = x0 @ my
    sy = mx @ y0
    sxy = x0 @ y0
    sxx = (x0 * x0) @ my
    syy = mx @ (y0 * y0)
    num = n * sxy - sx * sy
    den = (n * sxx - sx * sx) * (n * syy - sy * sy)
    with np.errstate(divide = "ignore", invalid = "ignore"):
        r2 = np.where(den > 0, num * num / den, 0.0)
    return r2


def ld_clump(geno: np.ndarray, chrom: np.ndarray,
             pos: np.ndarray, p: np.ndarray,
             cutoff: float, window_bp: float,
             r2_min: float) -> np.ndarray:
    """
    LD groups for one trait, returns an int32 (M,) with 0 for variants that are not grouped
    Candidates are the variants with p <= cutoff, nothing else gets a group
    Two candidates are linked when they sit on the same chromosome, no more than window_bp apart, with r2 >= r2_min
    A group is a connected component of that graph, so it can chain past the window through intermediate variants
    Ids count from 1 in order of each group's smallest p-value, ties resolve by variant order
    r2 comes from geno, the raw genotypes over all strains (N, M) with NaN for missing calls, so it does not depend on
    the trait
    Pairwise over the strains called in both variants, like plink --r2, so missing calls do not shrink r2
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
    gc = geno[:, cand_s].astype(np.float64)
    obs = ~np.isnan(gc)
    g0 = np.where(obs, gc, 0.0)
    complete = bool(obs.all())
    if complete:  # no missing call anywhere, one standardisation then a plain dot product per pair
        sd = g0.std(axis = 0)
        zc = np.where(sd > 0, (g0 - g0.mean(axis = 0)) / np.where(sd > 0, sd, 1.0), 0.0)
    n = geno.shape[0]
    hi = np.searchsorted(key, key + window_bp, side = "right")
    rows, cols = [], []
    for i in range(cand_s.size):
        js = np.arange(i + 1, hi[i])
        if js.size:
            r2 = ((zc[:, js].T @ zc[:, i]) / n) ** 2 if complete else _pair_r2(g0[:, i], obs[:, i], g0[:, js], obs[:, js])
            keep = js[r2 >= r2_min]
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
