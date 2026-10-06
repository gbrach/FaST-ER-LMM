"""
clump.ld_clump on small synthetic panels
covers the connected-component grouping (chains merge), the window and r2 cutoffs, chromosome boundaries,
group ids following the smallest p-value, the p cutoff leaving variants ungrouped, and determinism.
cpu-only, no fixtures
"""

from __future__ import annotations

import numpy as np

from fasterlmm.clump import ld_clump
from fasterlmm.io import standardise_columns


def _panel(n = 400, seed = 7):
    """six variants: 0 and 1 are copies of one signal, 2 is independent, 3 copies 0 but sits far away, 4 copies 0 on chr 2, 5 is a noisy copy"""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 3, size = n).astype(np.float64)
    other = rng.integers(0, 3, size = n).astype(np.float64)
    noisy = np.where(rng.random(n) < 0.1, rng.integers(0, 3, size = n), base)
    z = np.stack([base, base, other, base, base, noisy], axis = 1)
    chrom = np.array([1, 1, 1, 1, 2, 1])
    pos = np.array([1000, 20000, 30000, 900000, 1000, 40000], dtype = np.float64)
    return standardise_columns(z.copy()).astype(np.float32), chrom, pos


def test_clump_groups_by_window_r2_and_chrom():
    z, chrom, pos = _panel()
    p = np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
    g = ld_clump(z, chrom, pos, p, cutoff = 1.0, window_bp = 50_000, r2_min = 0.5)
    assert g[0] == g[1] == g[5] == 1  # signal copies within 50 kb, the noisy one still above r2 0.5
    assert g[2] == 2  # independent variant, own group
    assert g[3] == 3 and g[4] == 4  # same signal but too far, and on another chromosome


def test_clump_ids_follow_the_smallest_p_of_each_group():
    z, chrom, pos = _panel()
    p = np.array([1e-5, 1e-9, 1e-7, 1e-6, 1e-4, 1e-3])
    g = ld_clump(z, chrom, pos, p, cutoff = 1.0, window_bp = 50_000, r2_min = 0.5)
    assert g[1] == 1 and g[0] == 1  # the group holding the top hit is group 1
    assert g[2] == 2 and g[3] == 3 and g[4] == 4


def test_clump_cutoff_leaves_the_rest_ungrouped():
    z, chrom, pos = _panel()
    p = np.array([1e-9, 1e-8, 0.5, 0.9, np.nan, 1e-4])
    g = ld_clump(z, chrom, pos, p, cutoff = 1e-6, window_bp = 50_000, r2_min = 0.5)
    assert list(g) == [1, 1, 0, 0, 0, 0]


def test_clump_wider_window_merges_far_copy_and_r2_gate_splits():
    z, chrom, pos = _panel()
    p = np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
    wide = ld_clump(z, chrom, pos, p, cutoff = 1.0, window_bp = 1_000_000, r2_min = 0.5)
    assert wide[3] == wide[0]
    strict = ld_clump(z, chrom, pos, p, cutoff = 1.0, window_bp = 50_000, r2_min = 0.999)
    assert strict[5] != strict[0] and strict[1] == strict[0]


def test_clump_deterministic_and_empty():
    z, chrom, pos = _panel()
    p = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])  # all tied, variant order breaks the ties
    a = ld_clump(z, chrom, pos, p, cutoff = 1.0, window_bp = 50_000, r2_min = 0.5)
    b = ld_clump(z, chrom, pos, p, cutoff = 1.0, window_bp = 50_000, r2_min = 0.5)
    assert np.array_equal(a, b) and a[0] == 1
    none = ld_clump(z, chrom, pos, p, cutoff = 1e-9, window_bp = 50_000, r2_min = 0.5)
    assert not none.any()


def test_clump_chains_through_an_intermediate_variant():
    """a ~ b and b ~ c but a !~ c, all inside the window: connected components put the three in one group"""
    rng = np.random.default_rng(3)
    n = 6000
    x, y = rng.standard_normal(n), rng.standard_normal(n)
    z = np.stack([x, (x + y) / np.sqrt(2), y], axis = 1).astype(np.float32)
    z = standardise_columns(z.copy()).astype(np.float32)
    chrom = np.array([1, 1, 1])
    pos = np.array([1000.0, 2000.0, 3000.0])
    p = np.array([1e-9, 1e-8, 1e-7])
    g = ld_clump(z, chrom, pos, p, cutoff = 1.0, window_bp = 50_000, r2_min = 0.35)
    assert list(g) == [1, 1, 1]
    far = ld_clump(z, chrom, np.array([1000.0, 2000.0, 4000.0]), p, cutoff = 1.0, window_bp = 1_500, r2_min = 0.35)
    assert far[0] == far[1] and far[2] != far[1]  # c is out of reach of b, the chain stops


def test_clump_single_candidate_is_group_one():
    z, chrom, pos = _panel()
    p = np.array([0.5, 0.5, 1e-9, 0.5, 0.5, 0.5])
    g = ld_clump(z, chrom, pos, p, cutoff = 1e-6, window_bp = 50_000, r2_min = 0.5)
    assert list(g) == [0, 0, 1, 0, 0, 0]


def test_clump_matches_a_brute_force_graph_on_a_blocky_panel():
    """same partition as the full pairwise graph (same chrom, distance and r2 cutoffs) built in one shot, the way addLinkageGroups.py does"""
    from scipy.sparse.csgraph import connected_components
    rng = np.random.default_rng(11)
    n, m = 500, 240
    latent = rng.standard_normal((n, m // 6))
    z = np.repeat(latent, 6, axis = 1) + 0.9 * rng.standard_normal((n, m))  # blocks of 6 correlated variants
    z = standardise_columns(z.copy()).astype(np.float32)
    chrom = np.repeat([1, 2, 3], m // 3)
    pos = np.tile(np.arange(m // 3) * 5_000.0, 3)
    p = rng.random(m) ** 4
    cutoff, window, r2_min = 0.3, 30_000.0, 0.15
    got = ld_clump(z, chrom, pos, p, cutoff = cutoff, window_bp = window, r2_min = r2_min)
    cand = np.flatnonzero(p <= cutoff)
    r2 = (z[:, cand].T @ z[:, cand] / n) ** 2
    adj = (r2 >= r2_min) & (chrom[cand][:, None] == chrom[cand][None, :]) & (np.abs(pos[cand][:, None] - pos[cand][None, :]) <= window)
    np.fill_diagonal(adj, False)
    _, lab = connected_components(adj, directed = False)
    assert (got[np.setdiff1d(np.arange(m), cand)] == 0).all()
    mine = {}
    for v, g in zip(cand, got[cand]):
        mine.setdefault(int(g), set()).add(int(v))
    ref = {}
    for v, l in zip(cand, lab):
        ref.setdefault(int(l), set()).add(int(v))
    assert sorted(map(sorted, mine.values())) == sorted(map(sorted, ref.values()))
    assert len(ref) > 5 and max(len(s) for s in ref.values()) > 1  # the panel has real multi-variant groups
