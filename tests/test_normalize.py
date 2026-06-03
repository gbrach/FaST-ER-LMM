"""
in-process tests for fasterlmm.normalize -- the Blom rank-inverse-normal transform
covers rint_columns on 1d / 2d / nan inputs, the 3D guard, the rint alias,
and a direct check of the Blom formula against scipy.  cpu-only, portable, no
fastlmm or gpu needed, every test stays well under a second
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm, rankdata

from fasterlmm.normalize import rint, rint_columns


def test_rint_1d_sorted_unit_input():
    """a sorted no-tie vector comes back symmetric around mean 0 with std near 1"""
    rng = np.random.default_rng(19930909)
    x = np.sort(rng.normal(size=64))
    out = rint_columns(x)
    assert out.shape == x.shape
    # Blom on a contiguous rank set is symmetric so the mean lands on 0
    assert np.isclose(out.mean(), 0.0, atol=1e-12)
    # not exactly unit variance (it's a quantile transform) but close-ish for n=64
    assert 0.85 < out.std() < 1.05
    # monotone: rank order is preserved
    assert np.all(np.diff(out) > 0)


def test_rint_2d_each_column_centered():
    """each column of a 2d input is transformed independently and centers on 0"""
    rng = np.random.default_rng(424242)
    Y = rng.gamma(shape=2.0, scale=3.0, size=(50, 4))
    out = rint_columns(Y)
    assert out.shape == Y.shape
    # no ties in continuous gamma draws so every column is a clean permutation of
    # the same symmetric Blom scores -> column means all sit on 0
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-12)
    # each column independently preserves the within-column rank order
    for j in range(Y.shape[1]):
        order = np.argsort(Y[:, j])
        assert np.all(np.diff(out[order, j]) > 0)


def test_rint_nan_round_trip():
    """nan cells stay nan, the finite cells get Blom values computed on the kept rows"""
    y = np.array([3.0, np.nan, 1.0, 5.0, np.nan, 2.0, 4.0])
    out = rint_columns(y)
    nan_mask = np.isnan(y)
    # the nans round-trip untouched
    assert np.array_equal(np.isnan(out), nan_mask)
    # the finite cells are all finite Blom scores
    assert np.all(np.isfinite(out[~nan_mask]))
    # and the n used is the count of kept rows, not the full length
    finite = y[~nan_mask]
    n = finite.size
    ranks = rankdata(finite, method="average")
    expect = norm.ppf((ranks - 3 / 8) / (n - 2 * (3 / 8) + 1))
    assert np.allclose(out[~nan_mask], expect, atol=1e-12)


def test_rint_2d_nan_per_column():
    """nan masks are per-column, a nan in one column doesn't leak into another"""
    Y = np.array([
        [1.0, 10.0],
        [np.nan, 20.0],
        [3.0, np.nan],
        [4.0, 40.0],
        [5.0, 50.0],
    ])
    out = rint_columns(Y)
    # the nan positions match exactly, column by column
    assert np.isnan(out[1, 0]) and not np.isnan(out[1, 1])
    assert np.isnan(out[2, 1]) and not np.isnan(out[2, 0])
    # column 0 used n=4 kept rows, column 1 used n=4 kept rows
    for j in range(2):
        col = Y[:, j]
        keep = ~np.isnan(col)
        n = int(keep.sum())
        ranks = rankdata(col[keep], method="average")
        expect = norm.ppf((ranks - 3 / 8) / (n - 2 * (3 / 8) + 1))
        assert np.allclose(out[keep, j], expect, atol=1e-12)


def test_rint_3d_raises():
    """anything past 2d isn't a phenotype table so it should be rejected"""
    Y = np.zeros((4, 3, 2))
    with pytest.raises(ValueError):
        rint_columns(Y)


def test_rint_alias_matches_rint_columns():
    """the thin rint() alias delegates straight to rint_columns with the same ties"""
    rng = np.random.default_rng(7)
    x = rng.normal(size=40)
    a = rint(x)
    b = rint_columns(x, ties="average")
    assert np.array_equal(a, b)
    # the alias keyword name differs but the default still maps to 'average'
    assert np.array_equal(rint(x, ties_method="average"), b)


def test_rint_direct_blom_formula_no_ties():
    """direct Blom check vs scipy on a no-tie vector: qnorm((rank - 3/8) / (n - 2*3/8 + 1))"""
    rng = np.random.default_rng(101)
    # continuous draws, vanishingly unlikely to tie at float64 precision
    x = rng.uniform(size=37)
    assert np.unique(x).size == x.size  # guard: genuinely no ties
    n = x.size
    ranks = rankdata(x, method="average")
    expect = norm.ppf((ranks - 3 / 8) / (n - 2 * (3 / 8) + 1))
    out = rint_columns(x)
    assert np.allclose(out, expect, atol=1e-12)


def test_rint_custom_c_parameter():
    """passing a different c flows into the Blom plotting position"""
    rng = np.random.default_rng(55)
    x = rng.uniform(size=30)
    n = x.size
    ranks = rankdata(x, method="average")
    expect = norm.ppf((ranks - 0.5) / (n - 2 * 0.5 + 1))
    out = rint_columns(x, c=0.5)
    assert np.allclose(out, expect, atol=1e-12)
