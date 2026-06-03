"""
unit cover for fasterlmm.extreme, the older Halko randomised path
randomized_eigh approximates the top eigenpairs of K = grm(Z), and loco_scan_extreme
runs a leave-one-chromosome-out scan off those approximate spectra.  cpu-only and
portable, no fastlmm and no gpu, small synthetic Z so it stays well under a couple secs
"""
from __future__ import annotations

import numpy as np
import torch

from fasterlmm.core import eigendecompose
from fasterlmm.extreme import loco_scan_extreme, randomized_eigh
from fasterlmm.io import grm, standardise_columns


def _toy_Z(N=50, M=200, seed=19930909):
    """small column-standardised genotype-ish matrix for the randomised tests"""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 3, size=(N, M)).astype(np.float64)
    Z = standardise_columns(torch.as_tensor(raw, dtype=torch.float64))
    return Z


def test_randomized_eigh_shapes():
    """U_top is (N, rank) and s_top is (rank,)"""
    Z = _toy_Z()
    N = Z.shape[0]
    rank = 30
    s_top, U_top = randomized_eigh(Z, rank=rank, n_iter=4)
    assert s_top.shape == (rank,)
    assert U_top.shape == (N, rank)


def test_randomized_eigh_descending():
    """s_top comes back sorted high to low"""
    Z = _toy_Z()
    s_top, _ = randomized_eigh(Z, rank=30, n_iter=4)
    diffs = s_top[:-1] - s_top[1:]
    # every consecutive gap non-negative means the run is monotone non-increasing
    assert torch.all(diffs >= -1e-10)


def test_randomized_eigh_nonneg():
    """clamped eigenvalues never go negative"""
    Z = _toy_Z()
    s_top, _ = randomized_eigh(Z, rank=30, n_iter=4)
    assert torch.all(s_top >= 0.0)


def test_randomized_eigh_near_orthonormal_columns():
    """lifted eigenvectors stay near orthonormal, U_topᵀ U_top close to identity"""
    Z = _toy_Z()
    rank = 30
    _, U_top = randomized_eigh(Z, rank=rank, n_iter=4)
    gram = U_top.T @ U_top
    eye = torch.eye(rank, dtype=U_top.dtype)
    # Q from qr is orthonormal and U_small from eigh is orthonormal, so the product is too
    assert torch.allclose(gram, eye, atol=1e-12)


def test_randomized_eigh_approximates_top_true():
    """top randomised eigenvalues track the largest true eigenvalues of grm(Z)"""
    Z = _toy_Z()
    rank = 30
    s_top, _ = randomized_eigh(Z, rank=rank, n_iter=4)
    K = grm(Z)
    s_true, _ = eigendecompose(K)  # ascending, clamped >= 0
    true_top = torch.flip(s_true, dims=[0])[:rank]  # flip to descending, take the largest rank
    # rank is close to N (30 of 50) and a few power iterations sharpen the leading block,
    # so the approximation is essentially exact here.  observed gap is ~1e-6 relative, the
    # tols below carry a comfortable margin over that.  leading chunk checked tightest
    lead = 15
    assert torch.allclose(s_top[:lead], true_top[:lead], rtol=1e-4, atol=1e-6)
    # the whole kept block matches the truth nearly as well
    assert torch.allclose(s_top, true_top, rtol=1e-3, atol=1e-5)


def test_randomized_eigh_deterministic():
    """same seed gives identical eigenpairs"""
    Z = _toy_Z()
    s1, u1 = randomized_eigh(Z, rank=30, n_iter=2, seed=19930909)
    s2, u2 = randomized_eigh(Z, rank=30, n_iter=2, seed=19930909)
    assert torch.equal(s1, s2)
    assert torch.equal(u1, u2)


def test_loco_scan_extreme_finite_shape():
    """loco_scan_extreme on a 3-chrom toy returns a finite (M, P) F grid"""
    rng = np.random.default_rng(19930909)
    N, M, P = 50, 90, 3
    raw = rng.integers(0, 3, size=(N, M)).astype(np.float64)
    Z = standardise_columns(torch.as_tensor(raw, dtype=torch.float64))
    # three fake chromosomes, 30 variants each so each leave-one-out kin block is wide enough
    chrom = np.repeat(np.array([1, 2, 3]), M // 3)
    X = torch.ones(N, 1, dtype=torch.float64)  # intercept-only design
    Y = torch.as_tensor(rng.normal(size=(N, P)), dtype=torch.float64)
    rank = 30
    f_out = loco_scan_extreme(Z, X, Y, chrom, rank=rank)
    assert f_out.shape == (M, P)
    assert torch.isfinite(f_out).all()
    # F statistics are non-negative
    assert torch.all(f_out >= 0.0)


def test_loco_scan_extreme_two_chrom():
    """two-chromosome toy also produces a finite full grid"""
    rng = np.random.default_rng(7)
    N, M, P = 40, 60, 2
    raw = rng.integers(0, 3, size=(N, M)).astype(np.float64)
    Z = standardise_columns(torch.as_tensor(raw, dtype=torch.float64))
    chrom = np.repeat(np.array([1, 2]), M // 2)
    X = torch.ones(N, 1, dtype=torch.float64)
    Y = torch.as_tensor(rng.normal(size=(N, P)), dtype=torch.float64)
    f_out = loco_scan_extreme(Z, X, Y, chrom, rank=25)
    assert f_out.shape == (M, P)
    assert torch.isfinite(f_out).all()
