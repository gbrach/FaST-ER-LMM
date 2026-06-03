"""
cpu unit tests for the exact low-rank spectral path in fasterlmm.lowrank
the whole point of this file is the cross-check: the low-rank decomposition never forms K, it folds the N - k null directions into an analytic off-rank term, and that has to land on the same F / beta / se as the full-rank compat path does on the same kinship.  everything here runs at float64 on small toy panels, no gpu, no fastlmm, portable on a fresh clone
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fasterlmm.core import (
    fastlmm_compat_rotate,
    fit_delta_grid_compat,
    snp_wald_scan_compat,
)
from fasterlmm.io import grm, standardise_columns
from fasterlmm.lowrank import (
    fit_delta_grid_lowrank,
    loco_scan_lowrank,
    lowrank_basis,
    lowrank_rotate,
    rotate_phenos,
    single_k_scan_lowrank,
    snp_wald_scan_lowrank,
)

DTYPE = torch.float64


def _toy(N=60, Mkin=40, Mtest=30, P=4, Dcov=2, seed=19930909):
    """toy panel: G pruned kinship markers, Z test variants, X = intercept + covars, Y with a planted Z effect plus a polygenic part, all column-standardised float64"""
    rng = np.random.default_rng(seed)
    G = rng.binomial(2, 0.3, size=(N, Mkin)).astype(np.float64)
    Z = rng.binomial(2, 0.25, size=(N, Mtest)).astype(np.float64)
    cov = rng.standard_normal((N, Dcov))
    X = np.concatenate([np.ones((N, 1)), cov], axis=1)
    g_std = standardise_columns(G)
    poly = g_std @ rng.standard_normal((Mkin, P)) / np.sqrt(Mkin)  # kinship-shaped polygenic part
    Y = poly + 0.4 * Z[:, [0]] * rng.standard_normal((1, P)) + rng.standard_normal((N, P))
    Y += X @ rng.standard_normal((X.shape[1], P))  # a covariate effect both paths should regress out
    return (torch.tensor(standardise_columns(G), dtype=DTYPE),
            torch.tensor(standardise_columns(Z), dtype=DTYPE),
            torch.tensor(X, dtype=DTYPE),
            torch.tensor(Y, dtype=DTYPE))


def test_lowrank_basis_k_lt_N_shapes():
    """k < N branch: s has one entry per kept singular value, U_eff is (N, k_kept), Neff = N - D not the rank"""
    G, Z, X, Y = _toy()
    N, Mkin = G.shape
    D = X.shape[1]
    basis = lowrank_basis(G, X)
    k_kept = basis.s.shape[0]
    assert k_kept <= Mkin  # at most one kept dim per pruned marker
    assert basis.U_eff.shape == (N, k_kept)
    assert basis.Neff == N - D  # reduced sample size, deliberately NOT the rank k
    assert torch.isfinite(basis.s).all()
    assert (basis.s >= 0).all()  # eigenvalues of a gram are non-negative


def test_lowrank_basis_U_eff_orthonormal():
    """the kept left singular vectors form an orthonormal basis, U_effᵀ U_eff = I"""
    G, Z, X, Y = _toy()
    basis = lowrank_basis(G, X)
    gram = basis.U_eff.T @ basis.U_eff
    eye = torch.eye(gram.shape[0], dtype=DTYPE)
    assert (gram - eye).abs().max().item() < 1e-10


def test_lowrank_basis_k_ge_N_branch():
    """k >= N flips to eighing the N-by-N kernel, still a valid orthonormal basis with Neff = N - D"""
    rng = np.random.default_rng(7)
    Nw, Mw, Dcov = 30, 45, 2  # more markers than strains so k_in >= N
    G = torch.tensor(standardise_columns(rng.binomial(2, 0.3, size=(Nw, Mw)).astype(np.float64)),
                     dtype=DTYPE)
    X = torch.tensor(np.concatenate([np.ones((Nw, 1)), rng.standard_normal((Nw, Dcov))], axis=1),
                     dtype=DTYPE)
    basis = lowrank_basis(G, X)
    D = X.shape[1]
    assert basis.Neff == Nw - D
    assert basis.U_eff.shape[0] == Nw
    assert basis.U_eff.shape[1] == basis.s.shape[0]
    # the OLS regress-out leaves a near-zero null space the sv floor trims, so the kept count is below the full N
    assert 0 < basis.s.shape[0] < Nw
    gram = basis.U_eff.T @ basis.U_eff
    eye = torch.eye(gram.shape[0], dtype=DTYPE)
    assert (gram - eye).abs().max().item() < 1e-10
    assert (basis.s >= 0).all()


def test_rotate_phenos_reconstruction():
    """U_eff @ UY + UUY reconstructs the OLS-regressed phenos Y - X pinv(X) Y exactly"""
    G, Z, X, Y = _toy()
    basis = lowrank_basis(G, X)
    UY, UUY = rotate_phenos(basis, Y)
    assert UY.shape == (basis.s.shape[0], Y.shape[1])
    assert UUY.shape == (Y.shape[0], Y.shape[1])
    Y_r = Y - X @ (torch.linalg.pinv(X) @ Y)
    recon = basis.U_eff @ UY + UUY
    assert (recon - Y_r).abs().max().item() < 1e-12


def test_rotate_phenos_off_rank_orthogonal():
    """the off-rank residual UUY lives outside the kinship span, U_effᵀ UUY = 0"""
    G, Z, X, Y = _toy()
    basis = lowrank_basis(G, X)
    _, UUY = rotate_phenos(basis, Y)
    proj = basis.U_eff.T @ UUY
    assert proj.abs().max().item() < 1e-10


def test_fit_delta_grid_lowrank_shape_finite():
    """delta fit returns one finite log-delta per pheno"""
    G, Z, X, Y = _toy()
    spec = lowrank_rotate(G, X, Y)
    ld = fit_delta_grid_lowrank(spec)
    assert ld.shape == (Y.shape[1],)
    assert torch.isfinite(ld).all()
    # the search is clamped to its grid bounds
    assert (ld >= -10.0 - 1e-9).all() and (ld <= 10.0 + 1e-9).all()


def test_snp_wald_scan_lowrank_finite_nonneg():
    """per-variant Wald F is finite and non-negative across the whole toy genome"""
    G, Z, X, Y = _toy()
    spec = lowrank_rotate(G, X, Y)
    ld = fit_delta_grid_lowrank(spec)
    res = snp_wald_scan_lowrank(spec, ld, Z)
    assert res.f.shape == (Z.shape[1], Y.shape[1])
    assert torch.isfinite(res.f).all()
    assert (res.f >= 0).all()
    assert torch.isfinite(res.se).all()
    assert (res.se > 0).all()  # SE comes from a clamped positive variance
    assert torch.isfinite(res.beta).all()
    assert torch.isfinite(res.max_F).all()


def test_single_k_scan_lowrank_matches_fullrank_compat():
    """THE cross-check: low-rank single-K agrees with the full-rank compat path on the SAME kinship K = grm(G)

    the full-rank compat path eighs K and keeps the near-zero null-space eigenvalues explicitly, the low-rank path folds those same null directions into its analytic off-rank term -- so F / beta / se / sfve land in the same place.  the two paths refit log-delta independently though, so the residual gap is the float noise of two separate golden-section searches, a touch under 1e-5 relative
    """
    G, Z, X, Y = _toy()
    res_lr = single_k_scan_lowrank(G, Z, X, Y)

    K = grm(G)
    spec = fastlmm_compat_rotate(K, X, Y)
    ld = fit_delta_grid_compat(spec)
    res_fr = snp_wald_scan_compat(spec, ld, Z)

    def max_rel(a, b):
        a, b = a.flatten(), b.flatten()
        return ((a - b).abs() / b.abs().clamp(min=1e-8)).max().item()

    # gaps observed at float64: F maxrel ~1.6e-6, beta ~8e-7, se ~1e-8, sfve ~8e-7, nullh2 ~9e-8
    assert max_rel(res_lr.f, res_fr.f) < 1e-5
    assert max_rel(res_lr.beta, res_fr.beta) < 1e-5
    assert max_rel(res_lr.se, res_fr.se) < 1e-6
    assert max_rel(res_lr.sfve, res_fr.sfve) < 1e-5
    assert max_rel(res_lr.nullh2, res_fr.nullh2) < 1e-6
    assert max_rel(res_lr.max_F, res_fr.max_F) < 1e-6
    # the F grids should be essentially the same field
    fl = res_lr.f.flatten().numpy()
    fr = res_fr.f.flatten().numpy()
    assert np.corrcoef(fl, fr)[0, 1] > 1.0 - 1e-9


def test_loco_scan_lowrank_matches_hand_rolled_per_chrom():
    """LOCO equals a hand-rolled per-chromosome reference: for each test chrom, kinship from the OTHER chrom's markers, scan that chrom's variants only"""
    G, Z, X, Y = _toy()
    Mkin, Mtest, P = G.shape[1], Z.shape[1], Y.shape[1]
    # two fake chromosomes, split both the kinship factor and the test SNPs down the middle
    g_chrom = np.array([1] * (Mkin // 2) + [2] * (Mkin - Mkin // 2))
    z_chrom = np.array([1] * (Mtest // 2) + [2] * (Mtest - Mtest // 2))

    res = loco_scan_lowrank(G, g_chrom, Z, z_chrom, X, Y)

    f_ref = torch.zeros(Mtest, P, dtype=DTYPE)
    beta_ref = torch.zeros(Mtest, P, dtype=DTYPE)
    se_ref = torch.zeros(Mtest, P, dtype=DTYPE)
    sfve_ref = torch.zeros(Mtest, P, dtype=DTYPE)
    nullh2_ref = torch.zeros(Mtest, P, dtype=DTYPE)
    maxF_ref = torch.full((P,), float("-inf"), dtype=DTYPE)
    for c in sorted(np.unique(z_chrom).tolist()):
        kin_mask = g_chrom != c  # kinship drops this chromosome's pruned markers
        test_mask = z_chrom == c
        rr = single_k_scan_lowrank(G[:, kin_mask], Z[:, test_mask], X, Y)
        f_ref[test_mask] = rr.f
        beta_ref[test_mask] = rr.beta
        se_ref[test_mask] = rr.se
        sfve_ref[test_mask] = rr.sfve
        nullh2_ref[test_mask] = rr.nullh2
        maxF_ref = torch.maximum(maxF_ref, rr.max_F)

    # same code path on the same masks, so this should be bit-for-bit
    assert torch.equal(res.f, f_ref)
    assert torch.equal(res.beta, beta_ref)
    assert torch.equal(res.se, se_ref)
    assert torch.equal(res.sfve, sfve_ref)
    assert torch.equal(res.nullh2, nullh2_ref)
    assert torch.equal(res.max_F, maxF_ref)


def test_loco_scan_lowrank_nullh2_per_chrom():
    """each chromosome's variants carry that chromosome's own null h2, so the two halves differ"""
    G, Z, X, Y = _toy()
    Mkin, Mtest = G.shape[1], Z.shape[1]
    g_chrom = np.array([1] * (Mkin // 2) + [2] * (Mkin - Mkin // 2))
    z_chrom = np.array([1] * (Mtest // 2) + [2] * (Mtest - Mtest // 2))
    res = loco_scan_lowrank(G, g_chrom, Z, z_chrom, X, Y)
    # null h2 is a single per-chrom value broadcast over that chrom's rows, so it's constant within a chrom block
    c1_rows = res.nullh2[z_chrom == 1]
    assert (c1_rows - c1_rows[0:1]).abs().max().item() < 1e-12
    assert torch.isfinite(res.nullh2).all()
    assert ((res.nullh2 >= 0) & (res.nullh2 <= 1)).all()


def test_single_k_scan_lowrank_n_real_split():
    """n_real trims the detailed columns: f / beta / se / sfve are (M, n_real) while max_F stays (P,)"""
    G, Z, X, Y = _toy()
    Mtest, P = Z.shape[1], Y.shape[1]
    res = single_k_scan_lowrank(G, Z, X, Y, n_real=2)
    assert res.f.shape == (Mtest, 2)
    assert res.beta.shape == (Mtest, 2)
    assert res.se.shape == (Mtest, 2)
    assert res.sfve.shape == (Mtest, 2)
    assert res.nullh2.shape == (Mtest, 2)
    assert res.max_F.shape == (P,)  # the running max covers every pheno, real and perm
    # the leading detail columns match the full scan exactly
    res_full = single_k_scan_lowrank(G, Z, X, Y)
    assert torch.equal(res.f, res_full.f[:, :2])
    assert torch.equal(res.max_F, res_full.max_F)


def test_lowrank_rotate_composes_basis_and_phenos():
    """lowrank_rotate is just lowrank_basis + rotate_phenos glued together, the spectrum carries the same pieces"""
    G, Z, X, Y = _toy()
    basis = lowrank_basis(G, X)
    UY, UUY = rotate_phenos(basis, Y)
    spec = lowrank_rotate(G, X, Y)
    assert torch.equal(spec.s, basis.s)
    assert torch.equal(spec.U_eff, basis.U_eff)
    assert spec.Neff == basis.Neff
    assert torch.equal(spec.UY, UY)
    assert torch.equal(spec.UUY, UUY)
