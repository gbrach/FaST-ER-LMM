"""
unit tests for the fastlmm-compat scan path in fasterlmm.core
covers eigendecompose / rotate / _pos_floor plus the projected compat chain:
fastlmm_compat_rotate (drop-D, OLS regress-out), fit_delta_grid_compat,
snp_wald_scan_compat and the loco / single-K wrappers around them
everything here is cpu-only and portable -- no GPU, no fastlmm, small synthetic
panels for the math, the shipped example bed only for the loco structure check
"""
from __future__ import annotations

import numpy as np
import torch

from fasterlmm.core import (
    _pos_floor,
    eigendecompose,
    fastlmm_compat_rotate,
    fit_delta_grid_compat,
    loco_scan_compat,
    rotate,
    single_k_scan_compat,
    snp_wald_scan_compat,
)
from fasterlmm.io import align_inputs, read_phen, read_plink, standardise_columns

DTYPE = torch.float64


def _toy_kinship(N, M, *, seed):
    """small standardised genotype + its grm, all float64 for exactness"""
    rng = np.random.default_rng(seed)
    Z = torch.from_numpy(standardise_columns(rng.normal(size=(N, M)))).to(DTYPE)
    K = (Z @ Z.T) / M
    return Z, K


# ---------------------------------------------------------------------------
# core._pos_floor
# ---------------------------------------------------------------------------


def test_pos_floor_dtype_specific():
    """float64 gets the tiny 1e-300 floor, float32 the representable 1e-30 one"""
    assert _pos_floor(torch.float64) == 1e-300
    assert _pos_floor(torch.float32) == 1e-30


# ---------------------------------------------------------------------------
# core.eigendecompose
# ---------------------------------------------------------------------------


def test_eigendecompose_ascending_and_orthonormal():
    """eigh hands back ascending eigenvalues and an orthonormal U"""
    _, K = _toy_kinship(30, 90, seed=1)
    s, U = eigendecompose(K, jitter=1e-8)
    assert s.shape == (30,)
    assert U.shape == (30, 30)
    # ascending order, no clamped value below zero
    assert torch.all(s[:-1] <= s[1:])
    assert (s >= 0.0).all()
    eye = torch.eye(30, dtype=DTYPE)
    assert torch.allclose(U.T @ U, eye, atol=1e-9)


def test_eigendecompose_reconstructs_jittered_K():
    """U diag(s) Uᵀ rebuilds K plus the jitter that got added on the diagonal"""
    _, K = _toy_kinship(25, 70, seed=2)
    jitter = 1e-7
    s, U = eigendecompose(K, jitter=jitter)
    K_recon = U @ torch.diag(s) @ U.T
    eye = torch.eye(25, dtype=DTYPE)
    # eigendecompose works on K + jitter*I, so the recon matches the jittered K
    assert torch.allclose(K_recon, K + jitter * eye, atol=1e-7)


# ---------------------------------------------------------------------------
# core.rotate
# ---------------------------------------------------------------------------


def test_rotate_X_rot_Y_rot_consistent_with_its_U():
    """rotate returns UᵀX and UᵀY against the same U it eigendecomposed"""
    _, K = _toy_kinship(24, 60, seed=3)
    rng = np.random.default_rng(33)
    X = torch.ones(24, 1, dtype=DTYPE)
    Y = torch.from_numpy(rng.normal(size=(24, 4))).to(DTYPE)
    spec = rotate(K, X, Y)
    assert spec.X_rot.shape == (24, 1)
    assert spec.Y_rot.shape == (24, 4)
    assert torch.allclose(spec.X_rot, spec.U.T @ X, atol=1e-10)
    assert torch.allclose(spec.Y_rot, spec.U.T @ Y, atol=1e-10)
    # the s it carries matches a fresh eigendecompose on the same K
    s2, _ = eigendecompose(K)
    assert torch.allclose(spec.s, s2, atol=1e-10)


# ---------------------------------------------------------------------------
# core.fastlmm_compat_rotate
# ---------------------------------------------------------------------------


def test_compat_rotate_drops_one_for_intercept_only():
    """D = 1 (intercept only) drops exactly one eigenvector"""
    N, M, P = 32, 90, 4
    _, K = _toy_kinship(N, M, seed=4)
    rng = np.random.default_rng(44)
    X = torch.ones(N, 1, dtype=DTYPE)
    Y = torch.from_numpy(rng.normal(size=(N, P))).to(DTYPE)
    spec = fastlmm_compat_rotate(K, X, Y)
    assert spec.s.shape == (N - 1,)
    assert spec.U_eff.shape == (N, N - 1)
    assert spec.UY.shape == (N - 1, P)
    # ascending and floored at zero
    assert torch.all(spec.s[:-1] <= spec.s[1:])
    assert (spec.s >= 0.0).all()


def test_compat_rotate_drops_D_for_intercept_plus_covars():
    """D = 6 ([intercept | 5 covars]) drops exactly six eigenvectors"""
    N, M, P = 32, 90, 3
    _, K = _toy_kinship(N, M, seed=5)
    rng = np.random.default_rng(55)
    cov = torch.from_numpy(rng.normal(size=(N, 5))).to(DTYPE)
    X = torch.cat([torch.ones(N, 1, dtype=DTYPE), cov], dim=1)
    Y = torch.from_numpy(rng.normal(size=(N, P))).to(DTYPE)
    spec = fastlmm_compat_rotate(K, X, Y)
    assert spec.s.shape == (N - 6,)
    assert spec.U_eff.shape == (N, N - 6)
    assert spec.UY.shape == (N - 6, P)


def test_compat_rotate_U_eff_columns_orthonormal():
    """the kept eigenvectors stay orthonormal: U_effᵀ U_eff == I"""
    N, M = 28, 70
    _, K = _toy_kinship(N, M, seed=6)
    rng = np.random.default_rng(66)
    X = torch.ones(N, 1, dtype=DTYPE)
    Y = torch.from_numpy(rng.normal(size=(N, 2))).to(DTYPE)
    spec = fastlmm_compat_rotate(K, X, Y)
    gram = spec.U_eff.T @ spec.U_eff
    assert torch.allclose(gram, torch.eye(N - 1, dtype=DTYPE), atol=1e-9)


def test_compat_rotate_UY_equals_regress_then_project():
    """UY == U_effᵀ (Y - X pinv(X) Y), the OLS regress-out then projection onto the kept basis"""
    N, M, P = 26, 60, 3
    _, K = _toy_kinship(N, M, seed=7)
    rng = np.random.default_rng(77)
    X = torch.from_numpy(rng.normal(size=(N, 4))).to(DTYPE)  # arbitrary 4-col design
    Y = torch.from_numpy(rng.normal(size=(N, P))).to(DTYPE)
    spec = fastlmm_compat_rotate(K, X, Y)
    Xpinv = torch.linalg.pinv(X)
    Y_r = Y - X @ (Xpinv @ Y)
    expected = spec.U_eff.T @ Y_r
    assert torch.allclose(spec.UY, expected, atol=1e-9)
    # the stored Xpinv is the same OLS hat
    assert torch.allclose(spec.Xpinv, Xpinv, atol=1e-9)


# ---------------------------------------------------------------------------
# core.fit_delta_grid_compat
# ---------------------------------------------------------------------------


def test_fit_delta_grid_compat_shape_finite_and_in_bounds():
    """returns (P,) finite log-deltas inside the [-10, 10] grid window"""
    N, M, P = 30, 100, 5
    _, K = _toy_kinship(N, M, seed=8)
    rng = np.random.default_rng(88)
    X = torch.ones(N, 1, dtype=DTYPE)
    Y = torch.from_numpy(rng.normal(size=(N, P))).to(DTYPE)
    spec = fastlmm_compat_rotate(K, X, Y)
    ld = fit_delta_grid_compat(spec)
    assert ld.shape == (P,)
    assert torch.isfinite(ld).all()
    assert (ld >= -10.0).all() and (ld <= 10.0).all()


def test_fit_delta_grid_compat_identical_phenos_identical_fit():
    """duplicate phenos must land the exact same fitted log-delta per column"""
    N, M, P = 28, 90, 4
    _, K = _toy_kinship(N, M, seed=9)
    rng = np.random.default_rng(99)
    y = rng.normal(size=(N,))
    Y = torch.from_numpy(np.tile(y[:, None], (1, P))).to(DTYPE)
    spec = fastlmm_compat_rotate(K, X=torch.ones(N, 1, dtype=DTYPE), Y=Y)
    ld = fit_delta_grid_compat(spec)
    assert torch.allclose(ld, ld[0].expand(P), atol=1e-10)


def test_fit_delta_grid_compat_refine_off_lands_on_grid():
    """with refine off the fit is exactly one of the 64 grid points"""
    N, M, P = 26, 80, 3
    _, K = _toy_kinship(N, M, seed=10)
    rng = np.random.default_rng(101)
    X = torch.ones(N, 1, dtype=DTYPE)
    Y = torch.from_numpy(rng.normal(size=(N, P))).to(DTYPE)
    spec = fastlmm_compat_rotate(K, X, Y)
    ld = fit_delta_grid_compat(spec, n_grid=64, refine=False)
    grid = torch.linspace(-10.0, 10.0, 64, dtype=DTYPE)
    # every fitted value coincides with a grid node
    for v in ld:
        assert torch.isclose(grid, v).any()


# ---------------------------------------------------------------------------
# core.snp_wald_scan_compat
# ---------------------------------------------------------------------------


def test_snp_wald_scan_compat_finite_nonneg_and_shapes():
    """F / beta / SE / sfve / nullh2 all (M, P), F finite and non-negative"""
    N, M, P = 40, 150, 3
    Z, K = _toy_kinship(N, M, seed=11)
    rng = np.random.default_rng(111)
    X = torch.ones(N, 1, dtype=DTYPE)
    Y = torch.from_numpy(rng.normal(size=(N, P))).to(DTYPE)
    spec = fastlmm_compat_rotate(K, X, Y)
    ld = fit_delta_grid_compat(spec)
    res = snp_wald_scan_compat(spec, ld, Z)
    for arr in (res.f, res.beta, res.se, res.sfve, res.nullh2):
        assert arr.shape == (M, P)
    assert torch.isfinite(res.f).all()
    assert (res.f >= 0.0).all()
    assert torch.isfinite(res.beta).all()
    assert torch.isfinite(res.se).all()
    # SE is the sqrt of a clamped-positive variance, never negative
    assert (res.se >= 0.0).all()
    # sfve is a clamped-then-sqrt fraction, lands in [0, 1]-ish but at least non-neg
    assert (res.sfve >= 0.0).all()


def test_snp_wald_scan_compat_max_F_is_per_column_max():
    """with n_real == P, max_F equals the per-column max of f over all variants"""
    N, M, P = 38, 140, 4
    Z, K = _toy_kinship(N, M, seed=12)
    rng = np.random.default_rng(121)
    X = torch.ones(N, 1, dtype=DTYPE)
    Y = torch.from_numpy(rng.normal(size=(N, P))).to(DTYPE)
    spec = fastlmm_compat_rotate(K, X, Y)
    ld = fit_delta_grid_compat(spec)
    res = snp_wald_scan_compat(spec, ld, Z)
    assert res.max_F.shape == (P,)
    assert torch.allclose(res.max_F, res.f.max(dim=0).values, atol=1e-10)


def test_snp_wald_scan_compat_nullh2_is_one_over_one_plus_delta():
    """nullh2 column == 1 / (1 + exp(log_delta)), constant across variants under single K"""
    N, M, P = 36, 120, 3
    Z, K = _toy_kinship(N, M, seed=13)
    rng = np.random.default_rng(131)
    X = torch.ones(N, 1, dtype=DTYPE)
    Y = torch.from_numpy(rng.normal(size=(N, P))).to(DTYPE)
    spec = fastlmm_compat_rotate(K, X, Y)
    ld = fit_delta_grid_compat(spec)
    res = snp_wald_scan_compat(spec, ld, Z)
    expected_h2 = 1.0 / (1.0 + ld.exp())
    # one h2 per pheno, broadcast down every variant row
    for p in range(P):
        assert torch.allclose(res.nullh2[:, p], expected_h2[p].expand(M), atol=1e-12)


def test_snp_wald_scan_compat_chunking_invariance():
    """tiling the snp / pheno axes can't move any output value"""
    N, M, P = 34, 130, 5
    Z, K = _toy_kinship(N, M, seed=14)
    rng = np.random.default_rng(141)
    X = torch.ones(N, 1, dtype=DTYPE)
    Y = torch.from_numpy(rng.normal(size=(N, P))).to(DTYPE)
    spec = fastlmm_compat_rotate(K, X, Y)
    ld = fit_delta_grid_compat(spec)
    big = snp_wald_scan_compat(spec, ld, Z, snp_chunk=4096, pheno_chunk=256)
    tiled = snp_wald_scan_compat(spec, ld, Z, snp_chunk=17, pheno_chunk=2)
    assert torch.allclose(big.f, tiled.f, atol=1e-12)
    assert torch.allclose(big.beta, tiled.beta, atol=1e-12)
    assert torch.allclose(big.se, tiled.se, atol=1e-12)
    assert torch.allclose(big.max_F, tiled.max_F, atol=1e-12)


def test_snp_wald_scan_compat_n_real_perm_split():
    """n_real = B < P keeps full detail for B real columns, max_F still spans all P"""
    N, M, P = 34, 120, 6
    B = 2
    Z, K = _toy_kinship(N, M, seed=15)
    rng = np.random.default_rng(151)
    X = torch.ones(N, 1, dtype=DTYPE)
    Y = torch.from_numpy(rng.normal(size=(N, P))).to(DTYPE)
    spec = fastlmm_compat_rotate(K, X, Y)
    ld = fit_delta_grid_compat(spec)
    res = snp_wald_scan_compat(spec, ld, Z, n_real=B)
    # detail tensors trimmed to the B leading real phenos
    assert res.f.shape == (M, B)
    assert res.beta.shape == (M, B)
    assert res.se.shape == (M, B)
    assert res.nullh2.shape == (M, B)
    # max_F still covers every column, real + perm
    assert res.max_F.shape == (P,)
    # the B real columns of max_F match the kept detail's per-column max
    assert torch.allclose(res.max_F[:B], res.f.max(dim=0).values, atol=1e-10)
    # a full scan should give the same first B columns of f and the same full max_F
    full = snp_wald_scan_compat(spec, ld, Z, n_real=P)
    assert torch.allclose(res.f, full.f[:, :B], atol=1e-12)
    assert torch.allclose(res.max_F, full.max_F, atol=1e-12)


# ---------------------------------------------------------------------------
# core.single_k_scan_compat / loco_scan_compat structure
# ---------------------------------------------------------------------------


def test_single_k_scan_compat_nullh2_constant_across_variants():
    """one K over the genome -> each pheno's Nullh2 column is one value repeated down every variant"""
    N, M, P = 40, 200, 3
    rng = np.random.default_rng(16)
    chrom = np.repeat(np.array([1, 2, 3, 4]), M // 4)
    Z = torch.from_numpy(standardise_columns(rng.normal(size=(N, M)))).to(DTYPE)
    X = torch.ones(N, 1, dtype=DTYPE)
    Y = torch.from_numpy(rng.normal(size=(N, P))).to(DTYPE)
    res = single_k_scan_compat(Z, X, Y)
    assert res.f.shape == (M, P)
    # single K -> nullh2 has a single distinct value per pheno column
    for p in range(P):
        col = res.nullh2[:, p]
        assert torch.allclose(col, col[0].expand(M), atol=1e-12)


def test_loco_scan_compat_nullh2_varies_by_chromosome():
    """LOCO refits delta per chromosome -> Nullh2 is blockwise constant per chrom, differs across chroms"""
    N, M, P = 40, 200, 2
    rng = np.random.default_rng(17)
    n_chr = 4
    chrom = np.repeat(np.array([1, 2, 3, 4]), M // n_chr)
    Z_np = standardise_columns(rng.normal(size=(N, M)))
    Z = torch.from_numpy(Z_np).to(DTYPE)
    X = torch.ones(N, 1, dtype=DTYPE)
    # give the phenos real kinship-shaped signal so each leave-one-chrom refit
    # of delta lands somewhere different -- pure noise just pins every chrom to
    # the same grid edge and the per-chrom h2 never moves
    beta = rng.normal(size=(M, P)) / np.sqrt(M)
    poly = Z_np @ beta
    Y_np = poly + 0.3 * rng.normal(size=(N, P))
    Y = torch.from_numpy(Y_np).to(DTYPE)
    res = loco_scan_compat(Z, X, Y, chrom)
    assert res.f.shape == (M, P)
    assert res.nullh2.shape == (M, P)
    # within each chromosome the null h2 is constant, but the per-chrom values
    # are not all identical -- dropping a different chrom each time refits delta
    per_chrom_vals = []
    for c in (1, 2, 3, 4):
        mask = torch.from_numpy(chrom == c)
        block = res.nullh2[mask, 0]
        assert torch.allclose(block, block[0].expand(block.shape[0]), atol=1e-12)
        per_chrom_vals.append(block[0].item())
    distinct = {round(v, 9) for v in per_chrom_vals}
    assert len(distinct) > 1  # at least two chromosomes give different null h2


def test_loco_scan_compat_max_F_is_genome_max(example_geno, example_pheno):
    """on the shipped example bed, loco max_F is the per-pheno max over the full f grid"""
    g = read_plink(example_geno)
    p = read_phen(example_pheno)
    a = align_inputs(g, p)
    # keep it quick: just the first two phenos
    Z = standardise_columns(a.Z).to(DTYPE)
    X = a.X.to(DTYPE)
    Y = a.Y[:, :2].to(DTYPE)
    res = loco_scan_compat(Z, X, Y, g.chrom)
    assert res.f.shape == (1500, 2)
    assert res.max_F.shape == (2,)
    assert torch.allclose(res.max_F, res.f.max(dim=0).values, atol=1e-9)


def test_loco_scan_compat_n_real_perm_split():
    """n_real = B < P on the loco wrapper: f shape (M, B), max_F shape (P,)"""
    N, M, P = 40, 160, 5
    B = 2
    rng = np.random.default_rng(18)
    chrom = np.repeat(np.array([1, 2, 3, 4]), M // 4)
    Z = torch.from_numpy(standardise_columns(rng.normal(size=(N, M)))).to(DTYPE)
    X = torch.ones(N, 1, dtype=DTYPE)
    Y = torch.from_numpy(rng.normal(size=(N, P))).to(DTYPE)
    res = loco_scan_compat(Z, X, Y, chrom, n_real=B)
    assert res.f.shape == (M, B)
    assert res.beta.shape == (M, B)
    assert res.nullh2.shape == (M, B)
    assert res.max_F.shape == (P,)
    # the leading B columns of the full-detail scan agree with this trimmed one
    full = loco_scan_compat(Z, X, Y, chrom, n_real=P)
    assert torch.allclose(res.f, full.f[:, :B], atol=1e-12)
    assert torch.allclose(res.max_F, full.max_F, atol=1e-12)
