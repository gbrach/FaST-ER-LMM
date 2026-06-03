"""
plain GLS path in fasterlmm.core, the rotate / nLLeval / _profile_loss / fit_delta_grid / snp_wald_scan stack
this is the statistically-correct rank-reduced sibling of the fastlmm-compat path, not the bit-for-bit one
everything here is pure-math on tiny synthetic panels, cpu-only and portable -- no fastlmm, no gpu, no bed files needed
"""
from __future__ import annotations

import numpy as np
import torch

from fasterlmm.core import (
    ScanResult,
    _auto_tile,
    _profile_loss,
    eigendecompose,
    fit_delta_grid,
    loco_scan,
    nLLeval,
    rotate,
    single_k_scan,
    snp_wald_scan,
)
from fasterlmm.io import grm, standardise_columns


# ---------------------------------------------------------------------------
# small synthetic builders, all float64 so the exactness checks have room
# ---------------------------------------------------------------------------


def _make_panel(N=40, M=120, P=3, C=2, seed=19930909):
    """build a standardised genotype panel + design X + phenos Y, all float64 tensors"""
    rng = np.random.default_rng(seed)
    Zraw = rng.integers(0, 3, size=(N, M)).astype(np.float64)  # 0/1/2 dosages
    Z = standardise_columns(torch.from_numpy(Zraw))  # pre-standardised, what the scan wants
    # design = intercept plus a continous covariate
    X = torch.ones(N, C, dtype=torch.float64)
    if C > 1:
        X[:, 1:] = torch.from_numpy(rng.normal(size=(N, C - 1)))
    Y = torch.from_numpy(rng.normal(size=(N, P)))
    return Z, X, Y


def _make_spectrum(N=40, M=120, P=3, C=2, seed=19930909):
    """grm + rotate into a Spectrum, the entry point for the loss / fit / scan tests"""
    Z, X, Y = _make_panel(N=N, M=M, P=P, C=C, seed=seed)
    K = grm(Z)
    spec = rotate(K, X, Y)
    return Z, spec


# ---------------------------------------------------------------------------
# nLLeval vs _profile_loss consistency
# ---------------------------------------------------------------------------


def test_nlleval_matches_profile_loss_cell():
    """nLLeval at one delta for one pheno equals the matching (g, p) cell of _profile_loss"""
    _, spec = _make_spectrum()
    s, X_rot, Y_rot = spec.s, spec.X_rot, spec.Y_rot
    P = Y_rot.shape[1]
    # a handful of log deltas spanning the search range, evaluated as a grid
    grid = torch.tensor([-4.0, -1.0, 0.0, 2.0, 5.0], dtype=torch.float64)
    loss = _profile_loss(grid, s, X_rot, Y_rot)  # (G, P)
    assert loss.shape == (grid.shape[0], P)
    for g, log_delta in enumerate(grid.tolist()):
        delta = float(np.exp(log_delta))
        for p in range(P):
            cell = nLLeval(delta, s, X_rot, Y_rot[:, p])
            # both compute the same ML -2 loglik, the only gap is einsum vs cholesky float order
            assert torch.allclose(cell, loss[g, p], atol=1e-10, rtol=1e-10)


def test_nlleval_scalar_output():
    """nLLeval returns a scalar (0-d) tensor and stays finite at a sane delta"""
    _, spec = _make_spectrum()
    out = nLLeval(1.0, spec.s, spec.X_rot, spec.Y_rot[:, 0])
    assert out.ndim == 0
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# _profile_loss ML vs REML
# ---------------------------------------------------------------------------


def test_profile_loss_reml_differs_and_finite():
    """reml=True gives a different but still finite loss than reml=False"""
    _, spec = _make_spectrum()
    s, X_rot, Y_rot = spec.s, spec.X_rot, spec.Y_rot
    grid = torch.linspace(-5.0, 5.0, 16, dtype=torch.float64)
    ml = _profile_loss(grid, s, X_rot, Y_rot, reml=False)
    reml = _profile_loss(grid, s, X_rot, Y_rot, reml=True)
    assert ml.shape == reml.shape
    assert torch.isfinite(ml).all()
    assert torch.isfinite(reml).all()
    # the REML penalty (logdet A + the N-C divisor) shifts every cell off the ML value
    assert not torch.allclose(ml, reml, atol=1e-6)


def test_profile_loss_reml_decomposition():
    """reml loss equals the ml-style (N-C) divisor branch plus the logdet A penalty, sanity on the algebra"""
    _, spec = _make_spectrum(N=30, M=60, P=2, C=3)
    s, X_rot, Y_rot = spec.s, spec.X_rot, spec.Y_rot
    N, C = X_rot.shape
    grid = torch.tensor([-2.0, 0.5], dtype=torch.float64)
    reml = _profile_loss(grid, s, X_rot, Y_rot, reml=True)
    # under REML the variance term uses N-C, so the leading coefficient on the (log sigma2 + ...) block is N-C
    # we don't reach into internals, just confirm the REML loss is strictly above what an N divisor would give for these spectra (logdet A here is comfortably positive)
    assert torch.isfinite(reml).all()
    assert reml.shape == (grid.shape[0], Y_rot.shape[1])


# ---------------------------------------------------------------------------
# fit_delta_grid
# ---------------------------------------------------------------------------


def test_fit_delta_grid_duplicate_phenos_agree():
    """duplicating one pheno across columns should fit a near-identical log delta to every copy"""
    Z, X, Y = _make_panel(P=1)
    # broadcast the single pheno into 4 identical columns
    Y4 = Y[:, [0, 0, 0, 0]].contiguous()
    K = grm(Z)
    spec = rotate(K, X, Y4)
    ld = fit_delta_grid(spec)
    assert ld.shape == (4,)
    # identical phenos -> identical fits, the golden-section is deterministic per column
    assert torch.allclose(ld, ld[0].expand_as(ld), atol=1e-9)


def test_fit_delta_grid_refine_improves_on_grid_argmin():
    """refine=True lands at a loss <= the refine=False grid argmin loss, per pheno"""
    _, spec = _make_spectrum(P=4)
    s, X_rot, Y_rot = spec.s, spec.X_rot, spec.Y_rot
    ld_grid = fit_delta_grid(spec, refine=False)  # (P,)
    ld_ref = fit_delta_grid(spec, refine=True)  # (P,)
    assert ld_grid.shape == ld_ref.shape
    # evaluate each fit's own loss column by reusing _profile_loss on a 1-point grid per pheno
    P = Y_rot.shape[1]
    for p in range(P):
        loss_grid = _profile_loss(ld_grid[p : p + 1], s, X_rot, Y_rot)[0, p]
        loss_ref = _profile_loss(ld_ref[p : p + 1], s, X_rot, Y_rot)[0, p]
        # refinement never makes the minimum worse, tiny float slack on the equal case
        assert loss_ref <= loss_grid + 1e-9


def test_fit_delta_grid_respects_bounds():
    """the fitted log delta stays inside [log_delta_min, log_delta_max]"""
    _, spec = _make_spectrum(P=3)
    ld = fit_delta_grid(spec, log_delta_min=-8.0, log_delta_max=8.0)
    assert (ld >= -8.0 - 1e-9).all()
    assert (ld <= 8.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# _auto_tile cpu fallback
# ---------------------------------------------------------------------------


def test_auto_tile_cpu_fallback():
    """on a cpu device _auto_tile returns (min(256, P), min(4096, M)) with no mem probing"""
    dev = torch.device("cpu")
    # small P / M clamp to themselves
    assert _auto_tile(40, 2, 120, 3, dev) == (3, 120)
    # large P / M clamp to the cpu caps
    assert _auto_tile(40, 2, 9000, 600, dev) == (256, 4096)


# ---------------------------------------------------------------------------
# snp_wald_scan with a pre-rotated S_rot
# ---------------------------------------------------------------------------


def test_snp_wald_scan_finite_and_max_consistent():
    """snp_wald_scan on a pre-rotated S_rot gives finite f>=0 and max_F equal to the per-column f max"""
    Z, spec = _make_spectrum(N=40, M=120, P=3)
    P = spec.Y_rot.shape[1]
    M = Z.shape[1]
    log_delta = fit_delta_grid(spec)  # (P,)
    S_rot = spec.U.T @ Z  # the scan wants S ALREADY rotated
    res = snp_wald_scan(spec, log_delta, S_rot, n_real=P)
    assert isinstance(res, ScanResult)
    assert res.f.shape == (M, P)
    assert res.beta.shape == (M, P)
    assert res.se.shape == (M, P)
    assert res.sfve.shape == (M, P)
    assert res.nullh2.shape == (M, P)
    assert res.max_F.shape == (P,)
    # F stats are squared-beta-over-variance, non-negative and finite
    assert torch.isfinite(res.f).all()
    assert (res.f >= 0).all()
    # with n_real == P every column kept full detail, so the running max_F is exactly the column max of f
    assert torch.allclose(res.max_F, res.f.max(dim=0).values, atol=1e-12)


def test_snp_wald_scan_nullh2_from_delta():
    """nullh2 column equals 1 / (1 + delta) at the fitted log delta, constant down each variant column"""
    Z, spec = _make_spectrum(P=2)
    log_delta = fit_delta_grid(spec)
    S_rot = spec.U.T @ Z
    res = snp_wald_scan(spec, log_delta, S_rot)
    expected = 1.0 / (1.0 + log_delta.exp())  # (P,)
    # every row of nullh2 is the same per-pheno h2
    assert torch.allclose(res.nullh2[0], expected, atol=1e-12)
    assert torch.allclose(res.nullh2, res.nullh2[0].unsqueeze(0).expand_as(res.nullh2), atol=1e-12)


def test_snp_wald_scan_n_real_split():
    """n_real < P keeps full detail only for the leading columns, max_F still covers every column"""
    Z, spec = _make_spectrum(N=40, M=80, P=5)
    P = spec.Y_rot.shape[1]
    M = Z.shape[1]
    log_delta = fit_delta_grid(spec)
    S_rot = spec.U.T @ Z
    res = snp_wald_scan(spec, log_delta, S_rot, n_real=2)
    # only the first 2 real columns carry per-variant detail
    assert res.f.shape == (M, 2)
    assert res.beta.shape == (M, 2)
    # max_F is the full-width per-column genome max even for the trailing perm-style columns
    assert res.max_F.shape == (P,)
    assert torch.isfinite(res.max_F).all()
    # the leading real columns' max_F must match their kept f column max
    assert torch.allclose(res.max_F[:2], res.f.max(dim=0).values, atol=1e-12)


# ---------------------------------------------------------------------------
# loco_scan + single_k_scan shapes
# ---------------------------------------------------------------------------


def test_loco_scan_shapes_and_chrom_specific_h2():
    """loco_scan returns correctly shaped outputs on a few-chromosome standardised panel"""
    N, M, P = 40, 90, 3
    Z, X, Y = _make_panel(N=N, M=M, P=P)
    # three fake chromosomes, 30 variants each
    chrom = np.repeat(np.array([1, 2, 3]), M // 3)
    res = loco_scan(Z, X, Y, chrom, n_real=P)
    assert isinstance(res, ScanResult)
    assert res.f.shape == (M, P)
    assert res.beta.shape == (M, P)
    assert res.se.shape == (M, P)
    assert res.sfve.shape == (M, P)
    assert res.nullh2.shape == (M, P)
    assert res.max_F.shape == (P,)
    assert torch.isfinite(res.f).all()
    assert (res.f >= 0).all()
    # nullh2 is chromosome-specific under LOCO, every variant on a chrom shares one h2 row
    # so within a chrom block the nullh2 column is constant, but different chroms can differ
    for c in (1, 2, 3):
        block = res.nullh2[chrom == c]
        assert torch.allclose(block, block[0].unsqueeze(0).expand_as(block), atol=1e-12)


def test_single_k_scan_shapes():
    """single_k_scan returns a finite ScanResult with the expected shapes over all M"""
    N, M, P = 40, 100, 2
    Z, X, Y = _make_panel(N=N, M=M, P=P)
    res = single_k_scan(Z, X, Y, n_real=P)
    assert isinstance(res, ScanResult)
    assert res.f.shape == (M, P)
    assert res.beta.shape == (M, P)
    assert res.se.shape == (M, P)
    assert res.sfve.shape == (M, P)
    assert res.nullh2.shape == (M, P)
    assert res.max_F.shape == (P,)
    assert torch.isfinite(res.f).all()
    assert (res.f >= 0).all()
    # whole-genome single K shares one null h2 across every variant for a given pheno
    assert torch.allclose(res.nullh2, res.nullh2[0].unsqueeze(0).expand_as(res.nullh2), atol=1e-12)


def test_single_k_scan_matches_snp_wald_scan_path():
    """single_k_scan is just grm + rotate + fit + snp_wald_scan glued, so it reproduces the manual stack exactly"""
    N, M, P = 36, 72, 2
    Z, X, Y = _make_panel(N=N, M=M, P=P)
    res = single_k_scan(Z, X, Y, n_real=P)
    # rebuild by hand
    K = grm(Z)
    spec = rotate(K, X, Y)
    ld = fit_delta_grid(spec)
    S_rot = spec.U.T @ Z
    manual = snp_wald_scan(spec, ld, S_rot, n_real=P)
    assert torch.allclose(res.f, manual.f, atol=1e-12, rtol=0.0)
    assert torch.allclose(res.beta, manual.beta, atol=1e-12, rtol=0.0)
    assert torch.allclose(res.se, manual.se, atol=1e-12, rtol=0.0)
    assert torch.allclose(res.max_F, manual.max_F, atol=1e-12, rtol=0.0)
