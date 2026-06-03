"""
external fastlmm parity for the low-rank scan path, the k < N branch of lmm_cov

this whole file is skip-gated on fastlmm being importable, so a fresh clone with
no fastlmm and no gpu skips it cleanly.  cpu-only, portable, small toy panel

two things get checked against fastlmm's own forcefullrank=False LMM, per pheno:
  the fitted log-delta from findH2 (a loose-tol fit comparison, golden-section
  vs fastlmm's own optimiser will drift in the last few digits), and then the
  per-variant F / beta / se at the COMMON fastlmm delta, which isolates the scan
  math from the fit and matches to near machine precision
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fasterlmm.io import standardise_columns
from fasterlmm.lowrank import (
    fit_delta_grid_lowrank,
    lowrank_rotate,
    snp_wald_scan_lowrank,
)

DTYPE = torch.float64


def _toy_panel(N=300, M_kin=80, M_test=60, P=4, D_cov=2, seed=19930909):
    """same toy panel test_lowrank_parity.py builds: G pruned kinship, Z test variants, X = intercept + covars, Y with a planted effect"""
    rng = np.random.default_rng(seed)
    G = rng.binomial(2, 0.3, size=(N, M_kin)).astype(np.float64)
    Z = rng.binomial(2, 0.25, size=(N, M_test)).astype(np.float64)
    cov = rng.standard_normal((N, D_cov))
    X = np.concatenate([np.ones((N, 1)), cov], axis=1)
    # plant signal: a couple test variants nudge the phenos, plus a kinship-shaped polygenic part
    g_std = standardise_columns(G)
    poly = g_std @ rng.standard_normal((M_kin, P)) / np.sqrt(M_kin)
    Y = poly + 0.4 * Z[:, [0]] * rng.standard_normal((1, P)) + rng.standard_normal((N, P))
    Y += X @ rng.standard_normal((X.shape[1], P))  # covariate effect, both paths regress it out
    return (
        torch.tensor(standardise_columns(G), dtype=DTYPE),
        torch.tensor(standardise_columns(Z), dtype=DTYPE),
        torch.tensor(X, dtype=DTYPE),
        torch.tensor(Y, dtype=DTYPE),
    )


def test_lowrank_external_fastlmm_parity():
    """low-rank fit + scan vs fastlmm lmm_cov forcefullrank=False on the same kinship factor G"""
    fastLMM = pytest.importorskip("fastlmm.inference.lmm_cov").LMM

    G, Z, X, Y = _toy_panel()
    # match the 1/M_kin eigenvalue scaling: fastlmm's G GT is the kinship, we divide G by sqrt(k)
    Gn = (G / np.sqrt(G.shape[1])).numpy()
    Xn, Yn, Zn = X.numpy(), Y.numpy(), Z.numpy()
    P = Yn.shape[1]

    fit_gap = []
    f_gap, beta_gap, se_gap = [], [], []
    for p in range(P):
        lmm = fastLMM(forcefullrank=False, X=Xn, Y=Yn[:, [p]], G=Gn)
        h2 = lmm.findH2()  # ML, this nLLeval path has no REML branch so it matches our loss
        h2val = float(np.ravel(h2["h2"])[0])
        delta = (1.0 - h2val) / h2val
        log_delta_fl = float(np.log(delta))

        # our spectral fit on the same pheno
        spec = lowrank_rotate(G, X, Y[:, [p]])
        ld_ours = fit_delta_grid_lowrank(spec).item()
        fit_gap.append(abs(ld_ours - log_delta_fl))

        # per-variant F / beta / se at the COMMON fastlmm delta, so any gap is scan math not fit
        out = lmm.nLLeval(delta=delta, snps=Zn)
        beta_fl = np.asarray(out["beta"]).flatten()
        var_fl = np.asarray(out["variance_beta"]).flatten()
        f_fl = beta_fl * beta_fl / var_fl
        se_fl = np.sqrt(var_fl)

        ld_common = torch.full((1,), log_delta_fl, dtype=DTYPE)
        res = snp_wald_scan_lowrank(spec, ld_common, Z)
        f_me = res.f.flatten().numpy()
        beta_me = res.beta.flatten().numpy()
        se_me = res.se.flatten().numpy()

        f_gap.append(np.nanmax(np.abs(f_me - f_fl) / (np.abs(f_fl) + 1e-8)))
        beta_gap.append(np.nanmax(np.abs(beta_me - beta_fl)))
        se_gap.append(np.nanmax(np.abs(se_me - se_fl)))

    # fitted log-delta: a fit-vs-fit comparison, golden-section drifts in the tail digits
    # observed max gap ~8e-08 on this panel, keep a comfortable loose ceiling
    assert max(fit_gap) < 1e-4, f"fitted log-delta gap {max(fit_gap):.3e}"

    # at the common delta the scan math is the same algebra, near machine precision
    # observed: F rel ~5e-13, beta abs ~3e-16, se abs ~6e-17
    assert max(f_gap) < 1e-9, f"per-variant F rel gap {max(f_gap):.3e}"
    assert max(beta_gap) < 1e-11, f"per-variant beta abs gap {max(beta_gap):.3e}"
    assert max(se_gap) < 1e-11, f"per-variant se abs gap {max(se_gap):.3e}"
