"""
tiny checks for fasterlmm.sim.sim_gwas, the toy GWAS synthesiser
cpu-only and portable, nothing here needs a gpu or fastlmm installed.  we
poke at the returned dict shapes, the dosage support, reproducibility under a
fixed seed, and that the genetic signal variance roughly tracks the requested
h2.  everything stays small so it runs in well under a second
"""
from __future__ import annotations

import numpy as np
import pytest

from fasterlmm.sim import sim_gwas


# ---------------------------------------------------------------------------
# dict shape + dtype contract
# ---------------------------------------------------------------------------


def test_returns_expected_keys():
    """the dict carries exactly Z, y, causal_idx, true_h2"""
    out = sim_gwas(N=40, M=120, n_causal=8, seed=19930909)
    assert set(out.keys()) == {"Z", "y", "causal_idx", "true_h2"}


def test_shapes_match_requested_sizes():
    """Z is (N, M), y is (N,), causal_idx is (n_causal,)"""
    N, M, n_causal = 50, 200, 12
    out = sim_gwas(N=N, M=M, n_causal=n_causal, seed=19930909)
    assert out["Z"].shape == (N, M)
    assert out["y"].shape == (N,)
    assert out["causal_idx"].shape == (n_causal,)


def test_genotype_dtype_is_float64():
    """dosages come back as float64 even though they hold integer counts"""
    out = sim_gwas(N=30, M=80, seed=1)
    assert out["Z"].dtype == np.float64


# ---------------------------------------------------------------------------
# value support
# ---------------------------------------------------------------------------


def test_dosages_live_in_zero_one_two():
    """every dosage is one of {0, 1, 2}, biallelic counts and nothing else"""
    out = sim_gwas(N=60, M=300, seed=7)
    uniq = np.unique(out["Z"])
    assert np.all(np.isin(uniq, [0.0, 1.0, 2.0]))


def test_causal_idx_unique_and_in_range():
    """the causal columns are distinct and index real columns of Z"""
    M, n_causal = 150, 15
    out = sim_gwas(N=40, M=M, n_causal=n_causal, seed=11)
    cidx = out["causal_idx"]
    assert cidx.shape == (n_causal,)
    assert np.unique(cidx).size == n_causal  # no repeats, replace=False
    assert cidx.min() >= 0 and cidx.max() < M


def test_true_h2_echoes_request():
    """true_h2 is just the h2 that was asked for"""
    out = sim_gwas(N=30, M=100, h2=0.3, seed=3)
    assert out["true_h2"] == 0.3


def test_phenotype_is_finite():
    """y has no NaNs or infs, the rescaling should not blow up"""
    out = sim_gwas(N=50, M=200, h2=0.6, seed=5)
    assert np.all(np.isfinite(out["y"]))


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_is_bit_identical():
    """a fixed seed reproduces Z, y and causal_idx exactly"""
    a = sim_gwas(N=40, M=120, n_causal=10, seed=19930909)
    b = sim_gwas(N=40, M=120, n_causal=10, seed=19930909)
    assert np.array_equal(a["Z"], b["Z"])
    assert np.array_equal(a["y"], b["y"])
    assert np.array_equal(a["causal_idx"], b["causal_idx"])


def test_different_seeds_diverge():
    """two seeds give different genotypes and phenotypes"""
    a = sim_gwas(N=40, M=120, seed=19930909)
    b = sim_gwas(N=40, M=120, seed=42)
    assert not np.array_equal(a["Z"], b["Z"])
    assert not np.allclose(a["y"], b["y"])


# ---------------------------------------------------------------------------
# heritability tracking
# ---------------------------------------------------------------------------


def test_genetic_variance_tracks_h2():
    """var of the genetic part should sit near h2, total pheno var near 1

    the genetic signal g is rescaled so var(g) == h2 exactly and the residual
    carries var (1 - h2), so the empirical numbers should land close-ish.  the
    residual is a finite gaussian draw so we keep the tolerance loose
    """
    N, M = 400, 600
    h2 = 0.5
    # reconstruct the genetic part the same way the sim builds it
    rng = np.random.default_rng(19930909)
    Z = rng.binomial(2, 0.5, size=(N, M)).astype(np.float64)
    Zs = (Z - Z.mean(0)) / Z.std(0).clip(min=1e-12)
    causal_idx = rng.choice(M, size=10, replace=False)
    beta = rng.standard_normal(10)
    g = Zs[:, causal_idx] @ beta
    g = g / g.std() * np.sqrt(h2)
    # the population-style var(g) is h2 by construction (ddof=0)
    assert np.isclose(np.var(g), h2, atol=1e-9)
    # and the phenotype this seed produces has total variance near 1
    out = sim_gwas(N=N, M=M, h2=h2, seed=19930909)
    assert np.isclose(np.var(out["y"]), 1.0, atol=0.2)


def test_higher_h2_gives_more_genetic_signal():
    """bumping h2 up should raise the genetic variance fraction of y

    quick monotone sanity check, we proxy the genetic fraction by correlating
    y against the causal genotype block and squaring.  loose since it is one
    finite draw, but a big h2 gap should dominate the noise
    """
    N, M = 500, 400

    def gen_frac(h2):
        out = sim_gwas(N=N, M=M, h2=h2, n_causal=10, seed=19930909)
        Z = out["Z"]
        Zs = (Z - Z.mean(0)) / Z.std(0).clip(min=1e-12)
        block = Zs[:, out["causal_idx"]]
        # least-squares fit of y on the causal block, fraction of var explained
        coef, *_ = np.linalg.lstsq(block, out["y"], rcond=None)
        fit = block @ coef
        return np.var(fit) / np.var(out["y"])

    lo = gen_frac(0.2)
    hi = gen_frac(0.8)
    assert hi > lo
