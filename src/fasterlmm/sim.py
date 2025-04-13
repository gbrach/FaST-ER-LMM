"""
Toy GWAS synthesis for sanity checks
Generates a (N, M) genotype matrix + a phenotype with a known h2 from a small number of causal SNPs.  Mostly handy for trying out the scan + perm threshold on something where the truth is known
"""

from __future__ import annotations

import numpy as np


def sim_gwas(N: int = 100,
             M: int = 500,
             *,
             h2: float = 0.5,
             n_causal: int = 10,
             seed: int = 19930909) -> dict:
    """
    Quantitative GWAS sim
    Z is iid biallelic dosages (binomial(2, 0.5)).
    Pheno y has heritability h2 driven by n_causal randomly-picked SNPs with iid normal effects, gaussian residual makes up the rest
    Returns a dict with Z (N, M), y (N,), causal_idx (n_causal,), and true_h2
    """
    rng = np.random.default_rng(seed)
    Z = rng.binomial(2, 0.5, size=(N, M)).astype(np.float64)
    Zs = (Z - Z.mean(0)) / Z.std(0).clip(min=1e-12)
    causal_idx = rng.choice(M, size=n_causal, replace=False)
    beta = rng.standard_normal(n_causal)
    g = Zs[:, causal_idx] @ beta
    g = g / g.std() * np.sqrt(h2)  # rescaling so var(g) = h2 exactly
    e = rng.standard_normal(N) * np.sqrt(1 - h2)
    return {"Z": Z, "y": g + e, "causal_idx": causal_idx, "true_h2": h2}
