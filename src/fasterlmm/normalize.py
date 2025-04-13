"""
Phenotype normalisations.  Just RINT (rank-inverse-normal transform) for now -- the workhorse for non-gaussian phenos like raw transcript counts where outliers tank the perm threshold
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def rint(y: np.ndarray, *, ties_method: str = "average") -> np.ndarray:
    """
    Rank-inverse-normal transform.  Ranks per column, scales to (0, 1), pushes through the standard normal quantile
    Operates along axis 0.  NaNs are preserved (not ranked) and stay NaN in the output
    Works on 1D or 2D y
    """
    if y.ndim == 1:
        return _rint_1d(y, ties_method)
    out = np.full_like(y, np.nan, dtype=np.float64)
    for j in range(y.shape[1]):
        out[:, j] = _rint_1d(y[:, j], ties_method)
    return out


def _rint_1d(y: np.ndarray, ties_method: str) -> np.ndarray:
    finite = ~np.isnan(y)
    out = np.full_like(y, np.nan, dtype=np.float64)
    if not finite.any():
        return out
    ranks = rankdata(y[finite], method=ties_method)
    n = finite.sum()
    quantiles = (ranks - 0.5) / n  # blom-like offset, avoids the -inf / +inf at the edges
    out[finite] = norm.ppf(quantiles)
    return out
