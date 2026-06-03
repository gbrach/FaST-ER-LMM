"""
Phenotype normalisations.  Just RINT (rank-inverse-normal transform) for now -- the workhorse for non-gaussian phenos like raw transcript counts where outliers tank the perm threshold
Canonical Blom (c=3/8) to match Victor's R helper (workflow/scripts/rank_based_Inverse_Normal_Transformation.R) so this is a drop-in for the starlight pipeline
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def rint_columns(Y: np.ndarray, *,
                 c: float = 3 / 8, ties: str = "average") -> np.ndarray:
    """
    Per-column Blom RINT.  out = qnorm((rank - c) / (n - 2c + 1)) with c=3/8 by default, ties.method='average' to match R's rank
    NaNs round-trip (not ranked, stay NaN in the output)
    Operates along axis 0, accepts 1D or 2D
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        return _rint_1d(Y, c, ties)
    if Y.ndim != 2:
        raise ValueError(f"rint_columns expects 1D or 2D, got shape {Y.shape}")
    out = np.full_like(Y, np.nan)
    for j in range(Y.shape[1]):
        out[:, j] = _rint_1d(Y[:, j], c, ties)
    return out


def _rint_1d(y: np.ndarray, c: float, ties: str) -> np.ndarray:
    """
    Blom RINT of one column, NaNs left untouched
    Ranks only the finite entries, maps them trough the normal quantile, scatters them back into a nan-filled output so the missing cells round-trip
    """
    out = np.full_like(y, np.nan, dtype=np.float64)
    mask = ~np.isnan(y)
    n = int(mask.sum())
    if n == 0:
        return out
    ranks = rankdata(y[mask], method=ties)
    out[mask] = norm.ppf((ranks - c) / (n - 2 * c + 1))
    return out


def rint(y: np.ndarray, *, ties_method: str = "average") -> np.ndarray:
    """
    Thin alias for rint_columns under the old name
    Kept so anything that imported rint from the previous (unused) version still resolves, can drop it once nothing points here
    """
    return rint_columns(y, ties=ties_method)
