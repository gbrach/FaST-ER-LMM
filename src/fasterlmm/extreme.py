"""
Randomised top-r eigendecomposition for big-N kernels
Full eigh on K = Z Zᵀ is N³.  When N gets into the few-thousand range that's slow, and the top-r eigenpairs are usually enough for the LMM fit since the rest of the spectrum collapses to a flat noise floor
Halko-Martinsson-Tropp 2011 random projection: K Q ≈ Q (Qᵀ K Q), eigh the small Qᵀ K Q, multiply U back.  Chunked Z streaming for the N x N step is a later add, the in-memory version is fine on a single GPU up to a few thousand strains
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from fasterlmm.core import Spectrum, fit_delta_grid, snp_wald_scan
from fasterlmm.io import grm, standardise_columns


def randomized_eigh(Z: Tensor,
                    rank: int = 200,
                    *,
                    n_oversample: int = 20,
                    n_iter: int = 2,
                    seed: int = 19930909) -> tuple[Tensor, Tensor]:
    """
    Top-r eigendecomposition of K = Z Zᵀ / M via randomised range-finder
    Returns (s_top (r,), U_top (N, r)) with the rest of the spectrum discarded
    Halko et al SIREV 2011 algorithm 4.4 + 5.3 with n_iter power iterations to sharpen the tail.  Oversampling and droping at the end avoids the noisy boundary eigenpairs
    """
    N, M = Z.shape
    target = rank + n_oversample  # extra columns absorb the noisy boundary eigenpairs, dropped at the end
    gen = torch.Generator(device=Z.device).manual_seed(seed)
    Omega = torch.randn(N, target, generator=gen, device=Z.device, dtype=Z.dtype)  # Omega ~ N(0, I), random probe
    Y = (Z @ Z.T) @ Omega / M  # Y = K Omega, captures K's range in random directions
    for _ in range(n_iter):
        Y = (Z @ (Z.T @ Y)) / M  # Y <- K Y, subspace iteration sharpens the leading singular directions
    Q, _ = torch.linalg.qr(Y)  # orthonormal basis Q for the captured range, shape (N, target)
    B = Q.T @ (Z @ Z.T @ Q) / M  # B = Qᵀ K Q, the small (target, target) restriction of K to range(Q)
    s_small, U_small = torch.linalg.eigh(B)  # full eigh on B is much cheaper than eigh on K
    s_small = torch.clamp(s_small, min=0.0)
    idx = torch.argsort(s_small, descending=True)[:rank]  # drop the oversample columns, keep top rank
    return s_small[idx], Q @ U_small[:, idx]  # lift eigenvectors back to the N-dimensional space via U_K ≈ Q @ U_small


def loco_scan_extreme(Z: Tensor,
                      X: Tensor,
                      Y: Tensor,
                      chrom: np.ndarray,
                      *,
                      rank: int = 200) -> Tensor:
    """
    PORT IS DONE!
    Big-N LOCO scan using a top-r approximation of each K_loco.  Same shape as loco_scan but with randomized_eigh instead of the full eigh
    The Spectrum the scan sees has rank-r spectrum + (N, r) U, snp_wald_scan doesnt care about the actual rank as long as U.T @ X / U.T @ Y / U.T @ S_test stay consistant
    Accuracy tradeoff: the low-eigenvalue tail gets dropped so the residual covariance is slighty off, but in practice the leading eigenpairs dominate the loss
    """
    M = Z.shape[1]
    P = Y.shape[1]
    f_out = torch.zeros(M, P, dtype=Z.dtype, device=Z.device)
    for c in sorted(np.unique(chrom).tolist()):
        kin_mask = chrom != c
        test_mask = chrom == c
        Z_kin = Z[:, kin_mask]
        s_top, U_top = randomized_eigh(Z_kin, rank=rank)
        spec = Spectrum(s=s_top, U=U_top,
                        X_rot=U_top.T @ X,
                        Y_rot=U_top.T @ Y)
        log_delta = fit_delta_grid(spec)
        S_rot = U_top.T @ Z[:, test_mask]
        f_out[test_mask, :] = snp_wald_scan(spec, log_delta, S_rot)
    return f_out
