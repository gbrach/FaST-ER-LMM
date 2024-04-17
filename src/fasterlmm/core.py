"""
Spectral-transform LMM math, batched on torch
The trick: eigendecomposing K = U diag(s) Uᵀ diagonalises the residual covariance, so per-pheno fits become 1D
Targets to port from lmm.py: LMM.setG/setK (eigendecomp + caching), plus LMM.findH2 + LMM.nLLeval for the h2 search
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Spectrum:
    """
    K's eigendecomposition + rotated design matrices, ready for the scan
    s (N,) eigenvalues ascending
    U (N, N) eigenvectors
    X_rot (N, C) Uᵀ X
    Y_rot (N, P) Uᵀ Y
    """
    s: Tensor
    U: Tensor
    X_rot: Tensor
    Y_rot: Tensor


def eigendecompose(K: Tensor) -> tuple[Tensor, Tensor]:
    """
    PORT IS DONE! 
    Eigendecomposition in LMM.setG / LMM.setK (lmm.py:154-186)
    fastlmm does SVD on G then squares the singular values when k<N (faster), with an eigh fallback if SVD blows up
    Here I go straight to torch.linalg.eigh on K beacuse the cases I care about are full-rank (k>=N)
    Negative eigenvalues from float noise get clamped to 0
    """
    s, U = torch.linalg.eigh(K)
    s = torch.clamp(s, min=0.0)
    return s, U


def rotate(K: Tensor, X: Tensor, Y: Tensor) -> Spectrum:
    """
    PORT IS DONE!
    Uᵀ X and Uᵀ Y rotations in LMM.setX / LMM.setY (lmm.py:60-98)
    fastlmm splits these across setX (line 70 -> self.UX) and setY (line 94 -> self.Uy), each lazy on prior eigendecomp state.  Bundling them here makes the data dependency explicit -> you cant call rotate before you have K
    """
    s, U = eigendecompose(K)
    X_rot = U.T @ X
    Y_rot = U.T @ Y
    return Spectrum(s=s, U=U, X_rot=X_rot, Y_rot=Y_rot)


def nLLeval(
    delta: float | Tensor,
    s: Tensor,
    X_rot: Tensor,
    y_rot: Tensor) -> Tensor:
    """
    PORT IS DONE!
    Profiled -2*loglik at a given delta = sigma2_e / sigma2_g (lmm.py at 348 LMM.nLLeval, ML branch)
    With Sd = s + delta the math is:
        beta_hat = (X~ᵀ diag(1/Sd) X~)⁻¹ X~ᵀ diag(1/Sd) y~
        sigma2_g = (y~ - X~ beta)ᵀ diag(1/Sd) (y~ - X~ beta) / N
        -2*loglik = N (log(2*pi*sigma2_g) + 1) + sum log(Sd)
    Vectorising the loss across a (G grid x P pheno) tensor, the per-pheno call is fine while there's no scan yet
    """
    N, _ = X_rot.shape
    Sd = s + delta
    w = 1.0 / Sd
    WX = w.unsqueeze(-1) * X_rot
    A = WX.T @ X_rot
    Xy = WX.T @ y_rot
    beta = torch.linalg.solve(A, Xy)
    rWr = (y_rot * w * y_rot).sum() - (Xy * beta).sum()
    sigma2_g = rWr / N
    log2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=s.dtype, device=s.device))
    return N * (log2pi + sigma2_g.log() + 1.0) + Sd.log().sum()
