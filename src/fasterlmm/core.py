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
