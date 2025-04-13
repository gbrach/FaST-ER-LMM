"""
Spectral-transform LMM math, batched on torch
The trick: eigendecomposing K = U diag(s) Uᵀ diagonalises the residual covariance, so per-pheno fits become 1D
To port from lmm.py: LMM.setG/setK (eigendecomp + caching), plus LMM.findH2 + LMM.nLLeval for the h2 search
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from fasterlmm.io import grm, standardise_columns


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
    Eigendecomposition in LMM.setG / LMM.setK (lmm.py lines 154-186)
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


def nLLeval(delta: float | Tensor,
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
    Sd = s + delta  # diagonal of V in the rotated basis (V = sigma2_g (diag(s) + delta I))
    w = 1.0 / Sd  # 1/(s+delta) is the diagonal of V⁻¹, acts as a per-strain weight in the WLS below
    WX = w.unsqueeze(-1) * X_rot  # weighted rotated design, cached because A and Xy both need it
    A = WX.T @ X_rot  # A = X~ᵀ V⁻¹ X~
    Xy = WX.T @ y_rot  # X~ᵀ V⁻¹ y~, the right-hand side of the normal equation A beta = Xy
    beta = torch.linalg.solve(A, Xy)  # beta_hat = A⁻¹ Xy, fixed-effect estimate at this delta
    rWr = (y_rot * w * y_rot).sum() - (Xy * beta).sum()  # residual SS in V-metric, identity avoids forming y - X beta
    sigma2_g = rWr / N  # MLE of sigma2_g with beta profiled out, ML uses N
    log2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=s.dtype, device=s.device))
    return N * (log2pi + sigma2_g.log() + 1.0) + Sd.log().sum()  # -2 loglik, +1 from profiling sigma2_g back in, sum log(Sd) is log det V


def _profile_loss(log_delta: Tensor,
                  s: Tensor,
                  X_rot: Tensor,
                  Y_rot: Tensor) -> Tensor:
    """
    PORT IS DONE!
    Vectorised -2*loglik on a (G grid x P pheno) tensor, same ML math as nLLeval just broadcast across both axes (lmm.py LMM.nLLeval, ML branch)
    Returns shape (G, P).  Replaces the per-pheno python loop in fit_delta_grid -- perms drag in hundreds of pheno columns so the loop was the bottleneck
    """
    N, _ = X_rot.shape
    delta = log_delta.exp()  # (G,)
    Sd = s.unsqueeze(0) + delta.unsqueeze(1)  # (G, N)
    w = 1.0 / Sd  # (G, N)
    WX = w.unsqueeze(2) * X_rot.unsqueeze(0)  # (G, N, C), weighted design per grid point
    A = torch.einsum("gnc,nd->gcd", WX, X_rot)  # (G, C, C), Gram per grid point
    L = torch.linalg.cholesky(A)  # (G, C, C)
    u = torch.einsum("gnc,np->gcp", WX, Y_rot)  # (G, C, P), rhs of normal eqs per (g, pheno)
    beta_x = torch.cholesky_solve(u, L)  # (G, C, P)
    uAu = (u * beta_x).sum(dim=1)  # (G, P), beta^T A beta
    yWy = w @ (Y_rot * Y_rot)  # (G, P), gemm avoids forming a (G, N, P) tensor
    rWr = torch.clamp(yWy - uAu, min=1e-300)  # residual SS in V-metric
    sigma2 = rWr / N  # MLE per (g, pheno)
    sum_log_Sd = Sd.log().sum(dim=1)  # (G,), log det V (up to sigma2_g scale)
    log2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=s.dtype, device=s.device))
    return N * (log2pi + sigma2.log() + 1.0) + sum_log_Sd.unsqueeze(1)  # (G, P)


def fit_delta_grid(spectrum: Spectrum,
                   *,
                   n_grid: int = 64,
                   log_delta_min: float = -10.0,
                   log_delta_max: float = 10.0) -> Tensor:
    """
    Multi-pheno 1D grid search for log delta, returns shape (P,)
    Uses _profile_loss vectorised over (G, P), so adding more phenos is basically free until G*N*P stops fitting on the gpu
    """
    s, X_rot, Y_rot = spectrum.s, spectrum.X_rot, spectrum.Y_rot
    grid = torch.linspace(log_delta_min, log_delta_max, n_grid, dtype=s.dtype, device=s.device)
    loss = _profile_loss(grid, s, X_rot, Y_rot)  # (G, P)
    return grid[loss.argmin(dim=0)]  # (P,)


def snp_wald_scan(spectrum: Spectrum,
                  log_delta: Tensor,
                  S_rot: Tensor) -> Tensor:
    """
    PORT IS DONE!
    Multi-pheno per-(SNP, pheno) Wald F-stat at the per-pheno fitted log delta, returns (M, P) (lmm.py LMM.getPosteriorWeights + the per-SNP loop in _internal_single_snp around line 1300)
    Generalises the single-pheno scan to a (P,) log_delta vector, every per-SNP term now broadcasts a P axis.  Way fewer python overhead per perm column than looping snp_wald_scan
    """
    s, X_rot, Y_rot = spectrum.s, spectrum.X_rot, spectrum.Y_rot
    N, C = X_rot.shape
    delta = log_delta.exp()  # (P,)

    w = 1.0 / (s.unsqueeze(0) + delta.unsqueeze(1))  # (P, N), per-pheno diag of V⁻¹
    WX = w.unsqueeze(2) * X_rot.unsqueeze(0)  # (P, N, C), weighted null design per pheno
    A = torch.einsum("pnc,nd->pcd", WX, X_rot)  # (P, C, C)
    L = torch.linalg.cholesky(A)
    u = torch.einsum("pnc,np->pc", WX, Y_rot)  # (P, C), rhs of per-pheno normal eqs
    beta_x = torch.cholesky_solve(u.unsqueeze(-1), L).squeeze(-1)  # (P, C)
    r = Y_rot - X_rot @ beta_x.T  # (N, P), per-pheno null residual
    rWr = (w * r.T * r.T).sum(dim=1)  # (P,) null SS in V-metric

    B = torch.einsum("pnc,nm->pcm", WX, S_rot)  # (P, C, M)
    AinvB = torch.cholesky_solve(B, L)  # (P, C, M)
    sWs = w @ (S_rot * S_rot)  # (P, M), s~ᵀ V⁻¹ s~
    quad = (B * AinvB).sum(dim=1)  # (P, M), X-projection chunk
    denom = torch.clamp(sWs - quad, min=1e-300)  # (P, M)
    wr = w * r.T  # (P, N)
    num = wr @ S_rot  # (P, M)
    beta = num / denom
    rss_full = torch.clamp(rWr.unsqueeze(1) - num * num / denom, min=1e-300)
    var_beta = rss_full / (N - C - 1) / denom
    f_stat = (beta * beta) / var_beta  # (P, M)
    return f_stat.T  # (M, P)


def loco_scan(Z: Tensor,
              X: Tensor,
              Y: Tensor,
              chrom: np.ndarray) -> Tensor:
    """
    PORT IS DONE!
    Multi-pheno Leave-One-Chromosome-Out scan, ports the LocoGwas path in single_snp.py (_internal_single_snp_LocoGwas around line 1100). Y is (N, P), returns (M, P) F-stats
    For each chromosome c the kinship is rebuilt without c, refit per-pheno log delta against K_loco, snp_wald_scan on c-only SNPs.
    Looping chromosomes here is fewer moving parts than fastlmm's LocoGwas + SnpReader subset and lets the GPU chew through the per-chrom eigh wihtout going back to pysnptools
    Z must be pre-standardised (use io.standardise_columns)
    """
    M = Z.shape[1]
    P = Y.shape[1]
    f_out = torch.zeros(M, P, dtype=Z.dtype, device=Z.device)
    for c in sorted(np.unique(chrom).tolist()):
        kin_mask = chrom != c
        test_mask = chrom == c
        K = grm(Z[:, kin_mask])  # K_loco for this chromosome
        spec = rotate(K, X, Y)
        log_delta = fit_delta_grid(spec)  # (P,)
        S_rot = spec.U.T @ Z[:, test_mask]
        f_out[test_mask, :] = snp_wald_scan(spec, log_delta, S_rot)
    return f_out
