"""
Spectral-transform LMM math, batched on torch
The trick: eigendecomposing K = U diag(s) Uᵀ diagonalises the residual covariance, so per-pheno fits become 1D
To port from lmm.py: LMM.setG/setK (eigendecomp + caching), plus LMM.findH2 + LMM.nLLeval for the h2 search
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


def fit_delta_grid(spectrum: Spectrum,
                   p: int = 0, 
                   *,
                   n_grid: int = 64,
                   log_delta_min: float = -10.0,
                   log_delta_max: float = 10.0) -> Tensor:
    """
    PORT IS DONE!
    1D grid search for the log delta that minimises nLLeval (lmm.py line 300 LMM.findH2, but I parametrise in log delta rather than h2)
    fastlmm does grid + Brent refinement via minimize1D in fastlmm/util/mingrid.py.
    Skipping it for now, a 64-point grid on [-10, 10] log delta is dense enough and a golden-section pass on top later should be okay most probabmy
    Per-pheno: pass the pheno index p, returns a scalar log delta
    """
    s, X_rot, Y_rot = spectrum.s, spectrum.X_rot, spectrum.Y_rot
    y = Y_rot[:, p]
    grid = torch.linspace(log_delta_min, log_delta_max, n_grid, dtype=s.dtype, device=s.device)
    losses = torch.stack([nLLeval(g.exp(), s, X_rot, y) for g in grid])
    return grid[losses.argmin()]


def snp_wald_scan(spectrum: Spectrum,
                  log_delta: Tensor,
                  S_rot: Tensor,
                  p: int = 0) -> Tensor:
    """
    PORT IS DONE!
    Closed-form Wald F-statistic for every SNP in S_rot at the fitted log delta (lmm.py line 540 LMM.getPosteriorWeights + per-SNP loop in single_snp.py _internal_single_snp around line 1300)
    In the rotated basis where V = sigma2_g diag(s + delta), for each test SNP s:
        beta_s = (s~ᵀ V⁻¹ r) / (s~ᵀ V⁻¹ s~) with r = residual from the null fit
        var_beta = rss_full / (N - C - 1) / (s~ᵀ V⁻¹ s~)
        F_stat = beta_s² / var_beta   ~ F(1, N-C-1) under the null
    fastlmm returns a dict with beta / p / h2 / etc out of lmm_cov.py.
    Returning F-stats only, p-values are a scipy.stats.f.sf call away
    Single-pheno + no chunking for nowand vectorising across phenos and breaking S_rot into chunks will come later if there are memories issues
    """
    s, X_rot, Y_rot = spectrum.s, spectrum.X_rot, spectrum.Y_rot
    y = Y_rot[:, p]
    N, C = X_rot.shape
    delta = log_delta.exp()

    w = 1.0 / (s + delta)  # diag of V⁻¹, per-strain weight
    WX = w.unsqueeze(-1) * X_rot  # weighted null design, cached
    A = WX.T @ X_rot  # X~ᵀ V⁻¹ X~
    L = torch.linalg.cholesky(A)  # factorise A once, reuse for every SNP below
    Xy = WX.T @ y  # X~ᵀ V⁻¹ y~
    beta_x = torch.cholesky_solve(Xy.unsqueeze(-1), L).squeeze(-1)  # null fixed-effect estimates
    r = y - X_rot @ beta_x  # residual under the null
    rWr = (w * r * r).sum()  # null residual SS in V-metric

    # per-SNP terms.  the projection (I - X (XᵀV⁻¹X)⁻¹ XᵀV⁻¹) is never formed explicitly, cholesky_solve on A handles it
    B = WX.T @ S_rot  # X~ᵀ V⁻¹ S~, shape (C, M)
    AinvB = torch.cholesky_solve(B, L)  # (C, M), reusing the L factor from the null fit
    sWs = (w.unsqueeze(-1) * S_rot * S_rot).sum(dim=0)  # s~ᵀ V⁻¹ s~ per SNP
    quad = (B * AinvB).sum(dim=0)  # s~ᵀ V⁻¹ X~ (X~ᵀV⁻¹X~)⁻¹ X~ᵀV⁻¹ s~, the X-projection chunk to subtract
    denom = torch.clamp(sWs - quad, min=1e-300)  # full denominator = s~ᵀ V⁻¹ (I - X-projection) s~

    wr = w * r  # weighted residual
    num = wr @ S_rot  # s~ᵀ V⁻¹ r per SNP, the numerator of beta_s
    beta = num / denom
    rss_full = torch.clamp(rWr - num * num / denom, min=1e-300)  # SSE under the alt = null SSE - beta_s * num
    var_beta = rss_full / (N - C - 1) / denom
    return (beta * beta) / var_beta  # F(1, N-C-1) statistic per SNP
