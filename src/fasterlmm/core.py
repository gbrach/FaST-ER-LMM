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


def _eigh_with_cpu_fallback(K: Tensor) -> tuple[Tensor, Tensor]:
    """
    Eigh on K's device, falling back to cpu lapack on cuda OOM
    cuSOLVER's eigh workspace is 3-8x the input.  At large N the workspace OOMs even when K, U, s themselves fit on the GPU.  Shipping K to cpu, calling lapack eigh, and bringing s and U back is slow but it's the only thing that works at the edge
    """
    try:
        return torch.linalg.eigh(K)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        device = K.device
        K_cpu = K.detach().cpu()
        del K
        if device.type == "cuda":
            torch.cuda.empty_cache()
        s, U = torch.linalg.eigh(K_cpu)
        return s.to(device), U.to(device)


def eigendecompose(K: Tensor, jitter: float = 1e-6) -> tuple[Tensor, Tensor]:
    """
    PORT IS DONE!
    Eigendecomposition in LMM.setG / LMM.setK (lmm.py lines 154-186)
    fastlmm does SVD on G then squares the singular values when k<N (faster), with an eigh fallback if SVD blows up
    Going straight to torch.linalg.eigh on K beacuse the cases that matter are full-rank (k>=N), with a tiny diagonal jitter for safety on near-singular K and a cpu lapack fallback if cuSOLVER's workspace OOMs
    """
    N = K.shape[0]
    Kj = K + jitter * torch.eye(N, dtype=K.dtype, device=K.device)
    s, U = _eigh_with_cpu_fallback(Kj)
    s = torch.clamp(s, min=0.0)  # negative eigenvalues from float noise get clamped to 0
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
    N, C = X_rot.shape
    delta = log_delta.exp()  # (G,) deltas in linear scale
    Sd = s.unsqueeze(0) + delta.unsqueeze(1)  # Sd[g, n] = s[n] + delta[g], diagonal of V per grid point
    w = 1.0 / Sd  # w[g, n] = 1/Sd[g, n], per-strain weight in the WLS for grid point g
    WX = w.unsqueeze(2) * X_rot.unsqueeze(0)  # WX[g, n, c] = w[g, n] * X_rot[n, c], cached weighted design
    A = torch.einsum("gnc,nd->gcd", WX, X_rot)  # A[g] = X~ᵀ V⁻¹ X~, the (C, C) Gram per grid point
    # cholesky_ex returns an info code rather than raising.  At extreme deltas A can drift non-PD on a few grid points, masking the bad rows with identity keeps cholesky_solve finite for the good rows and the bad rows get +inf loss at the end so argmin never picks them
    L, info = torch.linalg.cholesky_ex(A)
    bad_g = info > 0  # (G,)
    if bad_g.any():
        eye = torch.eye(C, device=A.device, dtype=A.dtype).expand_as(L)
        L = torch.where(bad_g.view(-1, 1, 1), eye, L)
    u = torch.einsum("gnc,np->gcp", WX, Y_rot)  # u[g, c, p] = X~ᵀ V⁻¹ Y_rot, rhs of the normal eqs
    beta_x = torch.cholesky_solve(u, L)  # beta_hat[g, c, p] = A⁻¹ u, fixed-effect estimates
    uAu = (u * beta_x).sum(dim=1)  # uAu[g, p] = uᵀ A⁻¹ u = sum_c u[g, c, p] * beta_x[g, c, p]
    yWy = w @ (Y_rot * Y_rot)  # yWy[g, p] = sum_n w[g, n] * Y_rot[n, p]², gemm avoids a (G, N, P) tensor
    rWr = torch.clamp(yWy - uAu, min=1e-300)  # rWr = yWy - uAu = (y~ - X~ beta)ᵀ V⁻¹ (y~ - X~ beta)
    sigma2 = rWr / N  # sigma2_g_hat at this (g, pheno), ML divides by N
    sum_log_Sd = Sd.log().sum(dim=1)  # log det V at sigma2=1: sum_n log Sd[g, n]
    log2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=s.dtype, device=s.device))
    loss = N * (log2pi + sigma2.log() + 1.0) + sum_log_Sd.unsqueeze(1)  # (G, P)
    if bad_g.any():
        loss = torch.where(bad_g.unsqueeze(1), torch.full_like(loss, float("inf")), loss)
    return loss


def fit_delta_grid(spectrum: Spectrum,
                   *,
                   n_grid: int = 64,
                   log_delta_min: float = -10.0,
                   log_delta_max: float = 10.0) -> Tensor:
    """
    Multi-pheno 1D grid search for log delta, returns shape (P,)
    Uses _profile_loss vectorised over (G, P), so adding more phenos is basically free until G*N*P stops fitting on the GPU
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
    delta = log_delta.exp()  # (P,) deltas in linear scale, one per pheno

    w = 1.0 / (s.unsqueeze(0) + delta.unsqueeze(1))  # w[p, n] = 1/(s[n] + delta[p]), per-pheno V⁻¹ diagonal
    WX = w.unsqueeze(2) * X_rot.unsqueeze(0)  # WX[p, n, c] = w[p, n] * X_rot[n, c], weighted null design per pheno
    A = torch.einsum("pnc,nd->pcd", WX, X_rot)  # A[p] = X~ᵀ V⁻¹ X~, the (C, C) Gram per pheno
    # cholesky_ex tolerates the occasional non-PD A from a poorly-fit delta on a perm column.  Bad rows get identity here and their F-stats end up at zero by the math below, much nicer than the whole batch raising
    L, info = torch.linalg.cholesky_ex(A)
    bad_p = info > 0
    if bad_p.any():
        eye = torch.eye(C, device=A.device, dtype=A.dtype).expand_as(L)
        L = torch.where(bad_p.view(-1, 1, 1), eye, L)
    u = torch.einsum("pnc,np->pc", WX, Y_rot)  # u[p, c] = X~ᵀ V⁻¹ Y_rot[:, p], rhs of per-pheno normal eqs
    beta_x = torch.cholesky_solve(u.unsqueeze(-1), L).squeeze(-1)  # beta_x[p, c] = A[p]⁻¹ u[p], null beta_X
    r = Y_rot - X_rot @ beta_x.T  # r[n, p] = Y_rot[n, p] - X_rot[n, :] @ beta_x[p, :], null residual per pheno
    rWr = (w * r.T * r.T).sum(dim=1)  # rWr[p] = sum_n w[p, n] * r[n, p]², null SS in V-metric

    B = torch.einsum("pnc,nm->pcm", WX, S_rot)  # B[p, c, m] = X~ᵀ V⁻¹ S~_m for SNP m, pheno p
    AinvB = torch.cholesky_solve(B, L)  # A[p]⁻¹ B[p], reuses the null fit's L factor per pheno
    sWs = w @ (S_rot * S_rot)  # sWs[p, m] = sum_n w[p, n] * S_rot[n, m]² = s~_mᵀ V⁻¹ s~_m
    quad = (B * AinvB).sum(dim=1)  # quad[p, m] = s~_mᵀ V⁻¹ X~ A⁻¹ X~ᵀ V⁻¹ s~_m, X-projection chunk
    denom = torch.clamp(sWs - quad, min=1e-300)  # denom[p, m] = s~_mᵀ V⁻¹ (I - X-projection) s~_m
    wr = w * r.T  # wr[p, n] = w[p, n] * r[n, p], weighted residual per pheno
    num = wr @ S_rot  # num[p, m] = s~_mᵀ V⁻¹ r, numerator of beta_s per (pheno, SNP)
    beta = num / denom  # beta_s[p, m] = num[p, m] / denom[p, m], MLE of the SNP effect at this pheno's delta
    rss_full = torch.clamp(rWr.unsqueeze(1) - num * num / denom, min=1e-300)  # SSE under the alt model
    var_beta = rss_full / (N - C - 1) / denom  # Var(beta_s) at df = N - C - 1 (matches fastlmm's single_snp.py:1415)
    f_stat = (beta * beta) / var_beta  # F(1, N-C-1) statistic per (pheno, SNP)
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
