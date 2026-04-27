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
                  Y_rot: Tensor,
                  *,
                  reml: bool = False) -> Tensor:
    """
    PORT IS DONE!
    Vectorised -2*loglik on a (G grid x P pheno) tensor, same algebra as nLLeval just broadcast across both axes (lmm.py LMM.nLLeval). ->> Returns shape (G, P)
    reml=False (default) is the ML branch.
    reml=True adds the X-side penalty term log det A and divides by (N - C) instead of N, same as fastlmm's REML
    Replaces the per-pheno python loop in fit_delta_grid since perms drag in hundreds of pheno columns and the loop was the bottleneck
    """
    N, C = X_rot.shape
    delta = log_delta.exp()  # (G,) deltas in linear scale
    Sd = s.unsqueeze(0) + delta.unsqueeze(1)  # Sd[g, n] = s[n] + delta[g], diagonal of V per grid point
    w = 1.0 / Sd  # w[g, n] = 1/Sd[g, n], per-strain weight in the WLS for grid point g
    WX = w.unsqueeze(2) * X_rot.unsqueeze(0)  # WX[g, n, c] = w[g, n] * X_rot[n, c], cached weighted design
    A = torch.einsum("gnc,nd->gcd", WX, X_rot)  # A[g] = X~ᵀ V⁻¹ X~, the (C, C) Gram per grid point
    L, info = torch.linalg.cholesky_ex(A)
    bad_g = info > 0
    if bad_g.any():
        eye = torch.eye(C, device=A.device, dtype=A.dtype).expand_as(L)
        L = torch.where(bad_g.view(-1, 1, 1), eye, L)
    logdet_A = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)  # (G,), needed for the REML penalty
    u = torch.einsum("gnc,np->gcp", WX, Y_rot)  # u[g, c, p] = X~ᵀ V⁻¹ Y_rot, rhs of the normal eqs
    beta_x = torch.cholesky_solve(u, L)  # beta_hat[g, c, p] = A⁻¹ u, fixed-effect estimates
    uAu = (u * beta_x).sum(dim=1)  # uAu[g, p] = uᵀ A⁻¹ u
    yWy = w @ (Y_rot * Y_rot)  # yWy[g, p] = sum_n w[g, n] * Y_rot[n, p]²
    rWr = torch.clamp(yWy - uAu, min=1e-300)  # rWr = (y~ - X~ beta)ᵀ V⁻¹ (y~ - X~ beta)
    sum_log_Sd = Sd.log().sum(dim=1)  # log det V at sigma2=1
    log2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=s.dtype, device=s.device))
    if reml:
        sigma2 = rWr / (N - C)
        loss = (N - C) * (log2pi + sigma2.log() + 1.0) + sum_log_Sd.unsqueeze(1) + logdet_A.unsqueeze(1)
    else:
        sigma2 = rWr / N
        loss = N * (log2pi + sigma2.log() + 1.0) + sum_log_Sd.unsqueeze(1)
    if bad_g.any():
        loss = torch.where(bad_g.unsqueeze(1), torch.full_like(loss, float("inf")), loss)
    return loss


def _profile_loss_per_pheno(log_delta: Tensor,
                            s: Tensor,
                            X_rot: Tensor,
                            Y_rot: Tensor,
                            *,
                            reml: bool = False) -> Tensor:
    """
    Per-pheno _profile_loss for the golden-section refinment: one log_delta value per pheno, returns (P,)
    Same algebra as _profile_loss but the grid axis is replaced by the pheno axis, so each pheno evaluates only its own log delta.  Way cheaper than rebuilding a (G, P) grid just to read one column per pheno
    """
    N, C = X_rot.shape
    delta = log_delta.exp()  # (P,)
    Sd = s.unsqueeze(0) + delta.unsqueeze(1)  # (P, N)
    w = 1.0 / Sd
    WX = w.unsqueeze(2) * X_rot.unsqueeze(0)  # (P, N, C)
    A = torch.einsum("pnc,nd->pcd", WX, X_rot)  # (P, C, C)
    L, info = torch.linalg.cholesky_ex(A)
    bad_p = info > 0
    if bad_p.any():
        eye = torch.eye(C, device=A.device, dtype=A.dtype).expand_as(L)
        L = torch.where(bad_p.view(-1, 1, 1), eye, L)
    logdet_A = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)  # (P,)
    u = torch.einsum("pnc,np->pc", WX, Y_rot)  # (P, C)
    beta_x = torch.cholesky_solve(u.unsqueeze(-1), L).squeeze(-1)  # (P, C)
    uAu = (u * beta_x).sum(dim=1)  # (P,)
    yWy = (w * Y_rot.T * Y_rot.T).sum(dim=1)  # (P,)
    rWr = torch.clamp(yWy - uAu, min=1e-300)
    sum_log_Sd = Sd.log().sum(dim=1)  # (P,)
    log2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=s.dtype, device=s.device))
    if reml:
        sigma2 = rWr / (N - C)
        loss = (N - C) * (log2pi + sigma2.log() + 1.0) + sum_log_Sd + logdet_A
    else:
        sigma2 = rWr / N
        loss = N * (log2pi + sigma2.log() + 1.0) + sum_log_Sd
    if bad_p.any():
        loss = torch.where(bad_p, torch.full_like(loss, float("inf")), loss)
    return loss


def fit_delta_grid(spectrum: Spectrum,
                   *,
                   n_grid: int = 64,
                   log_delta_min: float = -10.0,
                   log_delta_max: float = 10.0,
                   refine: bool = True,
                   reml: bool = False) -> Tensor:
    """
    Multi-pheno log-delta fit: 64-point grid then optional golden-section refinment, returns (P,)
    fastlmm's findH2 uses grid + Brent via minimize1D (fastlmm/util/mingrid.py).  Golden-section here gets to ~1e-10 in log delta after 50 iterations.  refine=False just returns the grid argmin if speed matters more than precision
    reml=False (default) matches the ML branch in _profile_loss, reml=True uses the same REML loss in both the grid and the refinment
    """
    s, X_rot, Y_rot = spectrum.s, spectrum.X_rot, spectrum.Y_rot
    grid = torch.linspace(log_delta_min, log_delta_max, n_grid, dtype=s.dtype, device=s.device)
    loss = _profile_loss(grid, s, X_rot, Y_rot, reml=reml)  # (G, P)
    idx = loss.argmin(dim=0)  # (P,)
    if not refine:
        return grid[idx]

    # vectorised golden-section refinment on [grid[idx-1], grid[idx+1]] per pheno.  50 iters gets to ~1e-10 in log delta wich is way past what the scan needs
    lo_idx = torch.clamp(idx - 1, 0, n_grid - 1)
    hi_idx = torch.clamp(idx + 1, 0, n_grid - 1)
    a = grid[lo_idx].clone()
    b = grid[hi_idx].clone()
    inv_phi = 2.0 / (1.0 + 5.0 ** 0.5)
    inv_phi2 = inv_phi * inv_phi
    c = a + inv_phi2 * (b - a)
    d = a + inv_phi  * (b - a)
    fc = _profile_loss_per_pheno(c, s, X_rot, Y_rot, reml=reml)
    fd = _profile_loss_per_pheno(d, s, X_rot, Y_rot, reml=reml)
    for _ in range(50):
        cond = fc < fd
        b = torch.where(cond, d, b)
        a = torch.where(cond, a, c)
        c = a + inv_phi2 * (b - a)
        d = a + inv_phi  * (b - a)
        fc = _profile_loss_per_pheno(c, s, X_rot, Y_rot, reml=reml)
        fd = _profile_loss_per_pheno(d, s, X_rot, Y_rot, reml=reml)
    return (0.5 * (a + b)).clamp(log_delta_min, log_delta_max)


def snp_wald_scan(spectrum: Spectrum,
                  log_delta: Tensor,
                  S_rot: Tensor,
                  *,
                  snp_chunk: int = 4096,
                  pheno_chunk: int = 256) -> Tensor:
    """
    PORT IS DONE!
    Multi-pheno per-(SNP, pheno) Wald F-stat at the per-pheno fitted log delta, returns (M, P) (lmm.py LMM.getPosteriorWeights + the per-SNP loop in _internal_single_snp around line 1300)
    Generalises the single-pheno scan to a (P,) log_delta vector, every per-SNP term broadcats a P axis.  snp_chunk / pheno_chunk tile both axis so peak memory stays bounded at (pheno_chunk, N, snp_chunk).  On a 32GB V100 with N=1k that handles (256, 4096) comfortbly, bigger N just srhinks the defaults
    The per-pheno null fit (cholesky_ex on A, beta_x, residual r, rWr) gets computed once per pheno chunk and reused for each SNP chunk insde it
    """
    s, X_rot, Y_rot = spectrum.s, spectrum.X_rot, spectrum.Y_rot
    N, C = X_rot.shape
    P = Y_rot.shape[1]
    M = S_rot.shape[1]
    out = torch.zeros(M, P, dtype=s.dtype, device=s.device)
    for p_start in range(0, P, pheno_chunk):
        p_end = min(p_start + pheno_chunk, P)
        Y_c = Y_rot[:, p_start:p_end]  # (N, Pc)
        ld_c = log_delta[p_start:p_end]
        delta_c = ld_c.exp()  # (Pc,)

        w = 1.0 / (s.unsqueeze(0) + delta_c.unsqueeze(1))  # (Pc, N) per-pheno V⁻¹ diagonal
        WX = w.unsqueeze(2) * X_rot.unsqueeze(0)  # (Pc, N, C)
        A = torch.einsum("pnc,nd->pcd", WX, X_rot)  # (Pc, C, C)
        L, info = torch.linalg.cholesky_ex(A)
        bad_p = info > 0
        if bad_p.any():
            eye = torch.eye(C, device=A.device, dtype=A.dtype).expand_as(L)
            L = torch.where(bad_p.view(-1, 1, 1), eye, L)
        u = torch.einsum("pnc,np->pc", WX, Y_c)  # (Pc, C)
        beta_x = torch.cholesky_solve(u.unsqueeze(-1), L).squeeze(-1)  # (Pc, C)
        r = Y_c - X_rot @ beta_x.T  # (N, Pc) null residual per pheno
        rWr = (w * r.T * r.T).sum(dim=1)  # (Pc,) null SS in V-metric
        wr = w * r.T  # (Pc, N) weighted residual, cached across SNP chunks

        for m_start in range(0, M, snp_chunk):
            m_end = min(m_start + snp_chunk, M)
            S_c = S_rot[:, m_start:m_end]  # (N, Mc)
            B = torch.einsum("pnc,nm->pcm", WX, S_c)  # (Pc, C, Mc)
            AinvB = torch.cholesky_solve(B, L)  # (Pc, C, Mc)
            sWs = w @ (S_c * S_c)  # (Pc, Mc)
            quad = (B * AinvB).sum(dim=1)  # (Pc, Mc)
            denom = torch.clamp(sWs - quad, min=1e-300)  # (Pc, Mc)
            num = wr @ S_c  # (Pc, Mc)
            beta = num / denom
            rss_full = torch.clamp(rWr.unsqueeze(1) - num * num / denom, min=1e-300)
            var_beta = rss_full / (N - C - 1) / denom  # df = N - C - 1, matches single_snp.py:1415
            out[m_start:m_end, p_start:p_end] = ((beta * beta) / var_beta).T
    return out


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
