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


@dataclass
class ScanResult:
    """
    Per-(SNP, pheno) Wald outputs for the real phenos, plus the per-column genome max
    f, beta, se, sfve, nullh2 are all (M, n_real), the leading real-pheno columns of the scan
    max_F is (P,) and covers every column, reals and perms both, since the perm columns only ever keep this running genome max and never the full per-SNP detail
    sfve is fastlmm's SnpFractVarExpl, nullh2 is the per-SNP null-model h2 (chromosome-specific under LOCO)
    """
    f: Tensor
    beta: Tensor
    se: Tensor
    sfve: Tensor
    nullh2: Tensor
    max_F: Tensor


def _eigh_with_cpu_fallback(K: Tensor) -> tuple[Tensor, Tensor]:
    """
    Eigh on K's device, falling back to cpu lapack on cuda OOM
    cuSOLVER's eigh workspace is 3-8x the input.  At large N the workspace OOMs even when K, U, s themselves fit on the GPU.  Shipping K to cpu, calling lapack eigh, and bringing s and U back is slow but it's the only thing that works at the edge
    mps has no symmetric-eigh kernel at all, so a metal K detours through cpu lapack every time -- it's once per chromosome so the round-trip barely registers next to the scan
    """
    if K.device.type == "mps":
        s, U = torch.linalg.eigh(K.detach().cpu())
        return s.to(K.device), U.to(K.device)
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


def _pos_floor(dtype: torch.dtype) -> float:
    """
    Smallest positive clamp floor that still survives the dtype
    1e-300 is fine in float64 but underflows clean to 0 in float32, and a 0 floor turns a constant column's denominator into a 0/0 nan, so the float32 path gets a representable floor insted
    """
    return 1e-300 if dtype == torch.float64 else 1e-30


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
    rWr = torch.clamp(yWy - uAu, min=_pos_floor(s.dtype))  # rWr = (y~ - X~ beta)ᵀ V⁻¹ (y~ - X~ beta)
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
    rWr = torch.clamp(yWy - uAu, min=_pos_floor(s.dtype))
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


def _auto_tile(N: int, C: int, M: int, P: int,
               device: torch.device) -> tuple[int, int]:
    """
    Pick the pheno / snp tile sizes for snp_wald_scan from the GPU memory free right now
    The inner loop's working set is roughly 8 * (Pc*N*C + N*Mc + (2*C + T)*Pc*Mc) bytes for a pheno tile Pc and a snp tile Mc, with T counting the dozen-ish (Pc, Mc) temporaries it keeps alive at once.  Budgeting half the free VRAM for that leaves slack for the cholesky workspace and allocator fragmentation
    one budget, two tiles fall out of it -- no machine-specific hand tuning.  cpu / mps have no mem_get_info so they fall back to a fixed 256 / 4096
    """
    if device.type != "cuda":
        return min(256, P), min(4096, M)
    free_bytes = torch.cuda.mem_get_info(device)[0]
    budget = int(free_bytes * 0.5)
    T = 12  # count of (Pc, Mc) intermediates the inner loop holds at once
    snp_chunk = min(M, 16384)
    while snp_chunk >= 512:
        per_pc = 8 * (N * C + (2 * C + T) * snp_chunk)  # bytes per unit of pheno tile
        fixed = 8 * N * snp_chunk  # the S_c tile, pheno-independent
        pheno_chunk = (budget - fixed) // max(per_pc, 1)
        if pheno_chunk >= 64:
            return int(min(pheno_chunk, P)), int(snp_chunk)
        snp_chunk //= 2  # snp tile too fat for even 64 phenos, halve it and retry
    return min(64, P), min(512, M)


def snp_wald_scan(spectrum: Spectrum,
                  log_delta: Tensor,
                  S_rot: Tensor,
                  *,
                  snp_chunk: int | None = None,
                  pheno_chunk: int | None = None,
                  n_real: int | None = None) -> ScanResult:
    """
    PORT IS DONE!
    Multi-pheno per-(SNP, pheno) Wald F-stat at the per-pheno fitted log delta (lmm.py LMM.getPosteriorWeights + the per-SNP loop in _internal_single_snp around line 1300)
    Generalises the single-pheno scan to a (P,) log_delta vector, every per-SNP term broadcats a P axis.  snp_chunk / pheno_chunk tile both axis, and left None they get auto-sized by _auto_tile to fill about half the free VRAM -- no hand tuning per machine
    The per-pheno null fit (cholesky_ex on A, beta_x, residual r, rWr) gets computed once per pheno chunk and reused for each SNP chunk insde it
    n_real splits the pheno axis: the first n_real columns are real phenos and keep their full per-SNP detail (F, beta, SE, SnpFractVarExpl), the rest are permutation columns and only feed a per-column running max.  That way a perm batch of thousands of columns never materialises an (M, P) tensor -- the threshold only ever wants the per-perm genome max anyway.  Returns a ScanResult
    """
    s, X_rot, Y_rot = spectrum.s, spectrum.X_rot, spectrum.Y_rot
    N, C = X_rot.shape
    P = Y_rot.shape[1]
    M = S_rot.shape[1]
    if pheno_chunk is None or snp_chunk is None:
        auto_pc, auto_mc = _auto_tile(N, C, M, P, s.device)
        pheno_chunk = auto_pc if pheno_chunk is None else pheno_chunk
        snp_chunk = auto_mc if snp_chunk is None else snp_chunk
    if n_real is None:
        n_real = P
    n_real = min(n_real, P)
    f_real = torch.zeros(M, n_real, dtype=s.dtype, device=s.device)
    beta_real = torch.zeros(M, n_real, dtype=s.dtype, device=s.device)
    se_real = torch.zeros(M, n_real, dtype=s.dtype, device=s.device)
    sfve_real = torch.zeros(M, n_real, dtype=s.dtype, device=s.device)
    max_F = torch.full((P,), float("-inf"), dtype=s.dtype, device=s.device)
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
        real_end = min(p_end, n_real)  # last real-pheno column landing in this chunk

        for m_start in range(0, M, snp_chunk):
            m_end = min(m_start + snp_chunk, M)
            S_c = S_rot[:, m_start:m_end]  # (N, Mc)
            B = torch.einsum("pnc,nm->pcm", WX, S_c)  # (Pc, C, Mc)
            AinvB = torch.cholesky_solve(B, L)  # (Pc, C, Mc)
            sWs = w @ (S_c * S_c)  # (Pc, Mc)
            quad = (B * AinvB).sum(dim=1)  # (Pc, Mc)
            denom = torch.clamp(sWs - quad, min=_pos_floor(s.dtype))  # (Pc, Mc)
            num = wr @ S_c  # (Pc, Mc)
            beta = num / denom
            rss_full = torch.clamp(rWr.unsqueeze(1) - num * num / denom, min=_pos_floor(s.dtype))
            var_beta = rss_full / (N - C - 1) / denom  # df = N - C - 1, matches single_snp.py:1415
            f_chunk = (beta * beta) / var_beta  # (Pc, Mc) Wald F per (pheno, SNP)
            # per-column running max over the genome -- that's the whole perm threshold
            max_F[p_start:p_end] = torch.maximum(max_F[p_start:p_end], f_chunk.max(dim=1).values)
            # full per-SNP detail kept only for the real phenos, perm columns stop at the max above
            if real_end > p_start:
                rsl = real_end - p_start
                f_real[m_start:m_end, p_start:real_end] = f_chunk[:rsl].T
                beta_real[m_start:m_end, p_start:real_end] = beta[:rsl].T
                se_real[m_start:m_end, p_start:real_end] = var_beta[:rsl].sqrt().T
                # SnpFractVarExpl = sqrt(num^2 / (denom * rWr)) in the X-regressed GLS space, lmm_cov.py:1068
                sfve = ((num[:rsl] * num[:rsl]) / (denom[:rsl] * rWr[:rsl].unsqueeze(1))).clamp(min=0.0).sqrt()
                sfve_real[m_start:m_end, p_start:real_end] = sfve.T
    # per-SNP null h2 = sigma2_g / (sigma2_g + sigma2_e) = 1 / (1 + delta), one value per real pheno
    h2 = 1.0 / (1.0 + log_delta[:n_real].exp())
    nullh2 = h2.unsqueeze(0).expand(M, n_real).contiguous()
    return ScanResult(f=f_real, beta=beta_real, se=se_real, sfve=sfve_real,
                      nullh2=nullh2, max_F=max_F)


def loco_scan(Z: Tensor,
              X: Tensor,
              Y: Tensor,
              chrom: np.ndarray,
              *,
              n_real: int | None = None,
              on_chrom=None) -> ScanResult:
    """
    PORT IS DONE!
    Multi-pheno Leave-One-Chromosome-Out scan, ports the LocoGwas path in single_snp.py (_internal_single_snp_LocoGwas around line 1100). Y is (N, P), returns a ScanResult
    For each chromosome c the kinship is rebuilt without c, refit per-pheno log delta against K_loco, snp_wald_scan on c-only SNPs.
    Looping chromosomes here is fewer moving parts than fastlmm's LocoGwas + SnpReader subset and lets the GPU chew through the per-chrom eigh wihtout going back to pysnptools
    The grm + eigendecomposition + rotations are genotype-only, so they cost the same whether Y carries one pheno or a whole batch of them.  That's the whole point of feeding loco_scan a wide Y -- one pheno per call repays the 16 eighs every pheno, a batch of them repays it once
    Each chromosome's SNPs carry that chromosome's null h2 (fastlmm's Nullh2), since under LOCO the kinship the SNP is tested against drops its own chromosome.  n_real flows to snp_wald_scan: leading columns keep full per-SNP detail, trailing perm columns only feed the per-column genome max.  Z must be pre-standardised (use io.standardise_columns).  on_chrom, if given, is called on_chrom(chroms_done, chroms_total) after each chromosome for progress reporting
    """
    M = Z.shape[1]
    P = Y.shape[1]
    if n_real is None:
        n_real = P
    n_real = min(n_real, P)
    f = torch.zeros(M, n_real, dtype=Z.dtype, device=Z.device)
    beta = torch.zeros(M, n_real, dtype=Z.dtype, device=Z.device)
    se = torch.zeros(M, n_real, dtype=Z.dtype, device=Z.device)
    sfve = torch.zeros(M, n_real, dtype=Z.dtype, device=Z.device)
    nullh2 = torch.zeros(M, n_real, dtype=Z.dtype, device=Z.device)
    max_F = torch.full((P,), float("-inf"), dtype=Z.dtype, device=Z.device)
    chroms = sorted(np.unique(chrom).tolist())
    for k_idx, c in enumerate(chroms):
        kin_mask = chrom != c
        test_mask = chrom == c
        K = grm(Z[:, kin_mask])  # K_loco for this chromosome
        spec = rotate(K, X, Y)
        log_delta = fit_delta_grid(spec)  # (P,)
        S_rot = spec.U.T @ Z[:, test_mask]
        res = snp_wald_scan(spec, log_delta, S_rot, n_real=n_real)
        if n_real > 0:
            f[test_mask, :] = res.f
            beta[test_mask, :] = res.beta
            se[test_mask, :] = res.se
            sfve[test_mask, :] = res.sfve
            nullh2[test_mask, :] = res.nullh2
        max_F = torch.maximum(max_F, res.max_F)  # genome max = max over each chrom's max
        if on_chrom is not None:
            on_chrom(k_idx + 1, len(chroms))
    return ScanResult(f=f, beta=beta, se=se, sfve=sfve, nullh2=nullh2, max_F=max_F)


def single_k_scan(Z: Tensor,
                  X: Tensor,
                  Y: Tensor,
                  *,
                  n_real: int | None = None) -> ScanResult:
    """
    Non-LOCO whole-genome scan: one K from all SNPs, one rotation + delta-fit, one snp_wald_scan over all M
    Less defensible than LOCO (proximal contamination -- the tested SNP sits in the K it's compared against) but cheaper, occasionally useful for sanity checks or tiny chrom counts where LOCO degenerates
    Same Z assumptions as loco_scan (pre-standardised).  Returns a ScanResult, n_real defaults to all phenos
    """
    K = grm(Z)
    spec = rotate(K, X, Y)
    log_delta = fit_delta_grid(spec)  # (P,)
    S_rot = spec.U.T @ Z
    return snp_wald_scan(spec, log_delta, S_rot, n_real=n_real)


# ---------------------------------------------------------------------------
# fastlmm-compat scan path
#
# the plain rotate / snp_wald_scan above regress the covariates out in the
# V-metric (GLS) and pay a (C, C) cholesky at every grid point.  fastlmm does
# something different and the compat path copies it: project X out with a
# plain OLS hat first, eigendecompose the projected K, drop the first D = ncols(X)
# eigenvectors.  the scan then runs in a covariate-free reduced basis with no
# inner cholesky at all, wich is both ~14x faster and the bit-for-bit match
# fastlmm parity actually needs.  the plain path stays around as the
# statistically-correct rank-reduced alternative
# ---------------------------------------------------------------------------


@dataclass
class CompatSpectrum:
    """
    Eigendecomposition for the fastlmm-compat path
    fastlmm's projection formulation (lmm_cov.py lines 117-131): form K_ = (I - PX)(K + I)(I - PX), eigh that, drop the first D = ncols(X) eigenvectors and subtract the 1 back off the kept eigenvalues
    The drop-D cut is exact when rank(X) == D but throws away D - rank(X) real dimensions when X is rank-deficient.  Reproducing that quirk on purpose, matching it is the whole point of the parity port
    s (N - D,) kept eigenvalues, ascending
    U_eff (N, N - D) the kept eigenvectors
    X (N, D) the full input design
    Xpinv (D, N) pseudo-inverse of X, the OLS hat
    UY (N - D, P) covariate-regressed and projected phenos
    """
    s: Tensor
    U_eff: Tensor
    X: Tensor
    Xpinv: Tensor
    UY: Tensor


def fastlmm_compat_rotate(K: Tensor,
                          X: Tensor,
                          Y: Tensor) -> CompatSpectrum:
    """
    PORT IS DONE!
    fastlmm's projection-based rotation (lmm_cov.py lines 117-131)
    Projects K onto the orthogonal complement of X with an OLS hat, eigendecomposes, drops the first D = ncols(X) eigenvectors.  Y gets the same OLS regress-out and lands projected onto the kept basis
    Regress-out is OLS here, not GLS -- that is one of the fastlmm quirks the port reproduces deliberately
    """
    N = K.shape[0]
    D = X.shape[1]
    device, dtype = K.device, K.dtype
    Xpinv = torch.linalg.pinv(X)
    PX = X @ Xpinv
    Iperp = torch.eye(N, dtype=dtype, device=device) - PX
    del PX  # done with it, dropping one NxN slab before Kp goes up
    # building Kp in place trough two matmuls, peak live tensors stay at Iperp + Kp
    Kp = K + torch.eye(N, dtype=dtype, device=device)
    tmp = Iperp @ Kp
    del Kp
    Kp = tmp @ Iperp
    del tmp
    if device.type == "cuda":
        torch.cuda.empty_cache()
    s_full, U_full = _eigh_with_cpu_fallback(Kp)
    del Kp, Iperp
    if device.type == "cuda":
        torch.cuda.empty_cache()
    s = torch.clamp(s_full[D:N] - 1.0, min=0.0)  # subtract the +I shift back off, floor float noise at 0
    U_eff = U_full[:, D:N].contiguous()
    del s_full, U_full
    Y_r = Y - X @ (Xpinv @ Y)  # OLS regress-out of the covariates
    UY = U_eff.T @ Y_r
    return CompatSpectrum(s=s, U_eff=U_eff, X=X, Xpinv=Xpinv, UY=UY)


def _profile_loss_compat(log_delta: Tensor,
                         s: Tensor,
                         UY: Tensor) -> Tensor:
    """
    PORT IS DONE!
    Vectorised -2*loglik on a (G grid x P pheno) tensor for the compat path (lmm_cov.py LMM._nLLeval, ML branch)
    X is already regressed out in the projected basis so it never appears, the loss is just the variance term plus log det V.  ML only, fastlmm's findH2 has no REML branch.  Returns shape (G, P)
    """
    Neff = s.shape[0]
    delta = log_delta.exp()  # (G,)
    Sd = s.unsqueeze(0) + delta.unsqueeze(1)  # Sd[g, n] = s[n] + delta[g]
    w = 1.0 / Sd
    yWy = w @ (UY * UY)  # (G, P), a plain GEMM dodges the (G, Neff, P) einsum intermediate
    rWr = torch.clamp(yWy, min=_pos_floor(s.dtype))
    sum_log_Sd = Sd.log().sum(dim=1)  # (G,)
    log2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=s.dtype, device=s.device))
    sigma2 = rWr / Neff
    return Neff * (log2pi + sigma2.log() + 1.0) + sum_log_Sd.unsqueeze(1)


def _profile_loss_compat_per_pheno(log_delta: Tensor,
                                   s: Tensor,
                                   UY: Tensor) -> Tensor:
    """
    Per-pheno _profile_loss_compat for the golden-section refinment: one log delta per pheno, returns (P,)
    """
    Neff = s.shape[0]
    delta = log_delta.exp()  # (P,)
    Sd = s.unsqueeze(0) + delta.unsqueeze(1)  # (P, Neff)
    w = 1.0 / Sd
    yWy = (w * UY.T * UY.T).sum(dim=1)  # (P,)
    rWr = torch.clamp(yWy, min=_pos_floor(s.dtype))
    sum_log_Sd = Sd.log().sum(dim=1)
    log2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=s.dtype, device=s.device))
    sigma2 = rWr / Neff
    return Neff * (log2pi + sigma2.log() + 1.0) + sum_log_Sd


def fit_delta_grid_compat(spec: CompatSpectrum,
                          *,
                          n_grid: int = 64,
                          log_delta_min: float = -10.0,
                          log_delta_max: float = 10.0,
                          refine: bool = True) -> Tensor:
    """
    Multi-pheno log-delta fit for the compat path: 64-point grid then golden-section refinment, returns (P,)
    Same search as fit_delta_grid but on the compat profile loss, wich carries no covariate term since X is already projected out.  ML only, no reml flag because fastlmm's findH2 doesnt have one
    """
    s, UY = spec.s, spec.UY
    grid = torch.linspace(log_delta_min, log_delta_max, n_grid, dtype=s.dtype, device=s.device)
    loss = _profile_loss_compat(grid, s, UY)  # (G, P)
    idx = loss.argmin(dim=0)  # (P,)
    if not refine:
        return grid[idx]

    # vectorised golden-section refinment on [grid[idx-1], grid[idx+1]] per pheno, 50 iters lands well past what the scan needs
    lo_idx = torch.clamp(idx - 1, 0, n_grid - 1)
    hi_idx = torch.clamp(idx + 1, 0, n_grid - 1)
    a = grid[lo_idx].clone()
    b = grid[hi_idx].clone()
    inv_phi = 2.0 / (1.0 + 5.0 ** 0.5)
    inv_phi2 = inv_phi * inv_phi
    c = a + inv_phi2 * (b - a)
    d = a + inv_phi  * (b - a)
    fc = _profile_loss_compat_per_pheno(c, s, UY)
    fd = _profile_loss_compat_per_pheno(d, s, UY)
    for _ in range(50):
        cond = fc < fd
        b = torch.where(cond, d, b)
        a = torch.where(cond, a, c)
        c = a + inv_phi2 * (b - a)
        d = a + inv_phi  * (b - a)
        fc = _profile_loss_compat_per_pheno(c, s, UY)
        fd = _profile_loss_compat_per_pheno(d, s, UY)
    return (0.5 * (a + b)).clamp(log_delta_min, log_delta_max)


def snp_wald_scan_compat(spec: CompatSpectrum,
                         log_delta: Tensor,
                         S: Tensor,
                         *,
                         snp_chunk: int = 4096,
                         pheno_chunk: int = 256,
                         n_real: int | None = None) -> ScanResult:
    """
    PORT IS DONE!
    Multi-pheno per-(SNP, pheno) Wald F-stat in the compat (projected) basis (lmm_cov.py around 1050, the per-SNP loop in single_snp.py _internal_single_snp near line 1300)
    Test SNPs get the same OLS regress-out against X and projection onto U_eff the phenos already went trough, so the whole test runs in the covariate-free reduced space -- no per-grid-point (C, C) cholesky, wich is what makes the compat path the fast one
    Per-(SNP, pheno) dof is Neff - 1 = N - D - 1, matching fastlmm's _nLLcore rebinding N := N - linreg.D and the F-test in single_snp.py:1415
    n_real splits the pheno axis the usual way: the first n_real columns keep full per-SNP detail (F, beta, SE, SnpFractVarExpl), the rest are permutation columns and only feed the per-column running max F.  Returns a ScanResult
    """
    s, UY, X, Xpinv, U_eff = spec.s, spec.UY, spec.X, spec.Xpinv, spec.U_eff
    Neff = s.shape[0]
    P = UY.shape[1]
    M = S.shape[1]
    if n_real is None:
        n_real = P
    n_real = min(n_real, P)
    var_df = Neff - 1  # = N - D - 1
    floor = _pos_floor(s.dtype)

    # OLS-regress X out of the test SNPs, then project onto the kept basis -- the same two steps the phenos took in fastlmm_compat_rotate
    S_r = S - X @ (Xpinv @ S)  # (N, M)
    US = U_eff.T @ S_r  # (Neff, M)

    f_real = torch.zeros(M, n_real, dtype=s.dtype, device=s.device)
    beta_real = torch.zeros(M, n_real, dtype=s.dtype, device=s.device)
    se_real = torch.zeros(M, n_real, dtype=s.dtype, device=s.device)
    sfve_real = torch.zeros(M, n_real, dtype=s.dtype, device=s.device)
    max_F = torch.full((P,), float("-inf"), dtype=s.dtype, device=s.device)
    for p_start in range(0, P, pheno_chunk):
        p_end = min(p_start + pheno_chunk, P)
        UY_c = UY[:, p_start:p_end]  # (Neff, Pc)
        delta_c = log_delta[p_start:p_end].exp()  # (Pc,)
        w = 1.0 / (s.unsqueeze(0) + delta_c.unsqueeze(1))  # (Pc, Neff) per-pheno V⁻¹ diagonal
        # the null residual is just UY_c -- X is already projected away, nothing left to regress
        rWr = (w * UY_c.T * UY_c.T).sum(dim=1)  # (Pc,) null SS in V-metric
        rWr = torch.clamp(rWr, min=floor)
        real_end = min(p_end, n_real)  # last real-pheno column landing in this chunk

        for m_start in range(0, M, snp_chunk):
            m_end = min(m_start + snp_chunk, M)
            US_c = US[:, m_start:m_end]  # (Neff, Mc)
            sWy = (w * UY_c.T) @ US_c  # (Pc, Mc)
            sWs = w @ (US_c * US_c)  # (Pc, Mc)
            denom = torch.clamp(sWs, min=floor)
            beta = sWy / denom
            rss_full = torch.clamp(rWr.unsqueeze(1) - sWy * sWy / denom, min=floor)
            var_beta = rss_full / var_df / denom  # df = N - D - 1
            f_chunk = (beta * beta) / var_beta  # (Pc, Mc) Wald F per (pheno, SNP)
            # per-column running max over the genome -- that's the whole perm threshold
            max_F[p_start:p_end] = torch.maximum(max_F[p_start:p_end], f_chunk.max(dim=1).values)
            # full per-SNP detail kept only for the real phenos, perm columns stop at the max above
            if real_end > p_start:
                rsl = real_end - p_start
                f_real[m_start:m_end, p_start:real_end] = f_chunk[:rsl].T
                beta_real[m_start:m_end, p_start:real_end] = beta[:rsl].T
                se_real[m_start:m_end, p_start:real_end] = var_beta[:rsl].sqrt().T
                # SnpFractVarExpl = sqrt(sWy² / (denom * YKY)), lmm_cov.py:1068. compat X is already regressed out so rWr is YKY directly
                sfve = ((sWy[:rsl] * sWy[:rsl]) / (denom[:rsl] * rWr[:rsl].unsqueeze(1))).clamp(min=0.0).sqrt()
                sfve_real[m_start:m_end, p_start:real_end] = sfve.T
    # per-SNP null h2 = sigma2_g / (sigma2_g + sigma2_e) = 1 / (1 + delta), one value per real pheno
    h2 = 1.0 / (1.0 + log_delta[:n_real].exp())
    nullh2 = h2.unsqueeze(0).expand(M, n_real).contiguous()
    return ScanResult(f=f_real, beta=beta_real, se=se_real, sfve=sfve_real,
                      nullh2=nullh2, max_F=max_F)


def loco_scan_compat(Z: Tensor,
                     X: Tensor,
                     Y: Tensor,
                     chrom: np.ndarray,
                     *,
                     n_real: int | None = None,
                     on_chrom=None) -> ScanResult:
    """
    PORT IS DONE!
    Multi-pheno Leave-One-Chromosome-Out scan in fastlmm-compat mode, ports the LocoGwas path in single_snp.py (_internal_single_snp_LocoGwas around line 1100). Y is (N, P), returns a ScanResult
    Same shape contract as loco_scan, but every chromosome rotates trough fastlmm_compat_rotate so the covariates get the OLS regress-out and the D-eigenvector drop that match fastlmm bit-for-bit
    For each chromosome c the kinship is rebuilt without c, refit per-pheno log delta against K_loco, snp_wald_scan_compat on c-only SNPs.  Each chromosome's SNPs carry that chromosome's null h2.  n_real flows down: leading columns keep full per-SNP detail, trailing perm columns only feed the per-column genome max.  Z must be pre-standardised (use io.standardise_columns).  on_chrom, if given, is called on_chrom(chroms_done, chroms_total) after each chromosome
    """
    M = Z.shape[1]
    P = Y.shape[1]
    if n_real is None:
        n_real = P
    n_real = min(n_real, P)
    f = torch.zeros(M, n_real, dtype=Z.dtype, device=Z.device)
    beta = torch.zeros(M, n_real, dtype=Z.dtype, device=Z.device)
    se = torch.zeros(M, n_real, dtype=Z.dtype, device=Z.device)
    sfve = torch.zeros(M, n_real, dtype=Z.dtype, device=Z.device)
    nullh2 = torch.zeros(M, n_real, dtype=Z.dtype, device=Z.device)
    max_F = torch.full((P,), float("-inf"), dtype=Z.dtype, device=Z.device)
    chroms = sorted(np.unique(chrom).tolist())
    for k_idx, c in enumerate(chroms):
        kin_mask = chrom != c
        test_mask = chrom == c
        K = grm(Z[:, kin_mask])  # K_loco for this chromosome
        spec = fastlmm_compat_rotate(K, X, Y)
        log_delta = fit_delta_grid_compat(spec)  # (P,)
        res = snp_wald_scan_compat(spec, log_delta, Z[:, test_mask], n_real=n_real)
        if n_real > 0:
            f[test_mask, :] = res.f
            beta[test_mask, :] = res.beta
            se[test_mask, :] = res.se
            sfve[test_mask, :] = res.sfve
            nullh2[test_mask, :] = res.nullh2
        max_F = torch.maximum(max_F, res.max_F)  # genome max = max over each chrom's max
        if on_chrom is not None:
            on_chrom(k_idx + 1, len(chroms))
    return ScanResult(f=f, beta=beta, se=se, sfve=sfve, nullh2=nullh2, max_F=max_F)


def single_k_scan_compat(Z: Tensor,
                         X: Tensor,
                         Y: Tensor,
                         *,
                         n_real: int | None = None) -> ScanResult:
    """
    Non-LOCO whole-genome scan in fastlmm-compat mode: one K from all SNPs, one compat rotation + delta-fit, one snp_wald_scan_compat over all M
    Same proximal-contamination caveat as single_k_scan, just the fastlmm-compat rotation insted of the plain one.  Z must be pre-standardised
    """
    K = grm(Z)
    spec = fastlmm_compat_rotate(K, X, Y)
    log_delta = fit_delta_grid_compat(spec)  # (P,)
    return snp_wald_scan_compat(spec, log_delta, Z, n_real=n_real)
