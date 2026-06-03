"""
Low-rank spectral LMM, the path that scales past the N-by-N eigendecomposition
When the kinship is built from k pruned markers and k < N, K = G Gᵀ has rank k, so decomposing the N-by-k factor G beats eighing the N-by-N kernel -- O(N k²) and O(N k) insted of O(N³) and O(N²)
This is fastlmm's own k<N branch (lmm_cov.py setSU_fromG), not a new model.  The full-rank compat path in core.py forms K and projects it, this one never forms K at all
The N - k null directions of K never get an eigenvector, they only ever see delta, so the likelihood carries them analyticaly through the off-rank residual UUA = A - U Uᵀ A and an (Neff - k) log delta term in the log det
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from fasterlmm.core import ScanResult, _pos_floor, _eigh_with_cpu_fallback


@dataclass
class LowRankBasis:
    """
    Genotype-only part of the low-rank decomposition, pheno-independent so it's computed once per chromosone and resued across every pheno chunk
    s (k,) kept eigenvalues, the squared singular values of the X-regressed G, scaled by 1/M_kin
    U_eff (N, k) the kept left singular vectors, an orthonormal basis for the kinship column space
    X (N, D) the full input design, Xpinv (D, N) its psuedo-inverse, the OLS hat
    Neff = N - D, the reduced sample size driving sigma2 and the F-test dof, NOT the rank k
    """
    s: Tensor
    U_eff: Tensor
    X: Tensor
    Xpinv: Tensor
    Neff: int


@dataclass
class LowRankSpectrum:
    """
    A LowRankBasis with one pheno chunk rotated onto it
    Mirrors CompatSpectrum but for the k < N case, so it also carries the off-rank residal of Y and the true reduced sample size Neff = N - D that the rank k no longer equals
    UY (k, P) on-rank rotated phenos Uᵀ (I - PX) Y
    UUY (N, P) off-rank residual (I - PX) Y - U UY, the part of Y ouside the kinship span
    """
    s: Tensor
    U_eff: Tensor
    X: Tensor
    Xpinv: Tensor
    Neff: int
    UY: Tensor
    UUY: Tensor


def lowrank_basis(G: Tensor,
                  X: Tensor,
                  *,
                  sv_floor: float = 1e-10) -> LowRankBasis:
    """
    PORT IS DONE!
    fastlmm's low-rank spectral decomposition (lmm_cov.py setSU_fromG lines 138-162)
    Regress X out of the kinship factor G with an OLS hat, decompose the N-by-k result, keep the nonzero singular values
    fastlmm SVDs the N-by-k PxG directly.  Going trough eigh of the tiny k-by-k Gᵀ G insted (lmm.py:197 takes the same route) -- it gives the same U and s up to float noise but the workspace is k-by-k not N-by-k, wich is what keeps the basis affordable at N = 100k
    The k < N branch is the whole point and the case extreme always lands in (capped pruned GRM).  k >= N flips to eighing the N-by-N kernel directly, since then k-by-k is the bigger of the two -- that's fastlmm's own setSU_fromG switch (lmm_cov.py:141), kept here as a robustness fallback for the corner where the GRM rank reaches N
    G must be column-standardised (use io.standardise_columns).  G is the pruned kinship markers, separate from the test SNPs -- that separation is what caps the rank
    """
    N, k_in = G.shape
    D = X.shape[1]
    M_kin = k_in  # eigenvalue scale, keeps delta on the same footing as core.grm's K = ZZᵀ/M
    Xpinv = torch.linalg.pinv(X)
    G_ = G - X @ (Xpinv @ G)  # OLS regress the covariates out of the kinship factor
    if k_in < N:
        GtG = G_.T @ G_  # (k, k), the small Gram we actualy decompose
        evals, V = torch.linalg.eigh(GtG)  # ascending, evals are the squared singular values of G_
        sv = torch.clamp(evals, min=0.0).sqrt()  # singular values of G_
        keep = sv > sv_floor  # drop the null directions the X regress-out and rank deficiency leave behind
        sv = sv[keep]
        V = V[:, keep]
        U_eff = G_ @ (V / sv)  # (N, k) orthonormal basis for the kinship span, U = G_ V / singular
        s = (sv * sv) / M_kin  # eigenvalues of K = G_ G_ᵀ / M_kin
    else:
        # k >= N: the k-by-k Gram is bigger than the kernel, so eigh K = G_ G_ᵀ / M_kin straight
        K = (G_ @ G_.T) / M_kin  # (N, N)
        evals, U = _eigh_with_cpu_fallback(K)  # ascending
        keep = torch.clamp(evals, min=0.0) > (sv_floor * sv_floor) / M_kin
        s = torch.clamp(evals[keep], min=0.0)
        U_eff = U[:, keep].contiguous()
    return LowRankBasis(s=s, U_eff=U_eff, X=X, Xpinv=Xpinv, Neff=N - D)


def rotate_phenos(basis: LowRankBasis,
                  Y: Tensor) -> tuple[Tensor, Tensor]:
    """
    PORT IS DONE!
    Rotate a pheno chunk onto an exsiting basis (lmm_cov.py rotate lines 182-211)
    OLS regress the covariates out, split into the on-rank rotation UY and the off-rank residual UUY.  Cheap relative to the basis eigh, so it's the part that reruns per pheno chunk
    Returns (UY (k, P), UUY (N, P)).  UUY is the exact residual fastlmm keeps to dodge cancelation, fine to materialise here since a pheno chunk is sized to fit
    """
    Y_r = Y - basis.X @ (basis.Xpinv @ Y)  # OLS regress-out of the covariates, same step the SNPs take later
    UY = basis.U_eff.T @ Y_r  # (k, P) on-rank rotation
    UUY = Y_r - basis.U_eff @ UY  # (N, P) off-rank residual
    return UY, UUY


def lowrank_rotate(G: Tensor,
                   X: Tensor,
                   Y: Tensor,
                   *,
                   sv_floor: float = 1e-10) -> LowRankSpectrum:
    """
    Basis decompositon + one pheno rotation in one call, the resident-matrix convenience that lowrank_basis + rotate_phenos compose
    """
    basis = lowrank_basis(G, X, sv_floor=sv_floor)
    UY, UUY = rotate_phenos(basis, Y)
    return LowRankSpectrum(s=basis.s, U_eff=basis.U_eff, X=basis.X, Xpinv=basis.Xpinv,
                           Neff=basis.Neff, UY=UY, UUY=UUY)


def _profile_loss_lowrank(log_delta: Tensor,
                          s: Tensor,
                          UY: Tensor,
                          yy_off: Tensor,
                          Neff: int) -> Tensor:
    """
    PORT IS DONE!
    Vectorised -2*loglik on a (G grid x P pheno) tensor for the low-rank path (lmm_cov.py nLLcore + computeAKA, ML branch)
    The quadratic form splits in two: the on-rank part weights UY by 1/(s + delta) over the k kept dims, the off-rank part is the residal SS yy_off shared across the Neff - k null dims, each seeing denom = delta.  log det picks up (Neff - k) log delta for those null dims
    Neff = N - D is the reduced sample size, not the rank k -- that is the whole diference from the full-rank compat loss.  Returns shape (G, P)
    """
    k = s.shape[0]
    delta = log_delta.exp()  # (G,)
    Sd = s.unsqueeze(0) + delta.unsqueeze(1)  # Sd[g, n] = s[n] + delta[g], the on-rank diagonal of V
    w = 1.0 / Sd
    yWy_on = w @ (UY * UY)  # (G, P) on-rank weighted SS
    yWy_off = yy_off.unsqueeze(0) / delta.unsqueeze(1)  # (G, P) off-rank SS over the null dims, denom = delta
    rWr = torch.clamp(yWy_on + yWy_off, min=_pos_floor(s.dtype))  # = YKY
    sum_log_Sd = Sd.log().sum(dim=1)  # (G,) on-rank log det
    logdet = sum_log_Sd + (Neff - k) * delta.log()  # off-rank null dims each contribute log delta
    log2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=s.dtype, device=s.device))
    sigma2 = rWr / Neff
    return Neff * (log2pi + sigma2.log() + 1.0) + logdet.unsqueeze(1)


def _profile_loss_lowrank_per_pheno(log_delta: Tensor,
                                    s: Tensor,
                                    UY: Tensor,
                                    yy_off: Tensor,
                                    Neff: int) -> Tensor:
    """
    Per-pheno _profile_loss_lowrank for the golden-section refinment: one log delta per pheno, returns (P,)
    """
    k = s.shape[0]
    delta = log_delta.exp()  # (P,)
    Sd = s.unsqueeze(0) + delta.unsqueeze(1)  # (P, k)
    w = 1.0 / Sd
    yWy_on = (w * UY.T * UY.T).sum(dim=1)  # (P,)
    yWy_off = yy_off / delta  # (P,)
    rWr = torch.clamp(yWy_on + yWy_off, min=_pos_floor(s.dtype))
    sum_log_Sd = Sd.log().sum(dim=1)  # (P,)
    logdet = sum_log_Sd + (Neff - k) * delta.log()
    log2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=s.dtype, device=s.device))
    sigma2 = rWr / Neff
    return Neff * (log2pi + sigma2.log() + 1.0) + logdet


def fit_delta_grid_lowrank(spec: LowRankSpectrum,
                           *,
                           n_grid: int = 64,
                           log_delta_min: float = -10.0,
                           log_delta_max: float = 10.0,
                           refine: bool = True) -> Tensor:
    """
    Multi-pheno log-delta fit for the low-rank path: 64-point grid then golden-section refinment, returns (P,)
    Same search as fit_delta_grid_compat on the low-rank profile loss.  ML only, fastlmm's findH2 has no REML branch
    """
    s, UY = spec.s, spec.UY
    yy_off = (spec.UUY * spec.UUY).sum(dim=0)  # (P,) off-rank residual SS per pheno
    Neff = spec.Neff
    grid = torch.linspace(log_delta_min, log_delta_max, n_grid, dtype=s.dtype, device=s.device)
    loss = _profile_loss_lowrank(grid, s, UY, yy_off, Neff)  # (G, P)
    idx = loss.argmin(dim=0)  # (P,)
    if not refine:
        return grid[idx]

    # vectorised golden-section refinment on [grid[idx-1], grid[idx+1]] per pheno, 50 iters lands past what the scan needs
    lo_idx = torch.clamp(idx - 1, 0, n_grid - 1)
    hi_idx = torch.clamp(idx + 1, 0, n_grid - 1)
    a = grid[lo_idx].clone()
    b = grid[hi_idx].clone()
    inv_phi = 2.0 / (1.0 + 5.0 ** 0.5)
    inv_phi2 = inv_phi * inv_phi
    c = a + inv_phi2 * (b - a)
    d = a + inv_phi  * (b - a)
    fc = _profile_loss_lowrank_per_pheno(c, s, UY, yy_off, Neff)
    fd = _profile_loss_lowrank_per_pheno(d, s, UY, yy_off, Neff)
    for _ in range(50):
        cond = fc < fd
        b = torch.where(cond, d, b)
        a = torch.where(cond, a, c)
        c = a + inv_phi2 * (b - a)
        d = a + inv_phi  * (b - a)
        fc = _profile_loss_lowrank_per_pheno(c, s, UY, yy_off, Neff)
        fd = _profile_loss_lowrank_per_pheno(d, s, UY, yy_off, Neff)
    return (0.5 * (a + b)).clamp(log_delta_min, log_delta_max)


def _auto_tile_lowrank(N: int, k: int, P: int, M: int,
                       device: torch.device,
                       elem_size: int) -> tuple[int, int]:
    """
    Pick pheno / snp tile sizes for the low-rank scan from the GPU memory free right now
    The heavy live tensors per (pheno tile Pc, snp tile Mc) are the two N-by-Mc slabs S_r and UUS, the N-by-Pc off-rank UUY, and the dozen-ish (Pc, Mc) stat temporaries.  budgeting half the free vram for that leaves slack for the cholesky-free inner and allocator fragmentaion
    cpu / mps have no mem_get_info so they fall back to the fixed 256 / 4096 the path shiped with
    """
    if device.type != "cuda":
        return min(256, P), min(4096, M)
    free_bytes = torch.cuda.mem_get_info(device)[0]
    budget = int(free_bytes * 0.5)
    T = 12  # count of (Pc, Mc) intermediates the inner loop holds at once
    snp_chunk = min(M, 16384)
    while snp_chunk >= 512:
        fixed = elem_size * (2 * N + k) * snp_chunk  # S_r + UUS + US_c, pheno-independent
        per_pc = elem_size * (N + k + T * snp_chunk)  # UUY slice + w + the (Pc, Mc) temporaries
        pheno_chunk = (budget - fixed) // max(per_pc, 1)
        if pheno_chunk >= 32:
            return int(min(pheno_chunk, P)), int(snp_chunk)
        snp_chunk //= 2  # snp tile too fat for even 32 phenos, halve it and retry
    return min(32, P), min(512, M)


def snp_wald_scan_lowrank(spec: LowRankSpectrum,
                          log_delta: Tensor,
                          S: Tensor,
                          *,
                          snp_chunk: int | None = None,
                          pheno_chunk: int | None = None,
                          n_real: int | None = None) -> ScanResult:
    """
    PORT IS DONE!
    Multi-pheno per-(SNP, pheno) Wald F-stat in the low-rank basis (lmm_cov.py nLLcore + computeAKA / computeAKB, the per-SNP loop in single_snp.py around line 1300)
    Test SNPs get the same OLS regress-out and split into on-rank US and off-rank UUS the phenos took in lowrank_rotate.  Every quadratic form carrys both parts: the on-rank weighted by 1/(s + delta), the off-rank by 1/delta
    Per-(SNP, pheno) dof is Neff - 1 = N - D - 1, the F-test in single_snp.py:1415.  sigma2 divides by Neff, not the rank k
    n_real splits the pheno axis the usual way: leading columns keep full per-SNP detail, the rest are perm colums feeding only the per-column running max F.  Returns a ScanResult
    """
    s, UY, UUY = spec.s, spec.UY, spec.UUY
    X, Xpinv, U_eff = spec.X, spec.Xpinv, spec.U_eff
    Neff = spec.Neff
    P = UY.shape[1]
    M = S.shape[1]
    N, k = U_eff.shape
    if pheno_chunk is None or snp_chunk is None:
        auto_pc, auto_mc = _auto_tile_lowrank(N, k, P, M, s.device, s.element_size())
        pheno_chunk = auto_pc if pheno_chunk is None else pheno_chunk
        snp_chunk = auto_mc if snp_chunk is None else snp_chunk
    if n_real is None:
        n_real = P
    n_real = min(n_real, P)
    var_df = Neff - 1  # = N - D - 1
    floor = _pos_floor(s.dtype)

    f_real = torch.zeros(M, n_real, dtype=s.dtype, device=s.device)
    beta_real = torch.zeros(M, n_real, dtype=s.dtype, device=s.device)
    se_real = torch.zeros(M, n_real, dtype=s.dtype, device=s.device)
    sfve_real = torch.zeros(M, n_real, dtype=s.dtype, device=s.device)
    max_F = torch.full((P,), float("-inf"), dtype=s.dtype, device=s.device)
    for p_start in range(0, P, pheno_chunk):
        p_end = min(p_start + pheno_chunk, P)
        UY_c = UY[:, p_start:p_end]  # (k, Pc)
        UUY_c = UUY[:, p_start:p_end]  # (N, Pc)
        delta_c = log_delta[p_start:p_end].exp()  # (Pc,)
        w = 1.0 / (s.unsqueeze(0) + delta_c.unsqueeze(1))  # (Pc, k) on-rank V⁻¹ diagonal
        yy_off_c = (UUY_c * UUY_c).sum(dim=0)  # (Pc,) off-rank residual SS
        rWr = (w * UY_c.T * UY_c.T).sum(dim=1) + yy_off_c / delta_c  # (Pc,) YKY
        rWr = torch.clamp(rWr, min=floor)
        real_end = min(p_end, n_real)  # last real-pheno column landing in this chunk

        for m_start in range(0, M, snp_chunk):
            m_end = min(m_start + snp_chunk, M)
            S_b = S[:, m_start:m_end]  # (N, Mc)
            S_r = S_b - X @ (Xpinv @ S_b)  # OLS regress the covariates out of the test SNPs
            US_c = U_eff.T @ S_r  # (k, Mc) on-rank rotation
            UUS = S_r - U_eff @ US_c  # (N, Mc) off-rank residual
            # snpsKsnps = on-rank sWs + off-rank ‖UUS‖² / delta, computeAKA lmm_cov.py:1119
            ss_off = (UUS * UUS).sum(dim=0)  # (Mc,) off-rank SNP SS, pheno-independent
            sWs = w @ (US_c * US_c) + ss_off.unsqueeze(0) / delta_c.unsqueeze(1)  # (Pc, Mc)
            denom = torch.clamp(sWs, min=floor)
            # snpsKY = on-rank + off-rank UUSᵀ UUY / delta, computeAKB lmm_cov.py:1101
            sWy = (w * UY_c.T) @ US_c + (UUS.T @ UUY_c).T / delta_c.unsqueeze(1)  # (Pc, Mc)
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
                # SnpFractVarExpl = sqrt(sWy² / (denom * YKY)), lmm_cov.py:1068
                sfve = ((sWy[:rsl] * sWy[:rsl]) / (denom[:rsl] * rWr[:rsl].unsqueeze(1))).clamp(min=0.0).sqrt()
                sfve_real[m_start:m_end, p_start:real_end] = sfve.T
    # per-SNP null h2 = sigma2_g / (sigma2_g + sigma2_e) = 1 / (1 + delta), one value per real pheno
    h2 = 1.0 / (1.0 + log_delta[:n_real].exp())
    nullh2 = h2.unsqueeze(0).expand(M, n_real).contiguous()
    return ScanResult(f=f_real, beta=beta_real, se=se_real, sfve=sfve_real,
                      nullh2=nullh2, max_F=max_F)


def loco_scan_lowrank(G: Tensor,
                      g_chrom: np.ndarray,
                      Z: Tensor,
                      z_chrom: np.ndarray,
                      X: Tensor,
                      Y: Tensor,
                      *,
                      n_real: int | None = None,
                      on_chrom=None) -> ScanResult:
    """
    Multi-pheno Leave-One-Chromosome-Out scan in the low-rank basis
    G is the pruned kinship factor with its own per-column chromosome labels g_chrom, Z is the test SNP matrix with labels z_chrom -- two seperate matrices, unlike the single-Z compat path, since the cap on the kinship rank lives in G
    For each chromosome c the kinship factor drops c's pruned markers, refit per-pheno log delta against the c-out basis, scan Z's c-only SNPs.  Each chromosone's SNPs carry that chromosome's null h2.  Both G and Z must be column-standardised
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
    chroms = sorted(np.unique(z_chrom).tolist())
    for k_idx, c in enumerate(chroms):
        kin_mask = g_chrom != c  # kinship drops this chromosome's pruned markers
        test_mask = z_chrom == c
        spec = lowrank_rotate(G[:, kin_mask], X, Y)
        log_delta = fit_delta_grid_lowrank(spec)  # (P,)
        res = snp_wald_scan_lowrank(spec, log_delta, Z[:, test_mask], n_real=n_real)
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


def single_k_scan_lowrank(G: Tensor,
                          Z: Tensor,
                          X: Tensor,
                          Y: Tensor,
                          *,
                          n_real: int | None = None) -> ScanResult:
    """
    Non-LOCO whole-genome scan in the low-rank basis: one kinship factor G, one rotation + delta-fit, one scan over all of Z
    Same proximal-contaminaton caveat as core.single_k_scan_compat, just the low-rank rotation insted of forming K.  G and Z both column-standardised
    """
    spec = lowrank_rotate(G, X, Y)
    log_delta = fit_delta_grid_lowrank(spec)  # (P,)
    return snp_wald_scan_lowrank(spec, log_delta, Z, n_real=n_real)
