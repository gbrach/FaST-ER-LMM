"""tier-2 batched Wald scan kernel — VENDORED from the dev core.

`snp_wald_scan_batched` lives in the dev `fasterlmm.core` (SAVEfasterlmm
core.py:393) but is ABSENT from the public `fasterlmm` core, which only ships
the per-anchor `snp_wald_scan`. Because it sits on the tier-2 parity surface
(it feeds every (anchor, test-SNP) Wald F-stat in the default fast path), lux
vendors it verbatim here rather than rewrite `scan.py` onto the slower
per-anchor fallback. Kept byte-faithful to the dev implementation so the epi
parity test stays trivially green; re-sync this file if the dev kernel changes.

Pure-torch, no other dev-core dependencies. The per-anchor counterpart
(`fasterlmm.core.snp_wald_scan`) is reused straight from the public core as the
`_LinAlgError` fallback in `scan.py`.
"""

from __future__ import annotations

import torch
from torch import Tensor


def snp_wald_scan_batched(
    s: Tensor,  # (N,) eigenvalues, shared across anchors
    X_rot_batched: Tensor,  # (K, N, C) per-anchor design matrices (C = C_base + 1 for tier-2)
    Y_rot: Tensor,  # (N, P) phenos, shared across anchors
    log_delta: Tensor,  # (P,) shared per-tile (FaST-LMM-Epi shortcut)
    S_rot_batched: Tensor,  # (K, N, M) per-anchor rotated test SNPs (S_i ⊙ S_j products)
    *,
    snp_chunk: int = 4096, pheno_chunk: int = 256,
    n_real: int | None = None, use_fp16_inner: bool = False,
    denom_floor_rel: float = 0.01) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """tier-2 batched Wald scan: K anchors at once. compute_pvalue is always False (caller defers F-sf to drain). returns (beta, se, chi2, max_F_per_pheno, argmax_snp_per_pheno) all with leading K dim. one big batched matmul per (pheno_chunk, snp_chunk) instead of K serial calls — at K=8, Pc=256, Mc=4096 the batched matmul is ~8x bigger which puts the V100S near peak utilisation instead of small-launch idling.

    use_fp16_inner: cast the inner snp-chunk matmuls (B, AinvB, sWs, num) to fp16 on cuda. cuBLAS half-precision is ~1.6x faster on V100 TensorCores but is NOT bitwise reproducible across launches, so top-K membership at the perm-threshold tail can flip between runs. off by default

    denom_floor_rel: when (sWs - quad) drops below this fraction of sWs the test column is essentially collinear with the anchor (rank-deficient pair), F blows up and poisons top-K. Lippert 2018 §3.1 calls this the "near-self" pair guard. set to 0 to disable
    """
    K, N, C = X_rot_batched.shape
    P = Y_rot.shape[1]
    M = S_rot_batched.shape[2]
    if n_real is None:
        n_real = P
    n_real = min(n_real, P)
    var_df = N - C - 1
    device, dtype = s.device, s.dtype

    beta_out = torch.empty((K, M, n_real), dtype=dtype)
    se_out = torch.empty((K, M, n_real), dtype=dtype)
    chi2_out = torch.empty((K, M, n_real), dtype=dtype)
    max_f = torch.full((K, P), float("-inf"), dtype=dtype, device=device)
    argmax_snp = torch.zeros((K, P), dtype=torch.int64, device=device)

    pheno_starts = list(range(0, P, pheno_chunk))
    for p_start in pheno_starts:
        p_end = min(p_start + pheno_chunk, P)
        Pc = p_end - p_start
        ld_c = log_delta[p_start:p_end]
        Y_c = Y_rot[:, p_start:p_end]  # (N, Pc) shared

        delta_c = ld_c.exp()
        w = 1.0 / (s.unsqueeze(0) + delta_c.unsqueeze(1))  # (Pc, N) shared

        # WX_kpc: (K, Pc, N, C) — w broadcast over K, X_rot_batched broadcast over Pc. WX is small (320 MB at K=8 Pc=256 N=542 C=74) so fine to materialise
        WX = w.unsqueeze(0).unsqueeze(-1) * X_rot_batched.unsqueeze(1)  # (K, Pc, N, C)
        WX_T = WX.transpose(-2, -1).contiguous()  # (K, Pc, C, N)
        # setup matmuls stay fp32. tried fp16 for A and hat builds but the cast-large-WX_T-each-pheno-chunk overhead dwarfed the matmul savings (regressed 10 cells/s → 1.4). only the inner snp-chunk matmuls benefit from fp16 because S_c is the dominant cost dim
        A = torch.empty((K, Pc, C, C), dtype=dtype, device=device)
        for k in range(K):
            A[k] = WX_T[k] @ X_rot_batched[k]  # (Pc, C, N) @ (N, C) = (Pc, C, C)
        L = torch.linalg.cholesky(A)
        # KEY PERF FIX: pre-compute A^{-1} explicitly + fuse into hat = A^{-1} @ X^T W. cuSOLVER's batched cholesky_solve is ~9x slower than cuBLAS GEMM at our shapes
        A_inv = torch.cholesky_inverse(L.flatten(0, 1)).view(K, Pc, C, C)  # (K, Pc, C, C)
        hat = torch.empty((K, Pc, C, N), dtype=dtype, device=device)
        for k in range(K):
            hat[k] = A_inv[k] @ WX_T[k]  # (Pc, C, N) for the inner-loop GLS solve
        u = torch.einsum("kpcn,np->kpc", WX_T, Y_c)  # (K, Pc, C)
        beta_x = torch.einsum("kpcd,kpd->kpc", A_inv, u)  # A_inv @ u, faster than cholesky_solve at our shape
        # r_kpn: (K, N, Pc) per-anchor X-fit residual via X @ beta_x.T
        r = Y_c.unsqueeze(0) - (X_rot_batched @ beta_x.transpose(-2, -1))  # (K, N, Pc)
        rWr = (w.unsqueeze(0) * (r * r).transpose(1, 2)).sum(dim=2)  # (K, Pc)
        rWr = torch.clamp(rWr, min=1e-300)

        real_end = min(p_end, n_real)
        Pc_real = max(0, real_end - p_start)
        if Pc_real > 0:
            beta_chunk = torch.empty((K, Pc_real, M), dtype=dtype, device=device)
            se_chunk = torch.empty((K, Pc_real, M), dtype=dtype, device=device)
            f_chunk = torch.empty((K, Pc_real, M), dtype=dtype, device=device)

        for m_start in range(0, M, snp_chunk):
            m_end = min(m_start + snp_chunk, M)
            S_c = S_rot_batched[:, :, m_start:m_end]  # (K, N, Mc)

            # inner snp-chunk matmuls optionally in fp16 on V100 TensorCore (~1.6x speedup at our shapes), but cuBLAS fp16 is non-deterministic across launches so default is fp32 for reproducibility. caller opts in via use_fp16_inner. cholesky stays fp32 either way
            Mc = m_end - m_start
            B = torch.empty((K, Pc, C, Mc), dtype=dtype, device=device)
            AinvB = torch.empty_like(B)
            use_fp16 = (use_fp16_inner and dtype == torch.float32 and device.type == "cuda")
            if use_fp16:
                S_c_h = S_c.half()
                WX_T_h = WX_T.half()
                hat_h = hat.half()
                w_h = w.half()
                wr_h = (w.unsqueeze(0) * r.transpose(1, 2)).half()  # (K, Pc, N)
                S_c_sq_h = (S_c * S_c).half()  # (K, N, Mc)
                for k in range(K):
                    B[k] = (WX_T_h[k] @ S_c_h[k]).float()
                    AinvB[k] = (hat_h[k] @ S_c_h[k]).float()
                sWs = (w_h.unsqueeze(0).expand(K, -1, -1) @ S_c_sq_h).float()  # (K, Pc, Mc)
                num = (wr_h @ S_c_h).float()  # (K, Pc, Mc)
                del S_c_h, WX_T_h, hat_h, w_h, wr_h, S_c_sq_h
            else:
                for k in range(K):
                    B[k] = WX_T[k] @ S_c[k]
                    AinvB[k] = hat[k] @ S_c[k]
                sWs = w.unsqueeze(0).expand(K, -1, -1) @ (S_c * S_c)  # (K, Pc, Mc)
                wr = w.unsqueeze(0) * r.transpose(1, 2)  # (K, Pc, N)
                num = wr @ S_c  # (K, Pc, Mc)
            quad = (B * AinvB).sum(dim=2)  # (K, Pc, Mc)
            denom_raw = sWs - quad
            denom = torch.clamp(denom_raw, min=1e-300)

            beta = num / denom
            rss_full = torch.clamp(rWr.unsqueeze(-1) - num * num / denom, min=1e-300)
            var_beta = rss_full / var_df / denom
            se = var_beta.sqrt()
            f_stat = (beta * beta) / var_beta
            # rank-deficiency guard: when the test column collapses into the anchor's span (S_i ≈ ±S_j up to scaling) denom shrinks to ~0 and F explodes, poisoning top-K. Lippert 2018 §3.1 flags these as near-self pairs. zero out F (won't survive the heap's max-F sort) for any (k, p, m) where denom is below denom_floor_rel * sWs
            if denom_floor_rel > 0.0:
                bad = denom_raw < (denom_floor_rel * sWs)
                if bad.any():
                    f_stat = torch.where(bad, torch.zeros_like(f_stat), f_stat)

            chunk_max_f, chunk_argmax = f_stat.max(dim=2)  # (K, Pc)
            global_idx = chunk_argmax + m_start
            update = chunk_max_f > max_f[:, p_start:p_end]
            max_f[:, p_start:p_end] = torch.where(update, chunk_max_f, max_f[:, p_start:p_end])
            argmax_snp[:, p_start:p_end] = torch.where(update, global_idx, argmax_snp[:, p_start:p_end])

            if Pc_real > 0:
                beta_chunk[:, :, m_start:m_end] = beta[:, :Pc_real]
                se_chunk[:, :, m_start:m_end] = se[:, :Pc_real]
                f_chunk[:, :, m_start:m_end] = f_stat[:, :Pc_real]

        if Pc_real > 0:
            beta_out[:, :, p_start:real_end] = beta_chunk.transpose(1, 2).cpu()
            se_out[:, :, p_start:real_end] = se_chunk.transpose(1, 2).cpu()
            chi2_out[:, :, p_start:real_end] = f_chunk.transpose(1, 2).cpu()

    return beta_out, se_out, chi2_out, max_f.cpu(), argmax_snp.cpu()
