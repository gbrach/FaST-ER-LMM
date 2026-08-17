"""tier-2 pairwise epistasis scan with LDCO kinship.

per-pheno top-K marginal SNPs picked from a fasterlmm gwas results dir.
for each marginal SNP m, every test SNP j gets a 1-dof Wald F on the
elementwise product t_ij = S_i ⊙ S_j after partialling [intercept,
covariates, S_m_std]. dof = N − D − 2 (extra −1 for the appended S_m).

LDCO: K is rebuilt from Z_std with chrom(m) and chrom(j) excluded. for
yeast that's 16² = 256 (m_chrom, j_chrom) tiles, but K is rebuilt 16
times per m and the eigh at N=471 is sub-second, so the LDCO premium is
~2x over single-K. addresses the polygenic-background half of phantom
epistasis (de los Campos 2019); local-LD-tagging artifacts unfixable

math invariants (project_epistasis_doability_audit_2026-05-09.md):
  - exclude self-pair (j == m)
  - variance floor on raw t_raw before scan (mask near-zero columns)
  - rank check is implicit via snp_wald_scan_compat's denom clamping;
    high-collinearity pairs (S_j ≈ ±S_m) get inflated F-stat from a
    near-zero residual denom — caller's heap absorbs them and the user
    can post-filter on a min-MAF or min-Var(t) threshold
  - rotation doesn't commute with ⊙, so per-marginal-SNP pays one full
    eigendecomp + scan; can't cache UᵀS

reference: Lippert et al. 2018, J Comput Biol (FaST-LMM-Epi)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from fasterlmm.core import (
    Spectrum, eigendecompose, fit_delta_grid, snp_wald_scan)
from fasterlmm.io import grm, standardise_columns
from fasterlmm.progress import pbar

# repointed onto the public core + lux vendored shims:
#   snp_wald_scan_batched -> vendored (absent from the public core)
#   permute_phenotypes    -> vendored (absent from public perms)
#   write_status / stage  -> _status (public progress lacks the snapshot
#                            state-machine the watcher reads + the stage cm)
from fasterlmm_lux._status import write_status, stage as progress_stage
from fasterlmm_lux.epi._kernel import snp_wald_scan_batched
from fasterlmm_lux.epi.heap import ThresholdBuffer, TopKBuffer
from fasterlmm_lux.epi.perms import permute_phenotypes


def _push_anchor_results(heaps, marginal_phenos, m: int, j_block: np.ndarray,
                         valid_mat: np.ndarray, k_local: int,
                         beta_np: np.ndarray, se_np: np.ndarray,
                         chi2_np: np.ndarray) -> None:
    """push one anchor's per-pheno (Mj, P) results into the per-pheno top-K
    buffers. shared between the batched and per-anchor-fallback paths"""
    # Apply pair exclusions in both paths, including fallback results whose
    # zeroed test columns can otherwise enter an oversized top-K buffer.
    valid = valid_mat[:, k_local]
    chi2_np[~valid, :] = np.nan
    p_list = marginal_phenos[m]
    for p_i in p_list:
        heaps[p_i].add_many(
            chi2_np[:, p_i], int(m), j_block,
            beta_np[:, p_i], se_np[:, p_i], pvalues=None)


@dataclass
class Tier2Result:
    """one pheno's tier-2 top-K pairs (drained from the heap, sorted)"""
    pheno_name: str
    snp_i_idx: np.ndarray  # (K,)
    snp_j_idx: np.ndarray  # (K,)
    beta: np.ndarray  # (K,)
    se: np.ndarray  # (K,)
    pvalue: np.ndarray  # (K,) ascending
    threshold: float = float("nan")  # perm-derived per-pheno p threshold; NaN when n_perm == 0
    # marginal stats for the anchor SNP_i, attached at drain time when scan_tier2 is given per_pheno_marginal_stats / per_pheno_marginal_threshold. all NaN when the marginal-dir didn't carry them (slim_marginals pre-2026-05-13 was SNP+PValue only)
    marginal_pvalue_i: np.ndarray | None = None  # (K,)
    marginal_beta_i: np.ndarray | None = None  # (K,)
    marginal_threshold_pheno: float = float("nan")  # per-pheno scalar, broadcast to every row at write time


def _ldco_kinship(
    Z_std: Tensor,  # (N, M_kin)
    chrom_kin: np.ndarray,  # (M_kin,)
    drop_chroms: set,
    *,
    device: str | None = None) -> Tensor:
    """K = Z_kept Z_keptᵀ / M_kept where M_kept excludes chroms in drop_chroms.
    raises ValueError if every SNP gets dropped (no kinship signal left)"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    keep_mask = ~np.isin(chrom_kin, list(drop_chroms))
    n_kept = int(keep_mask.sum())
    if n_kept == 0:
        raise ValueError(
            f"LDCO drop {drop_chroms} leaves zero SNPs in kinship; "
            f"genotype must cover at least 3 chroms")
    keep_idx = torch.from_numpy(np.where(keep_mask)[0]).to(device=device, dtype=torch.long)
    Z_dev = Z_std.to(device=device)
    Z_kept = Z_dev.index_select(1, keep_idx)
    return grm(Z_kept)


def _build_test_snps(
    Z_std_in: Tensor,  # (N, M) standardised genotypes, NaNs already zeroed
    m_idx: int,
    j_block: np.ndarray,  # (Mj,) test SNP column indices
    *,
    var_floor: float = 1e-8) -> tuple[Tensor, np.ndarray]:
    """elementwise product t_ij = Z_std[:, j] ⊙ Z_std[:, m] for j in j_block,
    then column-standardised to mean-0 unit-variance.

    Z_std must be the unit-standardised genotypes (pysnptools.Unit), so
    NaN strains have already been zero-imputed. raw genotypes here would
    poison the variance check at any anchor with missing values

    returns (T_std, valid_mask) where valid_mask is False at j == m and
    at columns whose product variance is below var_floor (low-MAF-pair
    dead columns). callers should NaN out invalid p-values before pushing
    """
    S_m = Z_std_in[:, m_idx:m_idx + 1]
    T_raw_np = (Z_std_in[:, j_block] * S_m).cpu().numpy() if Z_std_in.is_cuda \
        else (Z_std_in[:, j_block] * S_m).numpy()
    var_raw = T_raw_np.var(axis=0)
    valid = var_raw > var_floor
    # excluding self-pair: standardised² is fully determined by the marginal so the test column collapses (intercept-redundant after centering for binary sites, dominance encoding for {0,1,2} per Vitezica 2017). either way, not the epistasis test
    self_pos = np.where(j_block == m_idx)[0]
    if self_pos.size:
        valid[self_pos[0]] = False
    T_std_np = standardise_columns(T_raw_np)
    T_std = torch.from_numpy(np.ascontiguousarray(T_std_np)).to(
        dtype=Z_std_in.dtype, device=Z_std_in.device)
    return T_std, valid


def scan_tier2(
    *,
    Z_std: Tensor,  # (N, M) standardised (pysnptools.Unit) genotypes, NaN-imputed. used for both kinship and the elementwise products
    Y: Tensor,  # (N, P) phenotypes
    X: Tensor,  # (N, D) covariates incl. intercept
    chrom: np.ndarray,  # (M,) chromosome per SNP (kinship == test universe)
    snp_id: list[str],
    pheno_names: list[str],
    per_pheno_marginals: dict[str, list[str]],  # snp IDs per pheno
    pos: np.ndarray | None = None,  # (M,) bp position per SNP, needed for exclude_window_kb
    top_k_pair: int = 1000,
    var_floor: float = 1e-8,
    pheno_chunk: int = 256,
    snp_chunk: int = 4096,
    anchor_batch: int = 8,
    device: str | None = None,
    dtype: torch.dtype = torch.float64,
    status_file: str | None = None,
    n_perm: int = 0,
    perm_seed: int = 19930909,
    perm_quantile: float = 0.05,
    exclude_window_kb: float = 0.0,
    use_fp16_inner: bool = False,
    p_threshold: float | None = None,
    symmetric_pairs: bool = False,
    per_pheno_marginal_stats: dict[str, dict[str, tuple[float, float]]] | None = None,
    per_pheno_marginal_threshold: dict[str, float] | None = None) -> list[Tier2Result]:
    """tier-2 LDCO pairwise scan. returns one Tier2Result per pheno.

    per (chrom_m, chrom_j) tile: build K_LDCO once, then for every
    marginal SNP m on chrom_m, augment X with S_m_std and scan all test
    SNPs on chrom_j. push (m, j) pairs to per-pheno heaps for any pheno
    that has m in its top-K marginals

    n_perm > 0 packs P*(1+n_perm) pheno cols into Y_aug (real + n_perm
    independent shuffles per real pheno) and tracks per-perm-col running
    min-of-best-pair-p across every (chrom_m, chrom_j, anchor) tile. at
    drain time threshold[p] = quantile(running_min[perm_cols_for_p], q).
    n_perm == 0 keeps the v1 codepath byte-identical

    p_threshold (default None) switches the per-pheno buffer from a
    bounded top-K heap to a ThresholdBuffer that keeps every pair with
    F > F_cut where F_cut = F.isf(p_threshold, 1, N - C - 2). intended
    for one-pheno all-pairs runs where the whole tail is fed to a
    downstream network builder, not a top-K shortlist. perm tracking
    (and the FWER-style `threshold` field) is unaffected — use it to
    flag genome-wide-significant rows in post-processing

    symmetric_pairs (default False) skips (chrom_m, chrom_j) tiles
    with chrom_m > chrom_j. correct only when the anchor universe
    equals the test universe (i.e. all-anchors mode); duplicate
    (i,j)+(j,i) rows are then dropped by construction. inner_done
    still advances over skipped tiles so the watcher's progress bar
    stays calibrated
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    P = len(pheno_names)
    N, M = Z_std.shape
    if Y.shape[0] != N or X.shape[0] != N:
        raise ValueError("Y and X must have N rows")
    if len(snp_id) != M or chrom.shape[0] != M:
        raise ValueError("snp_id / chrom must align with Z_std columns")
    if n_perm < 0:
        raise ValueError(f"n_perm must be >= 0, got {n_perm}")
    if not (0.0 < perm_quantile < 1.0):
        raise ValueError(f"perm_quantile must be in (0, 1), got {perm_quantile}")
    if exclude_window_kb < 0.0:
        raise ValueError(f"exclude_window_kb must be >= 0, got {exclude_window_kb}")
    if exclude_window_kb > 0.0 and pos is None:
        raise ValueError("exclude_window_kb > 0 needs pos array of bp positions per SNP")
    if pos is not None and pos.shape[0] != M:
        raise ValueError(f"pos must align with Z_std cols (got {pos.shape[0]}, expected {M})")
    exclude_window_bp = exclude_window_kb * 1000.0
    if p_threshold is not None and not (0.0 < p_threshold < 1.0):
        raise ValueError(f"p_threshold must be in (0, 1), got {p_threshold}")

    # f-test dof matches what snp_wald_scan_batched uses internally (var_df = N - C - 1 where C = UX_tile.shape[1] + 1 because tier-2 appends one anchor col). compute up front so the threshold-buffer F-cut can be derived before the per-pheno buffer alloc
    f_df2_tier2 = int(N - X.shape[1] - 2)
    if f_df2_tier2 <= 0:
        raise ValueError(
            f"degenerate dof: N={N}, X.shape[1]={X.shape[1]}; "
            f"too few strains for tier-2 covar load")
    if p_threshold is not None:
        import scipy.stats as _ss_pre
        f_cut = float(_ss_pre.f.isf(p_threshold, 1, f_df2_tier2))
        print(f"[tier2] p_threshold={p_threshold:g} -> F_cut={f_cut:.4f} "
              f"(F.isf at df=(1, {f_df2_tier2}))",
              file=sys.stderr, flush=True)
    else:
        f_cut = None

    # resolving marginal IDs to column indices, building per-marginal pheno fan-out
    sid_to_idx = {sid: i for i, sid in enumerate(snp_id)}
    name_to_p = {nm: p for p, nm in enumerate(pheno_names)}
    marginal_phenos: dict[int, list[int]] = {}
    for nm, sids in per_pheno_marginals.items():
        if nm not in name_to_p:
            continue
        p_i = name_to_p[nm]
        for sid in sids:
            if sid not in sid_to_idx:
                continue
            marginal_phenos.setdefault(sid_to_idx[sid], []).append(p_i)

    if not marginal_phenos:
        raise ValueError("no marginal SNPs map to any pheno in pheno_names; "
                         "check that --pheno overlaps --marginal-dir contents")

    union_idx = sorted(marginal_phenos.keys())
    print(f"[tier2] {len(union_idx)} unique marginal SNPs across {len(per_pheno_marginals)} phenos",
          file=sys.stderr, flush=True)

    # chroms_m grouped by marginal SNPs that live on each
    by_chrom_m: dict = {}
    for m in union_idx:
        by_chrom_m.setdefault(chrom[m], []).append(m)
    chrom_ms = sorted(by_chrom_m.keys())
    chrom_js = sorted(np.unique(chrom).tolist())

    # per-pheno buffer, lazy-init only for phenos that have any marginals. default = numpy-backed top-K argpartition (TopKBuffer); per-cell push cost drops from O(Mj) python ops to O(Mj) C-coded numpy ops. p_threshold mode swaps in ThresholdBuffer for unbounded growth past the F-cut — same add_many signature, different drain
    def _make_buffer():
        if f_cut is not None:
            return ThresholdBuffer(f_cut)
        return TopKBuffer(top_k_pair)
    heaps: dict[int, TopKBuffer | ThresholdBuffer] = {
        p_i: _make_buffer()
        for p_i in range(P)
        if any(p_i in plist for plist in marginal_phenos.values())}

    # one-shot prep: device-resident Z_std (used for both kinship and the elementwise products) and Y. Z_raw is unused on device since the products go through Z_std (NaN-imputed)
    # building Y_aug = [real, perm_block_0, perm_block_1, ...] when n_perm > 0; perm cols feed snp_wald_scan_compat's max-F tracker only (n_real keeps the full (M, P_real) detail). matches perms.permute_phenotypes layout
    if n_perm > 0:
        with progress_stage("permute_phenotypes", status_file,
                            n_perm=n_perm, n_real=P):
            Y_np = Y.cpu().numpy() if isinstance(Y, Tensor) else np.asarray(Y)
            Y_aug_np = permute_phenotypes(Y_np, n_perm=n_perm, seed=perm_seed)
        Y_aug = torch.from_numpy(Y_aug_np)
    else:
        Y_aug = Y
    P_aug = Y_aug.shape[1]
    Y_dev = Y_aug.to(device=device, dtype=dtype)
    X_dev = X.to(device=device, dtype=dtype)
    Z_std_dev = Z_std.to(device=device, dtype=dtype)

    # per-pheno-col running max of best-pair-F-stat across every tile. only used when n_perm > 0; sized to P_aug so perm cols all live alongside the real ones at offsets P, 2P, ..., n_perm*P. tracking max-F instead of min-p (monotone in F(1, df2)) so the inner loop can skip the per-cell scipy F-sf round-trip; conversion to p happens once per pheno at threshold time
    perm_running_max_F = (np.full((P_aug,), -np.inf, dtype=np.float64)
                          if n_perm > 0 else None)

    # progress accounting: total tile count = (chrom_m, chrom_j) × marginals_on_chrom_m
    total_inner = sum(len(by_chrom_m[cm]) * len(chrom_js) for cm in chrom_ms)
    write_status(status_file, {"stage": "tier2_scan", "stage_state": "starting",
                                "marginals_total": len(union_idx),
                                "chrom_m_total": len(chrom_ms),
                                "chrom_j_total": len(chrom_js),
                                "inner_total": total_inner,
                                "top_k_pair": top_k_pair,
                                "n_perm": n_perm,
                                "perm_quantile": perm_quantile})

    inner_done = 0
    n_rank_deficient_anchors = 0  # anchors where [X, S_i] is collinear, fit_delta_grid blew up; skipped not crashed
    overall_t0 = time.time()
    for cm in chrom_ms:
        m_indices = by_chrom_m[cm]
        for cj in chrom_js:
            # symmetric_pairs: skip the lower-triangle (cm > cj) since pair (i, j) with i on cm and j on cj produces the same standardised t_ij as anchor j on cj × test i on cm. inner_done still advances so the watcher's progress bar matches the unfiltered total
            if symmetric_pairs and cm > cj:
                inner_done += len(m_indices)
                continue
            tile_t0 = time.time()
            drop = {cm} if cm == cj else {cm, cj}
            with progress_stage(f"ldco_kinship_{cm}_{cj}", status_file,
                                drop_chroms=sorted(drop)):
                K_dev = _ldco_kinship(Z_std_dev, chrom, drop, device=device)

            j_block = np.where(chrom == cj)[0]
            if j_block.size == 0:
                inner_done += len(m_indices)
                continue

            # HOIST: eigh of K_dev only depends on K, not on the per-anchor X_aug.
            # rotating Y and the global covars X once per tile cuts the per-anchor
            # cost from ~700ms to ~25ms at N=4390 (eigh is the bottleneck). switch
            # from compat to standard rotate path; bit-equivalent when X_aug is
            # full rank (always true for [intercept, anchor_std]: anchor is
            # mean-zero so independent from the intercept). uses Spectrum +
            # snp_wald_scan, which use sigma2 = rWr / (N - C) and dof = N - C - 1
            # — same as compat's Neff = N - D conventions when D == rank(X)
            with progress_stage(f"rotate_tile_{cm}_{cj}", status_file):
                s_tile, U_tile = eigendecompose(K_dev)
                UY_tile = U_tile.T @ Y_dev
                UX_tile = U_tile.T @ X_dev
            del K_dev
            if device == "cuda":
                torch.cuda.empty_cache()

            # FaST-LMM-Epi shortcut: fit δ once per pheno on X_only (not [X, S_i]) per
            # tile. per-anchor δ refit was the dominant per-cell cost (164 _profile_loss
            # calls × ~1ms launch each). the appended anchor S_i is mean-zero standardised
            # so its contribution to the variance-component fit is small; fastlmm-epi
            # explicitly reuses the X-only δ across all anchors. ~50-100x per-anchor speedup
            # at the cost of a small bias on δ when S_i has a strong main effect (which it
            # does by construction since S_i is a top marginal hit, but the bias is bounded
            # by the s/(s+δ) eigenvalue weighting and below the threshold the lab cares about)
            spec_tile = Spectrum(s=s_tile, U=U_tile, X_rot=UX_tile, Y_rot=UY_tile)
            # LDCO + rank-reduced X can still hit non-PD cholesky on some tiles when the LDCO eigenvectors zero out X cols. fall back to a small diagonal jitter on the Gram via an inflated-X ridge: prepend a tiny eps * I_C to X_rot's "weight" by adding eps to diag(A) inside _profile_loss. since core.py is parity-locked, do it here by perturbing UX_tile with a tiny multiple of an orthonormal basis. simplest robust fallback: skip the whole tile on _LinAlgError, mark all its anchor × j_block work as rank-deficient
            try:
                log_delta_tile = fit_delta_grid(spec_tile)  # (P,) shared across all anchors
            except torch._C._LinAlgError:
                n_skipped = len(m_indices)
                n_rank_deficient_anchors += n_skipped
                inner_done += n_skipped
                continue

            # batch anchors in groups of `anchor_batch` so the U.T @ T_std
            # projection (the dominant per-anchor matmul after the eigh hoist)
            # amortizes the GPU launch overhead. each batch builds T_std for
            # K anchors at once via a vectorized (N, K) ⊙ (N, Mj) outer-style
            # product, standardises per (k, j) col, and rotates as one big
            # (N, K*Mj) -> (Neff, K*Mj) matmul. fit + scan are still per-anchor
            # (vectorising those across anchors needs a per-pheno X_rot in
            # _profile_loss, deferred). caller can disable batching with
            # anchor_batch=1 for memory-tight runs
            j_block_idx = torch.from_numpy(j_block).to(device=device, dtype=torch.long)
            Z_j = Z_std_dev.index_select(1, j_block_idx)  # (N, Mj), shared across anchors

            for batch_start in range(0, len(m_indices), anchor_batch):
                batch = m_indices[batch_start:batch_start + anchor_batch]
                K = len(batch)

                # build T_std for K anchors at once. T_raw[n, j, k] = Z_j[n, j] * S_m_k[n]
                m_idx_t = torch.tensor(batch, device=device, dtype=torch.long)
                S_stack = Z_std_dev.index_select(1, m_idx_t)  # (N, K)
                # outer-style: (N, Mj, 1) * (N, 1, K) -> (N, Mj, K) — fp32 at K=8 Mj=2k N=4k is ~256 MB, fits comfortably
                T_raw = Z_j.unsqueeze(2) * S_stack.unsqueeze(1)
                # column-standardise per (j, k) anchor-test-pair: subtract mean over n, divide by std-over-n. self-pair and low-var cols collapse and are masked below
                t_mean = T_raw.mean(dim=0, keepdim=True)
                t_std_dev = T_raw.std(dim=0, unbiased=False, keepdim=True)
                # var-floor mask BEFORE division so we don't divide by ~0 and inflate
                t_var = t_std_dev.squeeze(0) ** 2  # (Mj, K)
                valid_mat = (t_var > var_floor).cpu().numpy()  # (Mj, K)
                # mark self-pair invalid (j == m for any column where j_block[j] == m). also drop near-self LD-trapped pairs within exclude_window_bp on the same chromosome (Bloom 2015 §Methods); local LD already saturates the F-stat so these are mostly confirmatory of the marginal hit, not novel epistasis
                for k_local, m_glob in enumerate(batch):
                    self_pos = np.where(j_block == m_glob)[0]
                    if self_pos.size:
                        valid_mat[self_pos[0], k_local] = False
                    if symmetric_pairs and cm == cj:
                        valid_mat[j_block <= m_glob, k_local] = False
                    if exclude_window_bp > 0.0 and pos is not None:
                        if chrom[m_glob] == cj:
                            anchor_bp = pos[m_glob]
                            j_pos = pos[j_block]
                            near = np.abs(j_pos - anchor_bp) < exclude_window_bp
                            if near.any():
                                valid_mat[near, k_local] = False
                # safe-divide: invalid cols get standardised to NaN, then we zero
                # them out below (matches the per-anchor _build_test_snps logic)
                t_std_safe = torch.where(t_std_dev > 0,
                                          t_std_dev,
                                          torch.ones_like(t_std_dev))
                T_std_bk = (T_raw - t_mean) / t_std_safe  # (N, Mj, K)

                # zero invalid (k, j) cols BEFORE rotation so the perm
                # max-F tracker can't see them
                bad_mask = ~torch.from_numpy(valid_mat).to(device=device)  # (Mj, K)
                if bad_mask.any():
                    T_std_bk = T_std_bk * (~bad_mask).unsqueeze(0)

                # collapse to (N, K*Mj) for one batched matmul. layout: cols 0..Mj-1 = anchor 0, Mj..2Mj-1 = anchor 1, ...
                T_std_flat = T_std_bk.permute(0, 2, 1).reshape(N, K * j_block.size)
                T_rot_flat = U_tile.T @ T_std_flat  # (Neff, K*Mj)
                T_rot_bk = T_rot_flat.view(s_tile.shape[0], K, j_block.size)

                del T_raw, T_std_bk, T_std_flat, T_rot_flat, t_mean, t_std_dev, t_std_safe

                # K-batched scan: one snp_wald_scan_batched call processes all K
                # anchors at once. dominant op is a (K*Pc, C, N, Mc) batched matmul
                # which keeps the V100S TensorCores fed instead of streaming small
                # per-anchor kernels at 8% utilisation. fall back to per-anchor on
                # _LinAlgError so one rank-deficient anchor doesn't kill the batch
                Neff = s_tile.shape[0]
                # build per-anchor X_aug: (K, Neff, C+1) — first C cols UX_tile shared, last col U_s_m
                U_s_batched = U_tile.T @ S_stack  # (Neff, K)
                X_rot_aug_batched = torch.cat([
                    UX_tile.unsqueeze(0).expand(K, -1, -1),
                    U_s_batched.T.unsqueeze(-1)
                ], dim=2).contiguous()  # (K, Neff, C+1)
                T_rot_batched = T_rot_bk.permute(1, 0, 2).contiguous()  # (K, Neff, Mj)

                try:
                    beta_b, se_b, chi2_b, max_F_b, _ = snp_wald_scan_batched(
                        s_tile, X_rot_aug_batched, UY_tile, log_delta_tile, T_rot_batched,
                        snp_chunk=snp_chunk, pheno_chunk=pheno_chunk, n_real=P,
                        use_fp16_inner=use_fp16_inner)
                except torch._C._LinAlgError:
                    # batched cholesky fails atomically: fall back to per-anchor with the
                    # try/except that skips just the bad ones. preserves all-or-nothing
                    # batching while gracefully handling rare rank-deficient anchors
                    for k_local, m in enumerate(batch):
                        U_s_m = U_s_batched[:, k_local:k_local + 1]
                        X_rot_aug = torch.cat([UX_tile, U_s_m], dim=1)
                        spec_one = Spectrum(s=s_tile, U=U_tile, X_rot=X_rot_aug, Y_rot=UY_tile)
                        try:
                            T_rot_m = T_rot_bk[:, k_local, :].contiguous()
                            # public-core snp_wald_scan: no compute_pvalue kwarg (it
                            # never does the scipy F-sf round-trip, so this already
                            # behaves like the dev compute_pvalue=False). its ScanResult
                            # renamed the dev fields: dev .chi2 (β²/SE² = the F-stat) is
                            # public .f; dev .min_pvalue (raw max-F per pheno under
                            # compute_pvalue=False) is public .max_F. mirrors the vendored
                            # batched kernel's chi2_b / max_F_b on the success path above.
                            res_one = snp_wald_scan(
                                spec_one, log_delta_tile, T_rot_m,
                                snp_chunk=snp_chunk, pheno_chunk=pheno_chunk,
                                n_real=P)
                        except torch._C._LinAlgError:
                            n_rank_deficient_anchors += 1
                            inner_done += 1
                            continue
                        # .cpu() before .numpy(): the public snp_wald_scan returns tensors
                        # on the scan device (cuda here), unlike the vendored batched kernel
                        # which already lands on cpu. no-op on cpu, so the parity path is
                        # unchanged; without it the GPU fallback crashes on .numpy() (the
                        # CPU fixed-seed parity fixture never trips the batched-cholesky
                        # LinAlgError, so this branch was latent until real GPU data hit it).
                        _push_anchor_results(
                            heaps, marginal_phenos, m, j_block, valid_mat, k_local,
                            res_one.beta.cpu().numpy(), res_one.se.cpu().numpy(),
                            res_one.f.cpu().numpy())
                        if n_perm > 0:
                            np.fmax(perm_running_max_F, res_one.max_F.cpu().numpy(),
                                    out=perm_running_max_F)
                        inner_done += 1
                    del U_s_batched, X_rot_aug_batched, T_rot_batched
                    continue

                # batched path succeeded. drain K anchors in a python loop (cheap, no GPU)
                beta_np_b = beta_b.numpy()  # (K, Mj, P)
                se_np_b = se_b.numpy()
                chi2_np_b = chi2_b.numpy()
                max_F_np_b = max_F_b.numpy()  # (K, P_aug)
                for k_local, m in enumerate(batch):
                    valid_k = valid_mat[:, k_local]
                    chi2_k = chi2_np_b[k_local]
                    if not valid_k.all():
                        chi2_k[~valid_k, :] = np.nan
                    _push_anchor_results(
                        heaps, marginal_phenos, m, j_block, valid_mat, k_local,
                        beta_np_b[k_local], se_np_b[k_local], chi2_k)
                    if n_perm > 0:
                        np.fmax(perm_running_max_F, max_F_np_b[k_local], out=perm_running_max_F)
                    inner_done += 1

                del U_s_batched, X_rot_aug_batched, T_rot_batched, beta_b, se_b, chi2_b, max_F_b

                del T_rot_bk, S_stack
                if device == "cuda":
                    torch.cuda.empty_cache()

            del Z_j, j_block_idx

            # tile boundary: drop the rotation cache before the next eigh
            del s_tile, U_tile, UY_tile, UX_tile
            if device == "cuda":
                torch.cuda.empty_cache()
            tile_elapsed = time.time() - tile_t0
            write_status(status_file, {
                "stage": "tier2_tile", "stage_state": "done",
                "chrom_m": int(cm), "chrom_j": int(cj),
                "inner_done": inner_done, "inner_total": total_inner,
                "tile_elapsed_s": tile_elapsed})

    total_elapsed = time.time() - overall_t0
    print(f"[tier2] scan done in {total_elapsed:.1f}s "
          f"({inner_done} marginal × chrom_j tiles)",
          file=sys.stderr, flush=True)
    if n_rank_deficient_anchors > 0:
        print(f"[tier2] {n_rank_deficient_anchors} anchor × tile fits skipped "
              f"(rank-deficient X_aug; usually low-MAF SNPs collinear with covariates)",
              file=sys.stderr, flush=True)

    # f_df2_tier2 was computed up-top so the threshold-buffer F_cut could be derived before buffer alloc; reuse here for the perm-threshold conversion and the per-pheno drain
    # per-pheno perm threshold. perm cols for real pheno p sit at indices p+P, p+2P, ..., p+n_perm*P in perm_running_max_F (matches permute_phenotypes' [real | perm_block_0 | perm_block_1 | ...] layout). converting each perm's max-F to p first then quantile in p-space; F.sf is non-linear so quantile-in-F + one F.sf is not the same as F.sf-then-quantile, and the threshold is applied to per-pair p-values downstream so p-space is the correct interpolation domain
    import scipy.stats as _ss
    # diag dump (FASTERLMM_DUMP_PERM_F=<path>): writes per-pheno perm-max-F array to .npz alongside the run output, so we can audit whether thresholds are noise or numerical blow-up
    _diag_dump_path = os.environ.get("FASTERLMM_DUMP_PERM_F")
    _diag_dump: dict[str, np.ndarray] = {}
    def _threshold_for(p_i: int) -> float:
        if perm_running_max_F is None:
            return float("nan")
        perm_idx = np.arange(1, n_perm + 1) * P + p_i
        perm_max_F = perm_running_max_F[perm_idx]
        if _diag_dump_path:
            _diag_dump[pheno_names[p_i]] = perm_max_F.copy()
        finite = perm_max_F[np.isfinite(perm_max_F)]
        if finite.size == 0:
            return float("nan")
        perm_min_p = _ss.f.sf(finite, 1, f_df2_tier2)
        return float(np.quantile(perm_min_p, perm_quantile))

    # marginal-i lookup helper. returns (pval_arr, beta_arr) aligned to snp_i_idx, both NaN if the caller didn't supply per_pheno_marginal_stats or the pheno is missing from it
    def _marginal_for(nm: str, i_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        K = i_idx.shape[0]
        if K == 0 or per_pheno_marginal_stats is None:
            return (np.full(K, np.nan), np.full(K, np.nan))
        stats = per_pheno_marginal_stats.get(nm)
        if stats is None:
            return (np.full(K, np.nan), np.full(K, np.nan))
        pv = np.empty(K, dtype=np.float64)
        bt = np.empty(K, dtype=np.float64)
        for k, j in enumerate(i_idx):
            rec = stats.get(snp_id[j])
            if rec is None:
                pv[k] = np.nan; bt[k] = np.nan
            else:
                pv[k], bt[k] = rec
        return pv, bt

    def _marginal_thr(nm: str) -> float:
        if per_pheno_marginal_threshold is None:
            return float("nan")
        return float(per_pheno_marginal_threshold.get(nm, float("nan")))

    # drain buffers into Tier2Result list, in pheno_names order. p-values come from one batched F-sf per pheno on the surviving F-stats (vs the per-cell scipy round-trip the heap-pure-python path used to do). ThresholdBuffer drains via drain_arrays (numpy direct) to dodge the per-row _PairEntry alloc at the ~45M-row scale
    out: list[Tier2Result] = []
    with progress_stage("tier2_drain", status_file, n_phenos=len(heaps)):
        for p_i, nm in enumerate(pheno_names):
            thr = _threshold_for(p_i)
            m_thr = _marginal_thr(nm)
            if p_i not in heaps:
                # pheno had no marginals → empty result
                empty_i = np.empty((0,), dtype=np.int64)
                m_pv, m_bt = _marginal_for(nm, empty_i)
                out.append(Tier2Result(
                    pheno_name=nm,
                    snp_i_idx=empty_i,
                    snp_j_idx=np.empty((0,), dtype=np.int64),
                    beta=np.empty((0,)), se=np.empty((0,)),
                    pvalue=np.empty((0,)),
                    threshold=thr,
                    marginal_pvalue_i=m_pv, marginal_beta_i=m_bt,
                    marginal_threshold_pheno=m_thr))
                continue
            buf = heaps[p_i]
            if isinstance(buf, ThresholdBuffer):
                f_arr, i_arr, j_arr, b_arr, s_arr = buf.drain_arrays()
                p_arr = _ss.f.sf(f_arr, 1, f_df2_tier2) if f_arr.size else f_arr
                m_pv, m_bt = _marginal_for(nm, i_arr)
                out.append(Tier2Result(
                    pheno_name=nm,
                    snp_i_idx=i_arr, snp_j_idx=j_arr,
                    beta=b_arr, se=s_arr,
                    pvalue=p_arr,
                    threshold=thr,
                    marginal_pvalue_i=m_pv, marginal_beta_i=m_bt,
                    marginal_threshold_pheno=m_thr))
                continue
            entries = buf.drain_sorted()
            f_arr = np.array([e.f_stat for e in entries], dtype=np.float64)
            p_arr = _ss.f.sf(f_arr, 1, f_df2_tier2) if f_arr.size else f_arr
            i_arr = np.array([e.snp_i for e in entries], dtype=np.int64)
            m_pv, m_bt = _marginal_for(nm, i_arr)
            out.append(Tier2Result(
                pheno_name=nm,
                snp_i_idx=i_arr,
                snp_j_idx=np.array([e.snp_j for e in entries], dtype=np.int64),
                beta=np.array([e.beta for e in entries]),
                se=np.array([e.se for e in entries]),
                pvalue=p_arr,
                threshold=thr,
                marginal_pvalue_i=m_pv, marginal_beta_i=m_bt,
                marginal_threshold_pheno=m_thr))

    write_status(status_file, {"stage": "tier2_scan", "stage_state": "done",
                                "elapsed_s": total_elapsed,
                                "inner_done": inner_done,
                                "inner_total": total_inner})

    if _diag_dump_path and _diag_dump:
        np.savez_compressed(_diag_dump_path, **_diag_dump, f_df2_tier2=np.asarray([f_df2_tier2]))
        print(f"[tier2 diag] dumped per-pheno perm-max-F to {_diag_dump_path}", file=sys.stderr, flush=True)

    return out
