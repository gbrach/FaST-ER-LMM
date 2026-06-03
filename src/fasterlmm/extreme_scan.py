"""
Streaming LOCO engine for the extreme path
Ties the lazy genotype reader (io_stream) to the low-rank math (lowrank).  the kinship factor is held resident and decomposed once per chromosone, test variants stream past it a block at a time, so the N x M genotype matrix never lands whole
This is the piece that holds the 512 GB budget at 100k: the eigh is O(N k²) on the resident N x k factor, every test block is O(N k Mc) and freed before the next, and a pheno chunk's spectrum is reused across all of a chromosome's blocks
Not a port -- fastlmm runs the per-chromosome loop trough its SnpReader + LocoGwas plumbing, this is the streamed equivalent for one resident pheno chunk
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from fasterlmm.core import ScanResult
from fasterlmm.io_stream import (
    BedHandle, chrom_test_blocks, resident_chrom_blocks,
)
from fasterlmm.lowrank import (
    lowrank_basis, rotate_phenos, LowRankSpectrum,
    fit_delta_grid_lowrank, snp_wald_scan_lowrank,
)


def _loco_scan_core(chroms: list,
                    block_source,
                    M: int,
                    G: Tensor,
                    g_chrom: np.ndarray,
                    X: Tensor,
                    Y: Tensor,
                    *,
                    n_real: int | None,
                    dtype: torch.dtype,
                    on_chrom) -> ScanResult:
    """
    The shared LOCO loop both the streamed and the resident path run, fed by a block_source(c) that yields that chromosome's (Z_block, col_idx)
    Holds the scan math in one place so streamed and resident stay bit-identical -- the ONLY thing that differs between them is where the test blocks come from (disk vs a resident slab).  per chromosome the kinship factor drops that chromosone's markers, decomposes once, refits per-pheno log delta, then evey block rides the cached spectrum.  n_real splits the pheno axis the usual way, returns a ScanResult sized to M
    """
    G = G.to(dtype)
    P = Y.shape[1]
    if n_real is None:
        n_real = P
    n_real = min(n_real, P)
    Y = Y.to(dtype)
    X = X.to(dtype)

    f = torch.zeros(M, n_real, dtype=dtype, device=G.device)
    beta = torch.zeros(M, n_real, dtype=dtype, device=G.device)
    se = torch.zeros(M, n_real, dtype=dtype, device=G.device)
    sfve = torch.zeros(M, n_real, dtype=dtype, device=G.device)
    nullh2 = torch.zeros(M, n_real, dtype=dtype, device=G.device)
    max_F = torch.full((P,), float("-inf"), dtype=dtype, device=G.device)

    for k_idx, c in enumerate(chroms):
        basis = lowrank_basis(G[:, g_chrom != c], X)  # decompose the c-out kinship, once per chromosome
        UY, UUY = rotate_phenos(basis, Y)
        spec = LowRankSpectrum(s=basis.s, U_eff=basis.U_eff, X=basis.X,
                               Xpinv=basis.Xpinv, Neff=basis.Neff, UY=UY, UUY=UUY)
        log_delta = fit_delta_grid_lowrank(spec)  # (P,)
        for S_b, cidx in block_source(c):
            res = snp_wald_scan_lowrank(spec, log_delta, S_b.to(G.device), n_real=n_real)
            ridx = torch.from_numpy(cidx).to(G.device)
            if n_real > 0:
                f[ridx] = res.f
                beta[ridx] = res.beta
                se[ridx] = res.se
                sfve[ridx] = res.sfve
                nullh2[ridx] = res.nullh2
            max_F = torch.maximum(max_F, res.max_F)  # running genome max over blocks and chroms
        if on_chrom is not None:
            on_chrom(k_idx + 1, len(chroms))
    return ScanResult(f=f, beta=beta, se=se, sfve=sfve, nullh2=nullh2, max_F=max_F)


def loco_scan_streamed(handle: BedHandle,
                       G: Tensor,
                       g_chrom: np.ndarray,
                       X: Tensor,
                       Y: Tensor,
                       *,
                       n_real: int | None = None,
                       block_size: int = 8192,
                       dtype: torch.dtype = torch.float32,
                       on_chrom=None) -> ScanResult:
    """
    Multi-pheno LOCO scan that streams the test variants off disk against a resident kinship factor
    handle is an aligned lazy bed (open_aligned_bed) holding the TEST variants, G (N, k) is the resident kinship factor with per-column chromosome labels g_chrom -- it comes either from a strided auto-prune of handle or from a separate --grm bed, the engine doesn't care which.  X / Y are resident for this pheno chunk, the wide-P loop that re-streams per pheno chunk lives one level up in the cli
    Re-reads and re-standardises the bed every time it's called, wich is the right thing when N x M won't fit -- when it does fit, loco_scan_resident pays that decode once insted.  same scan underneath, see _loco_scan_core
    """
    M = len(handle.sid)
    chroms = sorted(set(handle.chrom.tolist()))

    def block_source(c):
        return chrom_test_blocks(handle, c, block_size=block_size, dtype=dtype)

    return _loco_scan_core(chroms, block_source, M, G, g_chrom, X, Y,
                           n_real=n_real, dtype=dtype, on_chrom=on_chrom)


def loco_scan_resident(Z: Tensor,
                       z_chrom: np.ndarray,
                       G: Tensor,
                       g_chrom: np.ndarray,
                       X: Tensor,
                       Y: Tensor,
                       *,
                       n_real: int | None = None,
                       block_size: int = 8192,
                       dtype: torch.dtype = torch.float32,
                       on_chrom=None) -> ScanResult:
    """
    Multi-pheno LOCO scan over a test genotype already decoded + standardised resident (Z (N, M) on host, z_chrom its per-column chromosome labels)
    The resident twin of loco_scan_streamed: identical scan, the test blocks just get sliced out of Z insted of re-read off disk every call.  used by the wide-P loop when the ram probe says N x M fits, so the decode + standardise gets paid once up front and not on evey pheno batch.  bit-for-bit equal to the streamed path, only faster -- the standardisation is per-variant so a resident block matches its streamed twin exactly
    """
    M = Z.shape[1]
    chroms = sorted(set(z_chrom.tolist()))

    def block_source(c):
        return resident_chrom_blocks(Z, z_chrom, c, block_size=block_size)

    return _loco_scan_core(chroms, block_source, M, G, g_chrom, X, Y,
                           n_real=n_real, dtype=dtype, on_chrom=on_chrom)
