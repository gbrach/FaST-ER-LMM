"""
Streaming genotype I/O for the extreme path
read_plink in io.py slurps the whole N x M bed into ram, wich is exactly the thing that falls over at 100k strains times a million variants.  this reads the bed lazily insted: metadata is free, and variants come back a block at a time so the full N x M matrix never lands resident
the kinship factor G is the one thing kept whole, and it can be -- it's only N x k for a pruned marker set, a couple GB even at 100k.  test variants stream past it block by block
not a port of anything.  fastlmm's SnpReader does its own lazy slicing trough pysnptools, this is just the thin slice of that the extreme scan actualy needs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from fasterlmm.io import standardise_columns


@dataclass
class BedHandle:
    """
    Lazy handle on a plink bed, rows alread aligned to a target strain order
    bed is the unread pysnptools Bed, row_idx maps the aligned strain order onto bed rows so evey block read comes back in that order
    iid (N,) aligned strains, sid (M,) variant ids, chrom + pos (M,) variant positions
    """
    bed: object
    row_idx: np.ndarray
    iid: list[str]
    sid: list[str]
    chrom: np.ndarray
    pos: np.ndarray


def open_aligned_bed(prefix: str | Path,
                     keep_iid: list[str],
                     *,
                     count_A1: bool = True) -> BedHandle:
    """
    Open a bed lazyly and align its rows to the common strain set keep_iid
    Keeps the bed's own row order filtered to keep_iid, the same convetion align_inputs uses so a streamed run reproduces the resident one strain-for-strain.  values stay on disk, only iid / sid / pos metadata is touched here
    """
    from pysnptools.snpreader import Bed

    bed = Bed(str(prefix), count_A1=count_A1)
    bed_iid = [str(x[1]) for x in bed.iid]
    common = set(keep_iid)
    row_idx = np.array([i for i, s in enumerate(bed_iid) if s in common], dtype=np.int64)
    iid = [bed_iid[i] for i in row_idx]
    return BedHandle(bed=bed,
                     row_idx=row_idx,
                     iid=iid,
                     sid=[str(s) for s in bed.sid],
                     chrom=np.asarray(bed.pos[:, 0]),
                     pos=np.asarray(bed.pos[:, 2]))


def read_block(h: BedHandle,
               col_idx: np.ndarray,
               *,
               dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Read one block of variants for the aligned strains, standardised, as a torch tensor (N, len(col_idx))
    col_idx is any index array into the M variants.  standardise_columns is per-column, so a block comes out identical wether it's standardised on its own or as part of the whole matrix -- that's what makes streaming exact and not an aproximation
    Read happens in float64 to match the resident path, the cast to dtype lands after.  pass float16 here for the fp16 streamed-block setting
    """
    val = h.bed[h.row_idx, col_idx].read(dtype=np.float64).val  # (N, b), only this block leaves disk
    std = standardise_columns(val)
    return torch.from_numpy(np.ascontiguousarray(std)).to(dtype)


def select_grm_markers(h: BedHandle,
                       k: int,
                       *,
                       seed: int = 19930909) -> np.ndarray:
    """
    Pick about k kinship markers by an evenly-spaced stride across the genome, the auto-prune default
    A strided subsample lands a representative, roughly MAF-spread set without a full pass over the bed -- the variants are genome-ordered so the stride spreads across every chromosome.  for a properly LD-pruned kinship pass a pre-pruned bed trough --grm insted, this is the no-prep fallback
    """
    M = len(h.sid)
    if k >= M:
        return np.arange(M, dtype=np.int64)
    return np.unique(np.linspace(0, M - 1, k).round().astype(np.int64))


def read_grm_factor(h: BedHandle,
                    col_idx: np.ndarray,
                    *,
                    dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Read the pruned kinship markers whole into a resident (N, k) factor, standardised
    This is the one genotype slab the extreme path keeps in memory -- N x k stays small even at 100k, and the spectrum gets resued across evey pheno chunk so it's worth holding
    """
    return read_block(h, col_idx, dtype=dtype)


def stream_genotype_var(h: BedHandle,
                        *,
                        block_size: int = 8192) -> np.ndarray:
    """
    Per-variant raw genotype varaince over the aligned strains, streamed (M,)
    The EffectSize column wants var of the raw 0/1/2 dosage (nan-aware, ddof=0), wich the standardised blocks have flattened to 1.  so this reads the raw values a block at a time -- one extra streamed pass, but bounded ram and tiny next to the scan
    """
    M = len(h.sid)
    out = np.empty(M, dtype=np.float64)
    for s in range(0, M, block_size):
        cidx = np.arange(s, min(s + block_size, M))
        val = h.bed[h.row_idx, cidx].read(dtype=np.float64).val  # (N, b) raw dosages
        out[cidx] = np.nanvar(val, axis=0)
    return out


def chrom_test_blocks(h: BedHandle,
                      chrom_value,
                      *,
                      block_size: int = 8192,
                      dtype: torch.dtype = torch.float32):
    """
    Yield standardised test-variant blocks for one chromosome, each with its global colum indices
    Each yield is (Z_block (N, b), col_idx (b,)) so the caller can scater the per-block results back into genome order.  block_size caps how many variants are resident at once, the lever that holds the N x M matrix off the heap
    """
    cols = np.where(h.chrom == chrom_value)[0]
    for s in range(0, len(cols), block_size):
        cidx = cols[s:s + block_size]
        yield read_block(h, cidx, dtype=dtype), cidx


def read_all_standardised(h: BedHandle,
                          *,
                          block_size: int = 8192,
                          dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Decode + standardise the whole test genotype once into a resident (N, M) host tensor
    The streamed path re-decodes and re-standardises every variant on every pheno batch, wich is pure repeat work -- the standardisation is per-variant so it dosn't depend on the phenos or on wich chromosome the loco fold drops.  when N x M fits the ram budget it's far cheaper to pay that decode once and slice the resident tensor per batch insted of re-reading the bed a hundred times
    Filled block by block trough the same read_block the streamed path uses, so a resident column is bit-identical to the streamed one, the only thing that moves is when it gets computed.  the caller decides resident-vs-stream from a ram probe, this just builds the slab
    """
    M = len(h.sid)
    N = len(h.iid)
    Z = torch.empty((N, M), dtype=dtype)
    for s in range(0, M, block_size):
        e = min(s + block_size, M)
        Z[:, s:e] = read_block(h, np.arange(s, e), dtype=dtype)
    return Z


def resident_chrom_blocks(Z: Tensor,
                          chrom_labels: np.ndarray,
                          chrom_value,
                          *,
                          block_size: int = 8192):
    """
    Yield one chromosome's already-standardised test blocks straight from a resident (N, M) tensor
    The resident twin of chrom_test_blocks: same column pick and same (Z_block, col_idx) shape, but it slices the pre-decoded slab insted of touching disk, so the per-batch decode + standardise cost is just gone.  blocks come back contigous, the caller ships them to the gpu the same way
    """
    cols = np.where(chrom_labels == chrom_value)[0]
    for s in range(0, len(cols), block_size):
        cidx = cols[s:s + block_size]
        yield Z.index_select(1, torch.from_numpy(cidx)), cidx
