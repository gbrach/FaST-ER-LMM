"""
streaming genotype I/O against the committed example bed
covers fasterlmm.io_stream: lazy open + row alignment, per-block standardised
reads, the strided grm marker pick, the whole-matrix and per-chrom resident
paths, and the raw per-variant variance pass.  cpu-only and portable, no gpu
and no fastlmm needed -- everything runs off data/example/example at float64
where exactness matters
the load-bearing invariant: a streamed, block-by-block reconstruction of the
standardised genotype must equal a resident read_plink + standardise_columns
of the same row order, otherwise streaming would be an aproximation and not a
bit-for-bit substitute
"""
from __future__ import annotations

import numpy as np
import torch

from fasterlmm.io import read_plink, read_phen, standardise_columns
from fasterlmm.io_stream import (
    BedHandle,
    chrom_test_blocks,
    open_aligned_bed,
    read_all_standardised,
    read_block,
    read_grm_factor,
    resident_chrom_blocks,
    select_grm_markers,
    stream_genotype_var,
)

DTYPE = torch.float64


def _handle(example_geno, example_pheno) -> BedHandle:
    """lazy handle aligned to the pheno strain set, the same keep_iid a real run uses"""
    keep = read_phen(example_pheno).iid
    return open_aligned_bed(str(example_geno), keep)


def _resident_aligned(example_geno, iid_order):
    """resident read of the bed, rows reordered onto iid_order, columns standardised"""
    g = read_plink(str(example_geno))
    pos = {s: i for i, s in enumerate(g.iid)}
    ridx = [pos[s] for s in iid_order]
    Z = g.Z[ridx, :]
    return standardise_columns(Z), g.chrom


# ---------------------------------------------------------------------------
# open_aligned_bed: metadata + row alignment
# ---------------------------------------------------------------------------


def test_open_aligned_bed_iid_is_bed_order_intersection(example_geno, example_pheno):
    """handle iid is the bed-order rows filtered to the keep set, lengths line up"""
    g = read_plink(str(example_geno))
    keep = read_phen(example_pheno).iid
    h = _handle(example_geno, example_pheno)
    # the example bed and pheno share every strain so the intersection is the whole bed
    expected = [s for s in g.iid if s in set(keep)]
    assert h.iid == expected
    assert len(h.iid) == len(h.row_idx)
    # row_idx points back at the right bed rows in bed order
    assert h.row_idx.tolist() == [g.iid.index(s) for s in h.iid]
    assert list(h.row_idx) == sorted(h.row_idx)


def test_open_aligned_bed_keep_subset_preserves_bed_order(example_geno, example_pheno):
    """passing a shuffled subset of keep_iid still comes back in bed file order, filtered"""
    g = read_plink(str(example_geno))
    # take a scrambled subset of strains and confirm the handle ignores the given order
    rng = np.random.default_rng(19930909)
    sub = list(rng.permutation(g.iid)[:40])
    h = open_aligned_bed(str(example_geno), sub)
    expected = [s for s in g.iid if s in set(sub)]
    assert h.iid == expected
    assert len(h.iid) == 40


def test_open_aligned_bed_metadata_matches_read_plink(example_geno, example_pheno):
    """sid / chrom / pos carried on the handle match the resident read_plink metadata"""
    g = read_plink(str(example_geno))
    h = _handle(example_geno, example_pheno)
    assert h.sid == g.sid
    assert len(h.sid) == g.Z.shape[1]
    assert np.array_equal(h.chrom, g.chrom)
    assert np.array_equal(h.pos, g.pos)


# ---------------------------------------------------------------------------
# read_block: per-column standardise exactness vs resident
# ---------------------------------------------------------------------------


def test_read_block_matches_resident_columns(example_geno, example_pheno):
    """a fancy-indexed block read equals the resident standardised columns it names"""
    h = _handle(example_geno, example_pheno)
    Z_res, _ = _resident_aligned(example_geno, h.iid)
    pick = np.array([0, 3, 7, 11, 500, 999, 1499], dtype=np.int64)
    blk = read_block(h, pick, dtype=DTYPE).numpy()
    assert blk.shape == (len(h.iid), len(pick))
    assert np.abs(blk - Z_res[:, pick]).max() < 1e-12


def test_read_block_dtype_cast(example_geno, example_pheno):
    """the returned tensor lands at the requested dtype, default float32"""
    h = _handle(example_geno, example_pheno)
    cidx = np.arange(20, dtype=np.int64)
    assert read_block(h, cidx, dtype=torch.float64).dtype == torch.float64
    assert read_block(h, cidx).dtype == torch.float32  # default is fp32


def test_streamed_reconstruction_equals_resident(example_geno, example_pheno):
    """the whole standardised matrix rebuilt from per-chrom streamed blocks equals the resident read to ~1e-12"""
    h = _handle(example_geno, example_pheno)
    Z_res, _ = _resident_aligned(example_geno, h.iid)
    Z_stream = np.zeros_like(Z_res)
    seen = 0
    # small block_size so several blocks land per chromosome on the 1500-variant bed
    for c in sorted(set(h.chrom.tolist())):
        for blk, cidx in chrom_test_blocks(h, c, block_size=137, dtype=DTYPE):
            Z_stream[:, cidx] = blk.numpy()
            seen += len(cidx)
    assert seen == len(h.sid)  # every variant got reassembled exactly once
    assert np.abs(Z_stream - Z_res).max() < 1e-12


# ---------------------------------------------------------------------------
# select_grm_markers: strided pick spreads across chromosomes
# ---------------------------------------------------------------------------


def test_select_grm_markers_spreads_across_chroms(example_geno, example_pheno):
    """a strided k-marker pick lands a representative spread, one chrom not hogging it all"""
    h = _handle(example_geno, example_pheno)
    all_chroms = set(h.chrom.tolist())
    gidx = select_grm_markers(h, 200)
    assert len(gidx) <= 200
    assert np.array_equal(gidx, np.unique(gidx))  # sorted + de-duped
    assert gidx.min() >= 0 and gidx.max() < len(h.sid)
    # the genome-ordered stride should touch every chromosome, not bunch up
    assert set(h.chrom[gidx].tolist()) == all_chroms


def test_select_grm_markers_k_ge_M_returns_all(example_geno, example_pheno):
    """k at or above the variant count returns the full arange(M)"""
    h = _handle(example_geno, example_pheno)
    M = len(h.sid)
    assert np.array_equal(select_grm_markers(h, M), np.arange(M))
    assert np.array_equal(select_grm_markers(h, M + 5000), np.arange(M))


def test_select_grm_markers_deterministic(example_geno, example_pheno):
    """the strided pick is deterministic for a given k -- no randomness despite the seed arg"""
    h = _handle(example_geno, example_pheno)
    a = select_grm_markers(h, 173, seed=1)
    b = select_grm_markers(h, 173, seed=999)
    assert np.array_equal(a, b)  # linspace stride ignores the seed entirely


# ---------------------------------------------------------------------------
# read_grm_factor == read_block on the same cols
# ---------------------------------------------------------------------------


def test_read_grm_factor_equals_read_block(example_geno, example_pheno):
    """read_grm_factor is just read_block under another name, identical on the same cols"""
    h = _handle(example_geno, example_pheno)
    cols = select_grm_markers(h, 120)
    a = read_grm_factor(h, cols, dtype=DTYPE)
    b = read_block(h, cols, dtype=DTYPE)
    assert torch.equal(a, b)
    assert a.shape == (len(h.iid), len(cols))


# ---------------------------------------------------------------------------
# read_all_standardised: resident slab == streamed reassembly
# ---------------------------------------------------------------------------


def test_read_all_standardised_equals_streamed_blocks(example_geno, example_pheno):
    """the resident whole-matrix slab equals the per-chrom chrom_test_blocks reassembly"""
    h = _handle(example_geno, example_pheno)
    Z = read_all_standardised(h, block_size=137, dtype=DTYPE)
    assert Z.shape == (len(h.iid), len(h.sid))
    Z_chk = torch.zeros_like(Z)
    for c in sorted(set(h.chrom.tolist())):
        for blk, cidx in chrom_test_blocks(h, c, block_size=137, dtype=DTYPE):
            Z_chk[:, cidx] = blk
    assert torch.equal(Z, Z_chk)


def test_read_all_standardised_equals_resident_read_plink(example_geno, example_pheno):
    """the slab also matches a plain read_plink + standardise_columns of the same row order"""
    h = _handle(example_geno, example_pheno)
    Z = read_all_standardised(h, block_size=512, dtype=DTYPE).numpy()
    Z_res, _ = _resident_aligned(example_geno, h.iid)
    assert np.abs(Z - Z_res).max() < 1e-12


# ---------------------------------------------------------------------------
# resident_chrom_blocks: slices a resident slab, matches the disk path
# ---------------------------------------------------------------------------


def test_resident_chrom_blocks_match_chrom_test_blocks(example_geno, example_pheno):
    """slicing the resident slab yields blocks identical to the on-disk chrom_test_blocks"""
    h = _handle(example_geno, example_pheno)
    Z = read_all_standardised(h, block_size=512, dtype=DTYPE)
    for c in sorted(set(h.chrom.tolist())):
        disk = list(chrom_test_blocks(h, c, block_size=137, dtype=DTYPE))
        res = list(resident_chrom_blocks(Z, h.chrom, c, block_size=137))
        assert len(disk) == len(res)
        for (d_blk, d_idx), (r_blk, r_idx) in zip(disk, res):
            assert np.array_equal(d_idx, r_idx)
            assert torch.equal(d_blk, r_blk)


def test_resident_chrom_blocks_cover_chrom_columns(example_geno, example_pheno):
    """the per-chrom column indices reassemble to exactly that chromosome's columns"""
    h = _handle(example_geno, example_pheno)
    Z = read_all_standardised(h, block_size=512, dtype=DTYPE)
    c = sorted(set(h.chrom.tolist()))[0]
    expected = np.where(h.chrom == c)[0]
    gathered = np.concatenate([cidx for _, cidx in resident_chrom_blocks(Z, h.chrom, c, block_size=137)])
    assert np.array_equal(gathered, expected)


# ---------------------------------------------------------------------------
# stream_genotype_var: raw nan-aware per-variant variance
# ---------------------------------------------------------------------------


def test_stream_genotype_var_matches_nanvar(example_geno, example_pheno):
    """the streamed raw variance matches np.nanvar of the row-aligned raw genotype (ddof 0)"""
    g = read_plink(str(example_geno))
    h = _handle(example_geno, example_pheno)
    pos = {s: i for i, s in enumerate(g.iid)}
    ridx = [pos[s] for s in h.iid]
    raw = g.Z[ridx, :]
    expected = np.nanvar(raw, axis=0)
    got = stream_genotype_var(h, block_size=137)
    assert got.shape == (len(h.sid),)
    assert np.abs(got - expected).max() < 1e-12


def test_stream_genotype_var_block_size_invariant(example_geno, example_pheno):
    """the streamed variance is identical regardless of block_size -- it's a per-column reduction"""
    h = _handle(example_geno, example_pheno)
    small = stream_genotype_var(h, block_size=64)
    whole = stream_genotype_var(h, block_size=100000)  # one block over the whole bed
    assert np.array_equal(small, whole)
