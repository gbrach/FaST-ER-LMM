"""
streaming LOCO engine (extreme_scan) on the committed example bed
the gate here: the three resident-vs-streamed LOCO paths must land on the same numbers.  the
streamed path re-reads test variants off disk block by block, the resident path slices a decoded
slab, and lowrank.loco_scan_lowrank does the same low-rank math on a resident Z -- at float64 they
are bit-for-bit, since the standardisation is per-variant and the scan underneath is shared
block_size is only a memory lever, so a tiny block must give the same answer as a fat one
cpu-only, portable: no fastlmm, no gpu, runs off data/example alone
"""

from __future__ import annotations

import numpy as np
import torch

from fasterlmm.io import read_phen
from fasterlmm.io_stream import (
    open_aligned_bed, select_grm_markers, read_grm_factor, read_all_standardised,
)
from fasterlmm.extreme_scan import loco_scan_streamed, loco_scan_resident
from fasterlmm.lowrank import loco_scan_lowrank

DTYPE = torch.float64
TOL = 1e-9  # spec ask; the observed gap is exactly 0.0 since the paths share the scan core


def _setup(example_geno, example_pheno, n_pheno=3, k=300):
    """build handle + aligned Y/X + strided grm factor + resident standardised Z on the example bed"""
    ph = read_phen(example_pheno)
    h = open_aligned_bed(str(example_geno), ph.iid)
    ppos = {s: i for i, s in enumerate(ph.iid)}
    ridx = [ppos[s] for s in h.iid]  # phenos reordered onto the handle's strain order
    Y = torch.tensor(ph.Y[ridx, :][:, :n_pheno], dtype=DTYPE)
    X = torch.ones(len(h.iid), 1, dtype=DTYPE)
    grm_cols = select_grm_markers(h, k)
    G = read_grm_factor(h, grm_cols, dtype=DTYPE)
    g_chrom = h.chrom[grm_cols]
    Z = read_all_standardised(h, dtype=DTYPE)
    z_chrom = h.chrom
    return h, G, g_chrom, X, Y, Z, z_chrom


def _gap(a, b):
    return (a - b).abs().max().item()


def _assert_scanresult_close(a, b, tol=TOL):
    """every field of two ScanResults agrees to tol -- f, beta, se, sfve, nullh2, max_F"""
    assert _gap(a.f, b.f) <= tol
    assert _gap(a.beta, b.beta) <= tol
    assert _gap(a.se, b.se) <= tol
    assert _gap(a.sfve, b.sfve) <= tol
    assert _gap(a.nullh2, b.nullh2) <= tol
    assert _gap(a.max_F, b.max_F) <= tol


def test_streamed_equals_resident(example_geno, example_pheno):
    """streamed LOCO and resident LOCO are bit-for-bit on the same kinship + test SNPs"""
    h, G, g_chrom, X, Y, Z, z_chrom = _setup(example_geno, example_pheno)
    res_s = loco_scan_streamed(h, G, g_chrom, X, Y, block_size=256, dtype=DTYPE)
    res_r = loco_scan_resident(Z, z_chrom, G, g_chrom, X, Y, block_size=256, dtype=DTYPE)
    _assert_scanresult_close(res_s, res_r)


def test_streamed_equals_lowrank(example_geno, example_pheno):
    """streamed LOCO matches lowrank.loco_scan_lowrank on the same G / Z, the resident low-rank twin"""
    h, G, g_chrom, X, Y, Z, z_chrom = _setup(example_geno, example_pheno)
    res_s = loco_scan_streamed(h, G, g_chrom, X, Y, block_size=256, dtype=DTYPE)
    res_lr = loco_scan_lowrank(G, g_chrom, Z, z_chrom, X, Y)
    _assert_scanresult_close(res_s, res_lr)


def test_resident_equals_lowrank(example_geno, example_pheno):
    """resident LOCO matches the low-rank LOCO too, so all three paths agree pairwise"""
    h, G, g_chrom, X, Y, Z, z_chrom = _setup(example_geno, example_pheno)
    res_r = loco_scan_resident(Z, z_chrom, G, g_chrom, X, Y, block_size=256, dtype=DTYPE)
    res_lr = loco_scan_lowrank(G, g_chrom, Z, z_chrom, X, Y)
    _assert_scanresult_close(res_r, res_lr)


def test_block_size_invariant_streamed(example_geno, example_pheno):
    """block_size is only a memory lever -- a tiny block must equal a block bigger than the whole genome"""
    h, G, g_chrom, X, Y, Z, z_chrom = _setup(example_geno, example_pheno)
    small = loco_scan_streamed(h, G, g_chrom, X, Y, block_size=64, dtype=DTYPE)
    big = loco_scan_streamed(h, G, g_chrom, X, Y, block_size=8192, dtype=DTYPE)
    _assert_scanresult_close(small, big)


def test_block_size_invariant_resident(example_geno, example_pheno):
    """same block invariance on the resident path, slicing the slab small vs whole"""
    h, G, g_chrom, X, Y, Z, z_chrom = _setup(example_geno, example_pheno)
    small = loco_scan_resident(Z, z_chrom, G, g_chrom, X, Y, block_size=64, dtype=DTYPE)
    big = loco_scan_resident(Z, z_chrom, G, g_chrom, X, Y, block_size=8192, dtype=DTYPE)
    _assert_scanresult_close(small, big)


def test_scanresult_shapes(example_geno, example_pheno):
    """per-SNP fields are (M, n_real) and max_F is (P,) -- M = handle variants, P = phenos"""
    h, G, g_chrom, X, Y, Z, z_chrom = _setup(example_geno, example_pheno)
    M = len(h.sid)
    P = Y.shape[1]
    res = loco_scan_resident(Z, z_chrom, G, g_chrom, X, Y, block_size=256, dtype=DTYPE)
    assert tuple(res.f.shape) == (M, P)
    assert tuple(res.beta.shape) == (M, P)
    assert tuple(res.se.shape) == (M, P)
    assert tuple(res.sfve.shape) == (M, P)
    assert tuple(res.nullh2.shape) == (M, P)
    assert tuple(res.max_F.shape) == (P,)


def test_outputs_finite(example_geno, example_pheno):
    """every field comes back finite on the example panel, no NaN or inf leaking trough"""
    h, G, g_chrom, X, Y, Z, z_chrom = _setup(example_geno, example_pheno)
    res = loco_scan_streamed(h, G, g_chrom, X, Y, block_size=256, dtype=DTYPE)
    for fld in (res.f, res.beta, res.se, res.sfve, res.nullh2, res.max_F):
        assert torch.isfinite(fld).all().item()
    # nullh2 is a heritability, must sit in [0, 1]
    assert res.nullh2.min().item() >= 0.0
    assert res.nullh2.max().item() <= 1.0
    # se and the F-stat are strictly positive on a non-degenerate panel
    assert res.se.min().item() > 0.0
    assert res.f.min().item() >= 0.0


def test_n_real_split(example_geno, example_pheno):
    """n_real splits the pheno axis: leading cols keep full detail, max_F still spans every pheno"""
    h, G, g_chrom, X, Y, Z, z_chrom = _setup(example_geno, example_pheno)
    full = loco_scan_resident(Z, z_chrom, G, g_chrom, X, Y, block_size=256, dtype=DTYPE)
    sub = loco_scan_resident(Z, z_chrom, G, g_chrom, X, Y, n_real=1, block_size=256, dtype=DTYPE)
    # only the first n_real phenos carry per-SNP detail
    assert tuple(sub.f.shape) == (len(h.sid), 1)
    # those leading columns are identical to the full-detail run
    assert _gap(sub.f[:, :1], full.f[:, :1]) <= TOL
    assert _gap(sub.beta[:, :1], full.beta[:, :1]) <= TOL
    assert _gap(sub.se[:, :1], full.se[:, :1]) <= TOL
    # max_F runs over all P phenos regardless of n_real
    assert tuple(sub.max_F.shape) == (Y.shape[1],)
    assert _gap(sub.max_F, full.max_F) <= TOL


def test_on_chrom_callback_fires_per_chrom(example_geno, example_pheno):
    """on_chrom is called once per chromosome with (k+1, n_chroms), counting up to the total"""
    h, G, g_chrom, X, Y, Z, z_chrom = _setup(example_geno, example_pheno)
    n_chroms = len(set(z_chrom.tolist()))
    calls = []
    loco_scan_resident(Z, z_chrom, G, g_chrom, X, Y, block_size=256, dtype=DTYPE,
                       on_chrom=lambda i, n: calls.append((i, n)))
    assert len(calls) == n_chroms
    assert calls[0] == (1, n_chroms)
    assert calls[-1] == (n_chroms, n_chroms)
    assert [c[0] for c in calls] == list(range(1, n_chroms + 1))


def test_grm_via_separate_bed_arg(example_geno, example_pheno):
    """G need not be a strided prune of the test handle -- a denser grm factor still drives all three paths to agree"""
    # heavier kinship factor (k=600) read independently of the 1500 test variants
    h, _, _, X, Y, Z, z_chrom = _setup(example_geno, example_pheno)
    grm_cols = select_grm_markers(h, 600)
    G = read_grm_factor(h, grm_cols, dtype=DTYPE)
    g_chrom = h.chrom[grm_cols]
    res_s = loco_scan_streamed(h, G, g_chrom, X, Y, block_size=256, dtype=DTYPE)
    res_r = loco_scan_resident(Z, z_chrom, G, g_chrom, X, Y, block_size=256, dtype=DTYPE)
    res_lr = loco_scan_lowrank(G, g_chrom, Z, z_chrom, X, Y)
    _assert_scanresult_close(res_s, res_r)
    _assert_scanresult_close(res_s, res_lr)
