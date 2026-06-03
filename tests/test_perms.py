"""
perms.perm_threshold against the committed example bed
covers the permutation null: per-pheno seeded shuffles, the (B, n_perm) perm_max_F
plus the (M, B) real F, both loco branches, and the batching invariant where a pheno
sees the same perms whether it rides alone or in a wider batch.
cpu-only and portable, no fastlmm and no gpu needed.  n_perm stays tiny so the whole
file runs in a couple seconds on a fresh clone
"""

from __future__ import annotations

import numpy as np
import torch

from fasterlmm.io import align_inputs, read_phen, read_plink
from fasterlmm.perms import perm_threshold


def _data(example_geno, example_pheno):
    """align the example genotype + phenotype into one AlignedDataset"""
    g = read_plink(example_geno)
    p = read_phen(example_pheno)
    return align_inputs(g, p)


def test_perm_threshold_same_seed_pheno_identical(example_geno, example_pheno):
    """same (seed, pheno_idx) -> bit-identical perm_max_F, the shuffles are deterministic"""
    data = _data(example_geno, example_pheno)
    _, pmf_a = perm_threshold(data, [0], n_perm=4, seed=19930909, loco=True)
    _, pmf_b = perm_threshold(data, [0], n_perm=4, seed=19930909, loco=True)
    # seeded per (seed, pheno) so two calls land on exactly the same row-orders
    assert torch.allclose(pmf_a, pmf_b, atol=0.0)
    assert torch.equal(pmf_a, pmf_b)


def test_perm_threshold_different_seeds_differ(example_geno, example_pheno):
    """diferent seeds drive different shuffles so the perm_max_F should not match"""
    data = _data(example_geno, example_pheno)
    _, pmf_a = perm_threshold(data, [0], n_perm=4, seed=19930909, loco=True)
    _, pmf_b = perm_threshold(data, [0], n_perm=4, seed=42, loco=True)
    assert not torch.allclose(pmf_a, pmf_b)


def test_perm_threshold_shapes(example_geno, example_pheno):
    """perm_max_F is (B, n_perm) and res.f is (M, B) with B the number of phenos asked for"""
    data = _data(example_geno, example_pheno)
    M = data.Z.shape[1]
    pheno_idx = [0, 1, 2]
    B = len(pheno_idx)
    n_perm = 3
    res, pmf = perm_threshold(data, pheno_idx, n_perm=n_perm, seed=19930909, loco=True)
    assert pmf.shape == (B, n_perm)
    assert res.f.shape == (M, B)
    # the underlying scan also kept the real cols plus every perm column on max_F
    assert res.max_F.shape == (B + B * n_perm,)


def test_perm_threshold_single_pheno_shape(example_geno, example_pheno):
    """a one-pheno batch still comes back (1, n_perm) and (M, 1)"""
    data = _data(example_geno, example_pheno)
    M = data.Z.shape[1]
    res, pmf = perm_threshold(data, [0], n_perm=4, seed=19930909, loco=True)
    assert pmf.shape == (1, 4)
    assert res.f.shape == (M, 1)


def test_perm_threshold_loco_true_finite(example_geno, example_pheno):
    """loco=True returns finite real F and finite perm_max_F, nothing nan or inf"""
    data = _data(example_geno, example_pheno)
    res, pmf = perm_threshold(data, [0, 1], n_perm=3, seed=19930909, loco=True)
    assert torch.isfinite(res.f).all()
    assert torch.isfinite(pmf).all()
    # F is a variance-ratio so it should sit at or above zero
    assert (res.f >= 0).all()
    assert (pmf >= 0).all()


def test_perm_threshold_loco_false_finite(example_geno, example_pheno):
    """loco=False (single K over all variants) also returns finite results"""
    data = _data(example_geno, example_pheno)
    res, pmf = perm_threshold(data, [0, 1], n_perm=3, seed=19930909, loco=False)
    assert torch.isfinite(res.f).all()
    assert torch.isfinite(pmf).all()
    assert (res.f >= 0).all()
    assert (pmf >= 0).all()


def test_perm_threshold_loco_branches_differ(example_geno, example_pheno):
    """the loco and single-K branches run different scans so their real F should differ"""
    data = _data(example_geno, example_pheno)
    res_loco, _ = perm_threshold(data, [0], n_perm=2, seed=19930909, loco=True)
    res_single, _ = perm_threshold(data, [0], n_perm=2, seed=19930909, loco=False)
    # same shape, but leaving each chrom out vs one shared K is not the same model
    assert res_loco.f.shape == res_single.f.shape
    assert not torch.allclose(res_loco.f, res_single.f)


def test_perm_threshold_batching_stability_perm_max_F(example_geno, example_pheno):
    """a 2-pheno batch gives pheno 0 the same perm null it gets when scanned alone"""
    data = _data(example_geno, example_pheno)
    _, pmf_batch = perm_threshold(data, [0, 1], n_perm=4, seed=19930909, loco=True)
    _, pmf_solo = perm_threshold(data, [0], n_perm=4, seed=19930909, loco=True)
    # row 0 of the batch is pheno 0's perm null.  the batched scan widens the pheno GEMM
    # from 1 to 2 columns, which can reroute the BLAS kernel and shift the reduction by a
    # ULP, so this is allclose not bit-equal -- a wrong seed or mixed-up pheno index would
    # redraw the whole null and blow past this tolerance anyway
    assert torch.allclose(pmf_batch[0], pmf_solo[0], rtol=1e-7, atol=1e-9)


def test_perm_threshold_batching_stability_real_f(example_geno, example_pheno):
    """the real per-variant F for pheno 0 matches whether batched with pheno 1 or not"""
    data = _data(example_geno, example_pheno)
    res_batch, _ = perm_threshold(data, [0, 1], n_perm=3, seed=19930909, loco=True)
    res_solo, _ = perm_threshold(data, [0], n_perm=3, seed=19930909, loco=True)
    # genotype-only eigendecomp means batching cannot move pheno 0's real-F column, modulo
    # the same 1-vs-2-column GEMM rounding as the perm-null test, so allclose not bit-equal
    assert torch.allclose(res_batch.f[:, 0], res_solo.f[:, 0], rtol=1e-7, atol=1e-9)


def test_perm_threshold_batch_order_independent(example_geno, example_pheno):
    """asking [0, 1] vs [1, 0] just permutes the rows, pheno 1 keeps its own perms either way"""
    data = _data(example_geno, example_pheno)
    _, pmf_01 = perm_threshold(data, [0, 1], n_perm=3, seed=19930909, loco=True)
    _, pmf_10 = perm_threshold(data, [1, 0], n_perm=3, seed=19930909, loco=True)
    # pheno 1 is row 1 in the first call, row 0 in the second, perms keyed on its own index
    assert torch.equal(pmf_01[1], pmf_10[0])
    assert torch.equal(pmf_01[0], pmf_10[1])


def test_perm_threshold_distinct_phenos_differ(example_geno, example_pheno):
    """two different phenos in one batch get independent shuffles, so their nulls differ"""
    data = _data(example_geno, example_pheno)
    _, pmf = perm_threshold(data, [0, 1], n_perm=4, seed=19930909, loco=True)
    # each pheno is seeded on its own column index so the two rows are not the same draw
    assert not torch.allclose(pmf[0], pmf[1])


def test_perm_threshold_on_chrom_progress_callback(example_geno, example_pheno):
    """on_chrom is a per-chromosome progress hook, it fires once per chrom and the result is unchanged"""
    data = _data(example_geno, example_pheno)
    seen = []

    def cb(done, total):
        seen.append((done, total))

    res_cb, pmf_cb = perm_threshold(data, [0], n_perm=2, seed=19930909, loco=True, on_chrom=cb)
    res_no, pmf_no = perm_threshold(data, [0], n_perm=2, seed=19930909, loco=True)
    n_chrom = len(np.unique(np.asarray(data.chrom)))
    # the hook fires once for every chromosome the loco scan walks
    assert len(seen) == n_chrom
    assert seen[-1] == (n_chrom, n_chrom)
    # the callback is cosmetic, the scan output must be untouched by passing it
    assert torch.equal(res_cb.f, res_no.f)
    assert torch.equal(pmf_cb, pmf_no)
