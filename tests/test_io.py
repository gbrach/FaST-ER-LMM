"""
cpu-only unit tests for fasterlmm.io: the loading + alignment + standardise layer
covers standardise_columns (numpy + torch branches, nan handling, constant cols),
read_plink / read_phen / read_covar against the committed example bed + tsv,
align_inputs (intercept-only and with covariates, strain intersection order),
and grm shape / symmetry / divisor.  portable, no gpu and no fastlmm needed
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fasterlmm.io import (
    align_inputs,
    grm,
    read_covar,
    read_phen,
    read_plink,
    standardise_columns,
)


# ---------------------------------------------------------------------------
# standardise_columns
# ---------------------------------------------------------------------------


def test_standardise_columns_mean0_std1():
    """a plain numeric block comes back with population mean 0 and std 1 per column"""
    rng = np.random.default_rng(19930909)
    a = rng.normal(size=(60, 4)) * 2.5 - 1.0
    out = standardise_columns(a)
    assert out.shape == a.shape
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-12)
    # ddof=0 population std, so compare against np.std default
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-12)


def test_standardise_columns_nan_zeroed_after():
    """NaN cells survive the mean / std math but land at exactly 0 after standardising"""
    a = np.array([[1.0, 10.0],
                  [np.nan, 20.0],
                  [3.0, np.nan],
                  [5.0, 40.0],
                  [7.0, 50.0]])
    out = standardise_columns(a)
    assert out[1, 0] == 0.0
    assert out[2, 1] == 0.0
    # the kept (non-nan) cells of each column still center on zero in the population sense
    col0_keep = out[[0, 2, 3, 4], 0]
    col1_keep = out[[0, 1, 3, 4], 1]
    assert np.isclose(col0_keep.mean(), 0.0, atol=1e-12)
    assert np.isclose(col1_keep.mean(), 0.0, atol=1e-12)


def test_standardise_columns_constant_column_zeros():
    """a constant column has std 0 so it should zero out cleanly, no inf or nan leaking"""
    a = np.array([[2.0, 1.0],
                  [2.0, 4.0],
                  [2.0, 9.0],
                  [2.0, 16.0]])
    out = standardise_columns(a)
    assert np.allclose(out[:, 0], 0.0)
    assert np.isfinite(out).all()
    # the varying column is untouched by the constant one, still standardised
    assert np.isclose(out[:, 1].mean(), 0.0, atol=1e-12)
    assert np.isclose(out[:, 1].std(), 1.0, atol=1e-12)


def test_standardise_columns_torch_matches_numpy():
    """the torch branch must agree with the numpy branch on the same data incl nans"""
    rng = np.random.default_rng(2024)
    a = rng.normal(size=(50, 6))
    # sprinkle a few nans and one fully constant column
    a[3, 0] = np.nan
    a[10, 2] = np.nan
    a[:, 4] = 5.0
    out_np = standardise_columns(a)
    out_t = standardise_columns(torch.from_numpy(a.copy()))
    assert isinstance(out_t, torch.Tensor)
    assert np.allclose(out_np, out_t.numpy(), atol=1e-12, equal_nan=True)


def test_standardise_columns_returns_same_type():
    """numpy in -> ndarray out, torch in -> Tensor out"""
    a = np.arange(12.0).reshape(4, 3)
    assert isinstance(standardise_columns(a), np.ndarray)
    assert isinstance(standardise_columns(torch.from_numpy(a)), torch.Tensor)


def test_standardise_columns_1d_input():
    """a 1d vector standardises as a single column"""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = standardise_columns(a)
    assert out.shape == a.shape
    assert np.isclose(out.mean(), 0.0, atol=1e-12)
    assert np.isclose(out.std(), 1.0, atol=1e-12)


def test_standardise_columns_does_not_mutate_input():
    """the helper copies, so the caller's array is left alone"""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    before = a.copy()
    _ = standardise_columns(a)
    assert np.array_equal(a, before)


# ---------------------------------------------------------------------------
# read_plink
# ---------------------------------------------------------------------------


def test_read_plink_shape_and_dtype(example_geno):
    """the example bed is 150 strains x 1500 variants, dosage matrix is float64"""
    g = read_plink(example_geno)
    assert g.Z.shape == (150, 1500)
    assert g.Z.dtype == np.float64
    assert len(g.iid) == 150
    assert len(g.sid) == 1500


def test_read_plink_chrom_subset(example_geno):
    """chrom labels are a subset of yeast 1..16, one entry per variant column"""
    g = read_plink(example_geno)
    assert g.chrom.shape == (1500,)
    assert g.pos.shape == (1500,)
    chroms = set(int(c) for c in g.chrom)
    assert chroms.issubset(set(range(1, 17)))
    # the example actually spans the full karyotype
    assert chroms == set(range(1, 17))


def test_read_plink_dosages_in_range(example_geno):
    """count_A1 dosages live in {0,1,2}, missing would be NaN (none here)"""
    g = read_plink(example_geno)
    finite = g.Z[~np.isnan(g.Z)]
    assert finite.min() >= 0.0
    assert finite.max() <= 2.0
    assert set(np.unique(finite).tolist()).issubset({0.0, 1.0, 2.0})


# ---------------------------------------------------------------------------
# read_phen
# ---------------------------------------------------------------------------


def test_read_phen_shape(example_pheno):
    """example pheno table is 150 strains x 20 phenotypes"""
    p = read_phen(example_pheno)
    assert len(p.iid) == 150
    assert len(p.names) == 20
    assert p.Y.shape == (150, 20)
    assert p.Y.dtype == np.float64


def test_read_phen_strain_col_dropped(example_pheno):
    """the first 'Strain' column becomes iid, never sneaks into names or Y"""
    p = read_phen(example_pheno)
    assert "Strain" not in p.names
    # the pheno labels are the actual ORF ids from the header
    assert p.names[0] == "YAL001C"
    assert all(isinstance(s, str) for s in p.iid)


# ---------------------------------------------------------------------------
# read_covar
# ---------------------------------------------------------------------------


def test_read_covar_names_and_shape(example_covar):
    """plink covar is FID IID c1 c2 ..., names come back as cov_0, cov_1, ..."""
    c = read_covar(example_covar)
    assert len(c.iid) == 150
    assert c.C.shape[0] == 150
    K = c.C.shape[1]
    assert c.names == [f"cov_{i}" for i in range(K)]
    assert c.names[0] == "cov_0"
    # the example covar carries 43 covariate columns past FID / IID
    assert K == 43


def test_read_covar_iid_is_second_column(example_covar, example_geno):
    """iid is taken from the IID (2nd) column, the same ids show up in the geno fam"""
    c = read_covar(example_covar)
    assert c.iid[0] == "ADL"
    assert c.iid[1] == "AEN"
    # every covar strain should resolve against a geno strain (full overlap here)
    g = read_plink(example_geno)
    assert set(c.iid).issubset(set(g.iid))


# ---------------------------------------------------------------------------
# align_inputs
# ---------------------------------------------------------------------------


def test_align_inputs_intercept_only(example_geno, example_pheno):
    """with no covar, X is just the all-ones intercept column (N,1)"""
    g = read_plink(example_geno)
    p = read_phen(example_pheno)
    ds = align_inputs(g, p)
    N = len(ds.iid)
    assert ds.X.shape == (N, 1)
    assert torch.allclose(ds.X, torch.ones((N, 1), dtype=ds.X.dtype))
    assert ds.Z.shape[0] == N
    assert ds.Y.shape[0] == N
    assert ds.Z.dtype == torch.float64
    assert ds.Y.dtype == torch.float64


def test_align_inputs_with_covar(example_geno, example_pheno, example_covar):
    """X = [intercept | covariates], first column ones, covar block has K cols"""
    g = read_plink(example_geno)
    p = read_phen(example_pheno)
    c = read_covar(example_covar)
    ds = align_inputs(g, p, c)
    N = len(ds.iid)
    K = c.C.shape[1]
    assert ds.X.shape == (N, 1 + K)
    assert torch.allclose(ds.X[:, 0], torch.ones(N, dtype=ds.X.dtype))
    # the covar block must be the covar rows reordered onto the aligned strain order
    c_idx = [c.iid.index(s) for s in ds.iid]
    expect = torch.from_numpy(c.C[c_idx, :]).to(ds.X.dtype)
    assert torch.allclose(ds.X[:, 1:], expect)


def test_align_inputs_keeps_geno_order(example_geno, example_pheno):
    """the aligned iid order follows the geno file order, not pheno order"""
    g = read_plink(example_geno)
    p = read_phen(example_pheno)
    ds = align_inputs(g, p)
    common = set(g.iid) & set(p.iid)
    geno_order = [s for s in g.iid if s in common]
    assert ds.iid == geno_order


def test_align_inputs_intersection_and_reorder():
    """a synthetic mismatch: align keeps the geno-order intersection and reorders Y to match"""
    from fasterlmm.io import Genotypes, Phenotypes

    # geno has strains in one order, pheno shuffled + with an extra strain pheno-only
    Z = np.array([[0.0, 1.0],
                  [1.0, 2.0],
                  [2.0, 0.0]])  # strains A B C
    g = Genotypes(Z=Z,
                  iid=["A", "B", "C"],
                  sid=["s0", "s1"],
                  chrom=np.array([1, 1]),
                  pos=np.array([10, 20]))
    # pheno carries C, A, D (D is geno-absent, B is pheno-absent)
    Y = np.array([[30.0], [10.0], [99.0]])
    p = Phenotypes(iid=["C", "A", "D"], names=["t0"], Y=Y)
    ds = align_inputs(g, p)
    # intersection is {A, C}, kept in geno order -> A then C
    assert ds.iid == ["A", "C"]
    # Y must be pulled onto that order: A's pheno is 10, C's is 30
    assert torch.allclose(ds.Y.squeeze(1), torch.tensor([10.0, 30.0], dtype=ds.Y.dtype))
    # Z rows likewise: A's geno row [0,1], C's [2,0]
    assert torch.allclose(ds.Z, torch.tensor([[0.0, 1.0], [2.0, 0.0]], dtype=ds.Z.dtype))


def test_align_inputs_no_overlap_raises():
    """zero strain overlap is an error, the scan would have nothing to fit"""
    from fasterlmm.io import Genotypes, Phenotypes

    g = Genotypes(Z=np.zeros((2, 1)),
                  iid=["A", "B"],
                  sid=["s0"],
                  chrom=np.array([1]),
                  pos=np.array([1]))
    p = Phenotypes(iid=["X", "Y"], names=["t0"], Y=np.zeros((2, 1)))
    with pytest.raises(ValueError):
        align_inputs(g, p)


def test_align_inputs_dtype_override(example_geno, example_pheno):
    """dtype kwarg flows through to all three tensors"""
    g = read_plink(example_geno)
    p = read_phen(example_pheno)
    ds = align_inputs(g, p, dtype=torch.float32)
    assert ds.Z.dtype == torch.float32
    assert ds.Y.dtype == torch.float32
    assert ds.X.dtype == torch.float32


# ---------------------------------------------------------------------------
# grm
# ---------------------------------------------------------------------------


def test_grm_shape_symmetry_divisor():
    """K = Z Zt / M, so it is N x N, symmetric, and divided by the variant count"""
    rng = np.random.default_rng(7)
    Zn = standardise_columns(rng.normal(size=(40, 25)))
    Z = torch.from_numpy(Zn)
    M = Z.shape[1]
    K = grm(Z)
    assert K.shape == (40, 40)
    assert torch.allclose(K, K.T, atol=1e-12)
    # divisor is M (variant count), not N
    assert torch.allclose(K, (Z @ Z.T) / M, atol=1e-12)
    # diagonal of a standardised-Z grm averages ~1 (each col contributes ~1 to the trace / M)
    assert np.isclose(K.diagonal().mean().item(), 1.0, atol=0.2)


def test_grm_psd():
    """grm is a gram matrix so all eigenvalues are non-negative"""
    rng = np.random.default_rng(11)
    Z = torch.from_numpy(standardise_columns(rng.normal(size=(30, 50))))
    K = grm(Z)
    evals = torch.linalg.eigvalsh(K)
    assert evals.min().item() > -1e-10
