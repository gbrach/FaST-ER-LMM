"""
separate kinship panel (--kinship-geno): LOCO / single-K scans, perm_threshold and alignment
cpu-only synthetic panels, no GPU and no fastlmm
"""
from __future__ import annotations

import numpy as np
import torch

from fasterlmm.core import loco_scan, loco_scan_compat, single_k_scan, single_k_scan_compat
from fasterlmm.io import Genotypes, Phenotypes, align_inputs, standardise_columns
from fasterlmm.perms import perm_threshold

DTYPE = torch.float64


def _panel(N = 60, M = 120, seed = 0):
    rng = np.random.default_rng(seed)
    Z = torch.from_numpy(standardise_columns(rng.normal(size = (N, M)))).to(DTYPE)
    chrom = np.repeat([1, 2, 3], M // 3).astype(float)
    return Z, chrom


def _Y(N, seed = 1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N, 4, generator = g, dtype = DTYPE)


def test_kin_equal_to_geno_reproduces_scans():
    Z, chrom = _panel()
    X = torch.ones(Z.shape[0], 1, dtype = DTYPE)
    Y = _Y(Z.shape[0])
    for fn in (loco_scan, loco_scan_compat):
        a = fn(Z, X, Y, chrom)
        b = fn(Z, X, Y, chrom, Z_kin = Z.clone(), chrom_kin = chrom.copy())
        assert torch.equal(a.f, b.f)
    for fn in (single_k_scan, single_k_scan_compat):
        assert torch.equal(fn(Z, X, Y).f, fn(Z, X, Y, Z_kin = Z.clone()).f)


def test_kin_panel_changes_loco_when_different():
    Z, chrom = _panel()
    Zk, _ = _panel(M = 240, seed = 5)
    chrom_k = np.repeat([1, 2, 3], 80).astype(float)
    X = torch.ones(Z.shape[0], 1, dtype = DTYPE)
    Y = _Y(Z.shape[0])
    a = loco_scan_compat(Z, X, Y, chrom)
    b = loco_scan_compat(Z, X, Y, chrom, Z_kin = Zk, chrom_kin = chrom_k)
    assert a.f.shape == b.f.shape
    assert not torch.allclose(a.f, b.f)


def test_kin_panel_drops_tested_chrom():
    # K for chrom k must exclude chrom k of the kinship panel, poisoning its chrom-2 columns cant move chrom-2 results
    Z, chrom = _panel()
    Zk, chrom_k = _panel(seed = 7)
    X = torch.ones(Z.shape[0], 1, dtype = DTYPE)
    Y = _Y(Z.shape[0])
    a = loco_scan_compat(Z, X, Y, chrom, Z_kin = Zk, chrom_kin = chrom_k)
    Zp = Zk.clone()
    Zp[:, chrom_k == 2] = torch.randn_like(Zp[:, chrom_k == 2])
    b = loco_scan_compat(Z, X, Y, chrom, Z_kin = Zp, chrom_kin = chrom_k)
    m2 = torch.from_numpy(chrom == 2)
    assert torch.equal(a.f[m2], b.f[m2])
    assert not torch.equal(a.f[~m2], b.f[~m2])


def test_perm_threshold_uses_kin_panel():
    rng = np.random.default_rng(3)
    N = 40
    iid = [f"s{i}" for i in range(N)]
    chrom = np.repeat([1, 2], 15).astype(float)
    geno = Genotypes(Z = rng.integers(0, 3, (N, 30)).astype(float), iid = iid, sid = [f"v{i}" for i in range(30)],
                     chrom = chrom, pos = np.arange(30))
    kin = Genotypes(Z = rng.integers(0, 3, (N, 24)).astype(float), iid = iid, sid = [f"k{i}" for i in range(24)],
                    chrom = np.repeat([1, 2], 12).astype(float), pos = np.arange(24))
    pheno = Phenotypes(iid = iid, names = ["p"], Y = rng.normal(size = (N, 1)))
    plain, _ = perm_threshold(align_inputs(geno, pheno), [0], n_perm = 4)
    sep, pmax = perm_threshold(align_inputs(geno, pheno, kin = kin), [0], n_perm = 4)
    assert pmax.shape == (1, 4)
    assert not torch.allclose(plain.f, sep.f)


def test_align_inputs_reorders_kin_strains():
    N = 6
    iid = [f"s{i}" for i in range(N)]
    rng = np.random.default_rng(0)
    Zt = rng.integers(0, 3, (N, 5)).astype(float)
    Zk = rng.integers(0, 3, (N, 7)).astype(float)
    perm = [3, 0, 5, 1, 4, 2]
    geno = Genotypes(Z = Zt, iid = iid, sid = list("abcde"), chrom = np.ones(5), pos = np.arange(5))
    kin = Genotypes(Z = Zk[perm], iid = [iid[i] for i in perm], sid = list("vwxyzuq"), chrom = np.ones(7),
                    pos = np.arange(7))
    pheno = Phenotypes(iid = iid, names = ["p"], Y = rng.normal(size = (N, 1)))
    al = align_inputs(geno, pheno, None, kin = kin)
    assert al.iid == iid
    assert np.array_equal(al.Z_kin.numpy(), Zk)


def test_align_inputs_counts_dropped_kin_strains():
    iid = [f"s{i}" for i in range(6)]
    rng = np.random.default_rng(1)
    geno = Genotypes(Z = rng.random((6, 3)), iid = iid, sid = list("abc"), chrom = np.ones(3), pos = np.arange(3))
    kin = Genotypes(Z = rng.random((4, 3)), iid = iid[:4], sid = list("xyz"), chrom = np.ones(3), pos = np.arange(3))
    pheno = Phenotypes(iid = iid, names = ["p"], Y = rng.normal(size = (6, 1)))
    al = align_inputs(geno, pheno, None, kin = kin)
    assert al.iid == iid[:4] and al.n_dropped_kin == 2
