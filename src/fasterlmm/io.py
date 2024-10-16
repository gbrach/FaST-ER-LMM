"""
Loading plink BED + a phen tsv
Not a port of anything in fastlmm: their loading goes through _snps_fixup and _pheno_fixup in fastlmm_predictor.py, which accept strings or already-built SnpReader/KernelReader objects.
I don't think I'll need that so opening the Bed myself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor


@dataclass
class Genotypes:
    """
    Plink BED contents.
    Z (N, M) is the dosage matrix in count_A1=True convention so values are 0/1/2 A1 alleles, same as fastlmm
    Missing dosages stay as NaN
    iid = N strain IDs, sid = M variant IDs, chrom + pos = variant positions
    """
    Z: np.ndarray
    iid: list[str]
    sid: list[str]
    chrom: np.ndarray
    pos: np.ndarray


@dataclass
class Phenotypes:
    """
    Wide phen table
    iid = N strains, names = P pheno labels, Y (N, P) the float values
    NaN is fine
    """
    iid: list[str]
    names: list[str]
    Y: np.ndarray


def read_plink(prefix: str | Path, *, count_A1: bool = True) -> Genotypes:
    """
    Reading a plink BED/BIM/FAM trio via pysnptools.snpreader.Bed.
    count_A1=True is fastlmm's default, single_snp.py sets it the same way.
    Leaving missing dosages as NaN.
    The mean-impute-then-zero-NaN pattern from pysnptools' Unit standardizer goes in a separate helper later, not pulling it in while there's nothing downstream
    """
    from pysnptools.snpreader import Bed

    bed = Bed(str(prefix), count_A1=count_A1).read()
    return Genotypes(
        Z=np.asarray(bed.val, dtype=np.float64),
        iid=[str(x[1]) for x in bed.iid],
        sid=[str(x) for x in bed.sid],
        chrom=np.asarray(bed.pos[:, 0]),
        pos=np.asarray(bed.pos[:, 2]))


def read_phen(path: str | Path) -> Phenotypes:
    """
    Reading a wide phen tsv: first col 'Strain', rest are pheno values, one row per strain.
    Not plink .phen layout (FID IID pheno).
    fastlmm's _pheno_fixup handles plink format and that's fine for one-pheno runs.
    But most of what I'll throw at this is 6k-pheno transcriptomics tables, wide tsv is easier
    """
    df = pd.read_csv(path, sep="\t")
    iid = df["Strain"].astype(str).tolist()
    names = list(df.columns[1:])
    Y = df[names].to_numpy(dtype=np.float64)
    return Phenotypes(iid=iid, names=names, Y=Y)


@dataclass
class Covariates:
    """plink-style covariate table.  C (N, K) is the covariate matrx, names are the column labels (cov_0, cov_1, ... since plink .cov has no header)"""
    iid: list[str]
    names: list[str]
    C: np.ndarray


def read_covar(path: str | Path) -> Covariates:
    """
    Reading a plink-style covariate file: 'FID IID c1 c2 ...', whitespace-separated, no header.
    Naming the cols cov_0, cov_1, ... since plink .cov dosen't carry covariate names throught
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    iid = df.iloc[:, 1].astype(str).tolist()
    cov = df.iloc[:, 2:].to_numpy(dtype=np.float64)
    names = [f"cov_{i}" for i in range(cov.shape[1])]
    return Covariates(iid=iid, names=names, C=cov)


@dataclass
class AlignedDataset:
    """
    Strain-aligned bundle, torch tensors ready for the scan
    iid is the N strain order, Z (N, M) the genotypes, Y (N, P) the phenos, X (N, C) is [intercept | covariates], chrom/pos/snp_id describe the M columns of Z, pheno_names labels the P columns of Y
    """
    iid: list[str]
    Z: Tensor
    Y: Tensor
    X: Tensor
    chrom: np.ndarray
    pos: np.ndarray
    snp_id: list[str]
    pheno_names: list[str]


def align_inputs(geno: Genotypes,
                 pheno: Phenotypes,
                 covar: Covariates | None = None,
                 *,
                 dtype: torch.dtype = torch.float64) -> AlignedDataset:
    """
    Strain-set intersection across geno, pheno, optional covar.  Reorders all three onto a common N row order, builds X = [intercept | covariates], and the whole thing comes back as torch tensors at the requested dtype
    fastlmm does the equivalent inside its SnpReader pipeline (intersect_apply in pysnptools/util/intersect_apply.py).  Doing it on my side becuase slicing things by hand later is way easier wehn alignment is its own step
    """
    g_set = set(geno.iid)
    p_set = set(pheno.iid)
    common = g_set & p_set
    if covar is not None:
        common &= set(covar.iid)
    if not common:
        raise ValueError("no strain overlap between geno / pheno / covar")
    iid = [s for s in geno.iid if s in common]  # keeping geno file order so runs reproduce
    g_idx = [geno.iid.index(s) for s in iid]
    p_idx = [pheno.iid.index(s) for s in iid]
    Z = geno.Z[g_idx, :]
    Y = pheno.Y[p_idx, :]
    if covar is not None:
        c_idx = [covar.iid.index(s) for s in iid]
        X = np.concatenate([np.ones((len(iid), 1)), covar.C[c_idx, :]], axis=1)
    else:
        X = np.ones((len(iid), 1))
    return AlignedDataset(iid=iid,
                          Z=torch.from_numpy(Z).to(dtype),
                          Y=torch.from_numpy(Y).to(dtype),
                          X=torch.from_numpy(X).to(dtype),
                          chrom=geno.chrom,
                          pos=geno.pos,
                          snp_id=geno.sid,
                          pheno_names=pheno.names)


def standardise_columns(arr: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """
    PORT IS DONE!
    Copy of pysnptools Unit standardizer (pysnptools/standardizer/standardizer.py lines 206-238)
    Ordering matters and fastlmm depends on it exactly:
      1. NaN-aware column mean and std (population, ddof=0)
      2. subtract mean, divide by std
      3. zero NaNs AFTER standardising
      4. constant cols (std=0) get std := inf so they zero out cleanly
    Numpy and torch branches both kept because Z lands either way dependig on whether the caller has handed me a raw bed.val or a torch tensor already
    """
    if isinstance(arr, Tensor):
        a = arr.clone()
        nan_mask = torch.isnan(a)
        a_stats = torch.where(nan_mask, torch.zeros_like(a), a)
        n_nonnan = (~nan_mask).sum(dim=0).clamp(min=1).to(a.dtype)
        col_mean = a_stats.sum(dim=0) / n_nonnan
        diff = a_stats - col_mean.unsqueeze(0)
        diff = torch.where(nan_mask, torch.zeros_like(diff), diff)
        col_std = ((diff * diff).sum(dim=0) / n_nonnan).sqrt()
        col_std = torch.where(col_std == 0, torch.full_like(col_std, float("inf")), col_std)
        a = (a - col_mean.unsqueeze(0)) / col_std.unsqueeze(0)
        return torch.where(nan_mask, torch.zeros_like(a), a)
    a = arr.astype(np.float64).copy()
    nan_mask = np.isnan(a)
    col_mean = np.nanmean(a, axis=0)
    col_std = np.nanstd(a, axis=0)
    col_std = np.where(col_std == 0, np.inf, col_std)
    a = (a - col_mean) / col_std
    a[nan_mask] = 0.0
    return a


def grm(Z: Tensor) -> Tensor:
    """
    K = Z Zᵀ / M.  Z must already be standardised (mean 0, std 1) per column, otherwise the eigenvalues come out scaled in nonsense units
    Same definition as fastlmm's SnpKernel (pysnptools/kernelreader/snpkernel.py _read), no need of rebuilding it from scrach
    """
    M = Z.shape[1]
    return (Z @ Z.T) / M
