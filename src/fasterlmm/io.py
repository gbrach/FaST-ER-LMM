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
