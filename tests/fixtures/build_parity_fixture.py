"""
Builds the real-data parity fixture under tests/_data/parity/, gitignored on purpose
Pulls a small but real starlight slice (full glucose strain intersect, ~5000 real variants across all 16 chroms, 5 real nuclear phenos, full 91-column glucose covar) so the fastlmm-compat scan path has a chance of converging away from grid edges and parity tolerances stay tight

Re-run from repo root:
    python tests/fixtures/build_parity_fixture.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pysnptools.snpreader import Bed, SnpData

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tests" / "_data" / "parity"

SRC_PLINK = Path("/shared/home/gbrach/starlight/data/GWAS/matrix.SNPs.InDels.SVs.CNVs.plink")
SRC_PHEN_DIR = Path("/shared/home/gbrach/starlight/data/GWAS/phenotypes/glucose")
SRC_COVAR = Path("/shared/home/gbrach/starlight/data/GWAS/covariates_matrix_glucose.tab")

# 5 real nuclear glucose phenos, picked to be different from data/example/'s YAL001C / YBR001C / ...
# so this fixture doesn't accidentally collide with the shipped demo set
PHENOS = ["YAL002W", "YBR002C", "YGR002C", "YLR002C", "YPR002W"]
N_VARIANTS = 5000  # spread across the 16 chroms by uniform random selection, big enough that h^2 fits stay interior
SEED = 19930909

OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

# load source bed
bed = Bed(str(SRC_PLINK), count_A1=True)
src_bim = pd.read_csv(SRC_PLINK.with_suffix(".plink.bim"), sep="\t", header=None,
                      names=["chrom", "sid", "cm", "pos", "a1", "a2"], dtype={"a1": str, "a2": str})

# biallelic SNPs only (drop CNVs / indels / SVs by allele length + ACGT filter)
snp_mask = ((src_bim["a1"].str.len() == 1) & (src_bim["a2"].str.len() == 1)
            & src_bim["a1"].isin(list("ACGT")) & src_bim["a2"].isin(list("ACGT")))
snp_idx = np.where(snp_mask.values)[0]

# strain intersect: strains with all 5 phenos non-NaN AND a row in the covar file
strain_to_iididx = {iid[1]: i for i, iid in enumerate(bed.iid)}
keep = set(strain_to_iididx.keys())
for p in PHENOS:
    df = pd.read_csv(SRC_PHEN_DIR / p / f"{p}.norm.phen", sep="\t")
    df.columns = ["Strain", p]
    keep &= set(df.loc[df[p].notna(), "Strain"])
covar_df = pd.read_csv(SRC_COVAR, sep=r"\s+", header=None)
covar_strains = set(covar_df.iloc[:, 1].astype(str))
keep &= covar_strains
print(f"strain intersect: {len(keep)} strains kept after pheno + covar intersect")

iid_idx = np.sort([strain_to_iididx[s] for s in keep])
sid_idx = np.sort(rng.choice(snp_idx, N_VARIANTS, replace=False))

# read the chosen submatrix, zero-fill NaN dosages so downstream readers see clean values
sub = bed[iid_idx, sid_idx].read()
sub.val[np.isnan(sub.val)] = 0
strains = [iid[1] for iid in sub.iid]

snp_data = SnpData(iid=sub.iid, sid=sub.sid, val=sub.val, pos=sub.pos)
Bed.write(str(OUT / "parity"), snp_data, count_A1=True)
print(f"wrote {OUT}/parity.{{bed,bim,fam}}  shape: {sub.val.shape}")

# wide pheno tsv on the kept strains
phen_cols: dict[str, dict[str, float]] = {}
for p in PHENOS:
    df = pd.read_csv(SRC_PHEN_DIR / p / f"{p}.norm.phen", sep="\t")
    df.columns = ["Strain", p]
    phen_cols[p] = dict(zip(df["Strain"], df[p]))
pheno_rows = [{"Strain": s, **{p: phen_cols[p][s] for p in PHENOS}} for s in strains]
wide = pd.DataFrame(pheno_rows)
wide.to_csv(OUT / "parity_pheno.tsv", sep="\t", index=False, na_rep="NA")
print(f"wrote {OUT}/parity_pheno.tsv  shape: {wide.shape}  phenos: {PHENOS}")

# covar slice on the kept strains, prune to a maximally full-rank subset
# the raw 91-col starlight covar is rank-deficient on this strain set (zero-variance + collinear cols)
# fasterlmm + fastlmm both drop D=ncols(X) eigvecs but their rank-deficient handling under pinv diverges
# numerically, so the parity test stays clean if we hand it a well-conditioned design matrix to start with
covar_df.columns = ["FID", "IID"] + [f"c{i}" for i in range(covar_df.shape[1] - 2)]
covar_df["IID"] = covar_df["IID"].astype(str)
covar_sub = covar_df.set_index("IID").loc[strains].reset_index()
raw_cols = [c for c in covar_sub.columns if c not in ("FID", "IID")]
C = covar_sub[raw_cols].to_numpy(dtype=np.float64)
# greedy full-rank pruning against the intercept-augmented design: walk the cols in order, keep one only if it bumps the rank of [ones | kept]
kept_idx = []
basis = np.ones((C.shape[0], 1))
for j in range(C.shape[1]):
    trial = np.concatenate([basis, C[:, j:j+1]], axis=1)
    if np.linalg.matrix_rank(trial, tol=1e-8) > basis.shape[1]:
        basis = trial
        kept_idx.append(j)
print(f"covar prune: kept {len(kept_idx)}/{C.shape[1]} cols (intercept-augmented design rank = {basis.shape[1]})")
kept_cols = [raw_cols[j] for j in kept_idx]
covar_out = covar_sub[["FID", "IID"] + kept_cols]
covar_out.to_csv(OUT / "parity_covar.tab", sep=" ", index=False, header=False)
print(f"wrote {OUT}/parity_covar.tab  shape: {covar_out.shape}  ({len(kept_cols)} covars after rank prune)")
