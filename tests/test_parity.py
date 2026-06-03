"""
parity contract test against the fastlmm golden reference
Runs fasterlmm gwas on the real-data parity fixture at tests/_data/parity/, compares the per-variant association table to the precomputed fastlmm output at tests/_data/parity/golden/
Single-K scan is parity-locked so any drift in PValue / SnpWeight / SnpWeightSE / EffectSize / Nullh2 across the 5 phenos x 2 variants (no_covar, with_covar) shows up here
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# 5 real starlight nuclear phenos in the parity fixture, see tests/fixtures/build_parity_fixture.py
PHENOS = ["YAL002W", "YBR002C", "YGR002C", "YLR002C", "YPR002W"]

# columns where fastlmm and fasterlmm should agree to float-noise on real-scale data
PARITY_COLS = ["PValue", "SnpWeight", "SnpWeightSE", "EffectSize", "Nullh2"]

# float tolerance for the parity check, fixture is N=542 real strains so h^2 fits stay interior
# atol=1e-6 covers near-zero SnpWeight values where rtol on |b| blows up even at sub-1e-7 absolute diffs
# both sides are float64 so a real porting drift would shift by orders of magnitude more, not by 1e-7
RTOL = 1e-5
ATOL = 1e-6


def _run_gwas(geno: Path, pheno: Path,
              outdir: Path, covar: Path | None) -> None:
    """driving the fasterlmm cli through the current python so the editable install is what runs"""
    # n-perm 1 because the cli trips on quantile() of an empty perm array, perms don't touch the per-snp table
    argv = [sys.executable, "-m", "fasterlmm", "gwas",
            "--geno", str(geno),
            "--pheno", str(pheno),
            "--outdir", str(outdir),
            "--device", "cpu",
            "--no-rint",
            "--n-perm", "1",
            "--bundle"]
    if covar is not None:
        argv.extend(["--covar", str(covar)])
    res = subprocess.run(argv, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(
            f"fasterlmm gwas exited {res.returncode}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")


@pytest.fixture(scope="module")
def no_covar_run(parity_geno: Path, parity_pheno: Path,
                 tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """one no-covar scan amortised across all per-pheno parity tests"""
    out = tmp_path_factory.mktemp("parity_no_covar")
    _run_gwas(parity_geno, parity_pheno, out, covar=None)
    return pd.read_parquet(str(out / "gwas_bundle.parquet"))


@pytest.fixture(scope="module")
def with_covar_run(parity_geno: Path, parity_pheno: Path,
                   parity_covar: Path,
                   tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """one with-covar scan amortised across all per-pheno parity tests, 91 starlight covar cols"""
    out = tmp_path_factory.mktemp("parity_with_covar")
    _run_gwas(parity_geno, parity_pheno, out, covar=parity_covar)
    return pd.read_parquet(str(out / "gwas_bundle.parquet"))


def _compare(gold: pd.DataFrame, ours: pd.DataFrame,
             pheno: str, variant: str) -> None:
    """sort both sides by SNP id, then check the per-variant numerics column by column with allclose"""
    assert len(gold) == len(ours), (
        f"{pheno} {variant}: row count mismatch gold={len(gold)} ours={len(ours)}")

    gold = gold.sort_values("SNP").reset_index(drop=True)
    ours = ours.sort_values("SNP").reset_index(drop=True)

    if not (gold["SNP"].values == ours["SNP"].values).all():
        n_diff = int((gold["SNP"].values != ours["SNP"].values).sum())
        raise AssertionError(
            f"{pheno} {variant}: SNP id sets differ in {n_diff} rows after sorting")

    failures = []
    for col in PARITY_COLS:
        if col not in gold.columns or col not in ours.columns:
            failures.append(f"  {col}: missing (gold={col in gold.columns}, ours={col in ours.columns})")
            continue
        a = np.asarray(gold[col].values, dtype=np.float64)
        b = np.asarray(ours[col].values, dtype=np.float64)
        if np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
            continue
        finite = np.isfinite(a) & np.isfinite(b)
        abs_diff = np.abs(a[finite] - b[finite])
        rel_diff = abs_diff / (np.abs(a[finite]) + 1e-300)
        i = int(np.argmax(abs_diff)) if abs_diff.size else -1
        snp_at = gold.loc[finite, "SNP"].values[i] if i >= 0 else "n/a"
        failures.append(
            f"  {col}: max_abs={abs_diff.max():.3e}  max_rel={rel_diff.max():.3e}  "
            f"worst@SNP={snp_at} gold={a[finite][i]:.6g} ours={b[finite][i]:.6g}")

    if failures:
        raise AssertionError(
            f"{pheno} {variant}: parity drift at rtol={RTOL}, atol={ATOL}\n"
            + "\n".join(failures))


@pytest.mark.parametrize("pheno", PHENOS)
def test_parity_no_covar(pheno: str, no_covar_run: pd.DataFrame,
                         parity_golden_dir: Path) -> None:
    """no-covar single-K scan, fasterlmm vs fastlmm golden for one of the 5 fixture phenos"""
    gold_path = parity_golden_dir / f"{pheno}__no_covar.parquet"
    if not gold_path.exists():
        pytest.skip(f"golden missing for {pheno} no_covar at {gold_path}")
    gold = pd.read_parquet(str(gold_path))
    ours = no_covar_run[no_covar_run["Pheno"] == pheno].copy()
    _compare(gold, ours, pheno, "no_covar")


@pytest.mark.parametrize("pheno", PHENOS)
def test_parity_with_covar(pheno: str, with_covar_run: pd.DataFrame,
                           parity_golden_dir: Path) -> None:
    """with-covar single-K scan, intercept + 91 starlight covar cols, fasterlmm vs fastlmm golden"""
    gold_path = parity_golden_dir / f"{pheno}__with_covar.parquet"
    if not gold_path.exists():
        pytest.skip(f"golden missing for {pheno} with_covar at {gold_path}")
    gold = pd.read_parquet(str(gold_path))
    ours = with_covar_run[with_covar_run["Pheno"] == pheno].copy()
    _compare(gold, ours, pheno, "with_covar")
