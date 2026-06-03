"""
parity contract for the single-K (--no-loco) path against the fastlmm golden
Sibling of test_parity.py but with one kinship built over every variant instead of leave-one-chromosome-out.
Runs fasterlmm gwas --no-loco on the parity fixture and compares the per-variant table to the fastlmm
single_snp(leave_out_one_chrom=False) golden at tests/_data/parity/golden/<pheno>__<variant>__singlek.parquet
Single-K is parity-locked too (single_k_scan_compat), so any drift in the checked columns surfaces here.
Skips cleanly when the parity fixture / golden are absent (a fresh clone), so it never blocks CI
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PHENOS = ["YAL002W", "YBR002C", "YGR002C", "YLR002C", "YPR002W"]
PARITY_COLS = ["PValue", "SnpWeight", "SnpWeightSE", "EffectSize", "Nullh2"]
RTOL = 1e-5
ATOL = 1e-6


def _run_gwas(geno: Path, pheno: Path,
              outdir: Path, covar: Path | None) -> None:
    """drive the cli through the current python, --no-loco for the single-K kinship"""
    argv = [sys.executable, "-m", "fasterlmm", "gwas",
            "--geno", str(geno),
            "--pheno", str(pheno),
            "--outdir", str(outdir),
            "--device", "cpu",
            "--no-rint",
            "--no-loco",
            "--n-perm", "1",
            "--bundle"]
    if covar is not None:
        argv.extend(["--covar", str(covar)])
    res = subprocess.run(argv, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(
            f"fasterlmm gwas --no-loco exited {res.returncode}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")


@pytest.fixture(scope="module")
def no_covar_singlek(parity_geno: Path, parity_pheno: Path,
                     tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """one no-covar single-K scan amortised across the per-pheno parity tests"""
    out = tmp_path_factory.mktemp("parity_singlek_no_covar")
    _run_gwas(parity_geno, parity_pheno, out, covar=None)
    return pd.read_parquet(str(out / "gwas_bundle.parquet"))


@pytest.fixture(scope="module")
def with_covar_singlek(parity_geno: Path, parity_pheno: Path,
                       parity_covar: Path,
                       tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """one with-covar single-K scan amortised across the per-pheno parity tests"""
    out = tmp_path_factory.mktemp("parity_singlek_with_covar")
    _run_gwas(parity_geno, parity_pheno, out, covar=parity_covar)
    return pd.read_parquet(str(out / "gwas_bundle.parquet"))


def _compare(gold: pd.DataFrame, ours: pd.DataFrame,
             pheno: str, variant: str) -> None:
    """sort both sides by SNP id, then check the per-variant numerics column by column"""
    assert len(gold) == len(ours), (
        f"{pheno} {variant}: row count mismatch gold={len(gold)} ours={len(ours)}")
    gold = gold.sort_values("SNP").reset_index(drop=True)
    ours = ours.sort_values("SNP").reset_index(drop=True)
    if not (gold["SNP"].values == ours["SNP"].values).all():
        n_diff = int((gold["SNP"].values != ours["SNP"].values).sum())
        raise AssertionError(f"{pheno} {variant}: SNP id sets differ in {n_diff} rows after sorting")

    failures = []
    for col in PARITY_COLS:
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
            f"{pheno} {variant}: single-K parity drift at rtol={RTOL}, atol={ATOL}\n" + "\n".join(failures))


@pytest.mark.parametrize("pheno", PHENOS)
def test_parity_single_k_no_covar(pheno: str, no_covar_singlek: pd.DataFrame,
                                  parity_golden_dir: Path) -> None:
    """no-covar single-K scan, fasterlmm --no-loco vs fastlmm single_snp(leave_out_one_chrom=False)"""
    gold_path = parity_golden_dir / f"{pheno}__no_covar__singlek.parquet"
    if not gold_path.exists():
        pytest.skip(f"single-K golden missing for {pheno} no_covar at {gold_path}")
    gold = pd.read_parquet(str(gold_path))
    ours = no_covar_singlek[no_covar_singlek["Pheno"] == pheno].copy()
    _compare(gold, ours, pheno, "no_covar singlek")


@pytest.mark.parametrize("pheno", PHENOS)
def test_parity_single_k_with_covar(pheno: str, with_covar_singlek: pd.DataFrame,
                                    parity_golden_dir: Path) -> None:
    """with-covar single-K scan, intercept + starlight covars, fasterlmm --no-loco vs fastlmm golden"""
    gold_path = parity_golden_dir / f"{pheno}__with_covar__singlek.parquet"
    if not gold_path.exists():
        pytest.skip(f"single-K golden missing for {pheno} with_covar at {gold_path}")
    gold = pd.read_parquet(str(gold_path))
    ours = with_covar_singlek[with_covar_singlek["Pheno"] == pheno].copy()
    _compare(gold, ours, pheno, "with_covar singlek")