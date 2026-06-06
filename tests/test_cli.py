"""
CLI integration tests
Drives the fasterlmm umbrella by subprocess on the shipped data/example/ subset and asserts every advertised flag lands its outputs in the right place with the right shape.  Numerical exactness is parity's beat, here we just make sure files exist, columns line up, and a handful of structural invariants hold (LOCO varies Nullh2 by chrom, single-K doesnt; RINT shifts the p-values vs no-RINT, etc)
Every test runs cpu-only so the suite is portable, and uses --n-perm 10 except where the perm output itself is the thing being checked
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import GWAS_SCHEMA_COLS as GWAS_TSV_COLS, PHENO_NAMES


def _run(*argv: str, check: bool = True) -> subprocess.CompletedProcess:
    """thin wrapper around subprocess.run that surfaces stderr on a failed exit so a broken cli is easy to read in the pytest report"""
    r = subprocess.run(["fasterlmm", *argv], capture_output=True, text=True, check=False)
    if check:
        assert r.returncode == 0, (
            f"fasterlmm {' '.join(argv)} exited {r.returncode}\n"
            f"--- stderr ---\n{r.stderr}\n--- stdout ---\n{r.stdout}")
    return r


def _basic_gwas_args(geno: Path, pheno: Path, outdir: Path, n_perm: int = 10) -> list[str]:
    """canonical cpu-only invocation, every test that runs gwas starts from this list"""
    return ["gwas",
            "--geno", str(geno),
            "--pheno", str(pheno),
            "--outdir", str(outdir),
            "--device", "cpu",
            "--no-multi-gpu",
            "--n-perm", str(n_perm)]


def test_cli_help() -> None:
    """umbrella --help mentions every advertised subcommand"""
    r = _run("--help")
    out = r.stdout
    assert "gwas" in out
    assert "watch" in out
    assert "concat" in out


def test_gwas_dry_run(example_geno, example_pheno, outdir) -> None:
    """--dry-run prints the planned numbers and exits clean without writing any per-pheno output"""
    args = _basic_gwas_args(example_geno, example_pheno, outdir) + ["--dry-run"]
    r = _run(*args)
    # the dry-run line carries N / M / P / n_perm so the user can sanity-check the slice before the scan
    assert "[dry-run]" in r.stdout or "[dry-run]" in r.stderr
    blob = r.stdout + r.stderr
    assert "N=150" in blob
    assert "M=1500" in blob
    assert "P=20" in blob
    # the cli mkdirs outdir before reaching dry-run, so the dir itself exists -- what must NOT exist
    # is any per-pheno subdir, those only get built once a real scan runs
    pheno_dirs = [d for d in outdir.iterdir() if d.is_dir()]
    assert pheno_dirs == [], f"dry-run leaked per-pheno dirs: {pheno_dirs}"


def test_gwas_basic_cpu_run(example_geno, example_pheno, outdir) -> None:
    """minimal scan -- every pheno gets its dir + gwas.tsv with the 14-column fastlmm schema, p-values in [0, 1]"""
    _run(*_basic_gwas_args(example_geno, example_pheno, outdir))
    for pheno in PHENO_NAMES:
        tsv = outdir / pheno / "gwas.tsv"
        assert tsv.exists(), f"missing gwas.tsv for {pheno}"
    # spot-checking one pheno's table: schema is the documented 14 columns, 1500 variants, p in [0, 1]
    df = pd.read_csv(outdir / PHENO_NAMES[0] / "gwas.tsv", sep="\t")
    assert list(df.columns) == GWAS_TSV_COLS
    assert len(df) == 1500
    assert df.PValue.between(0.0, 1.0).all()
    assert (df.Pheno == PHENO_NAMES[0]).all()


def test_gwas_threshold_perm_files(example_geno, example_pheno, outdir) -> None:
    """--n-perm 20 lands a perms.tsv (20 rows) + threshold.txt (one float in [0, 1]) per pheno"""
    args = _basic_gwas_args(example_geno, example_pheno, outdir, n_perm=20)
    _run(*args)
    for pheno in PHENO_NAMES:
        sub = outdir / pheno
        perms = sub / "perms.tsv"
        thresh = sub / "threshold.txt"
        assert perms.exists() and thresh.exists()
        # perms.tsv carries one min-p row per perm (plus the header) so a 20-perm scan = 21 lines
        perm_df = pd.read_csv(perms, sep="\t")
        assert len(perm_df) == 20
        # threshold.txt is one scientific-notation float, parseable, sitting in [0, 1]
        t = float(thresh.read_text().strip())
        assert 0.0 <= t <= 1.0


def test_gwas_with_covar(example_geno, example_pheno, example_covar, outdir) -> None:
    """passing --covar still lands every per-pheno output, exactness lives in test_parity"""
    args = _basic_gwas_args(example_geno, example_pheno, outdir) + ["--covar", str(example_covar)]
    _run(*args)
    for pheno in PHENO_NAMES:
        assert (outdir / pheno / "gwas.tsv").exists()
        assert (outdir / pheno / "threshold.txt").exists()


def test_gwas_bundle(example_geno, example_pheno, outdir) -> None:
    """--bundle drops a gwas_bundle.parquet dir holding every pheno's 16-column table"""
    args = _basic_gwas_args(example_geno, example_pheno, outdir) + ["--bundle"]
    _run(*args)
    bundle = outdir / "gwas_bundle.parquet"
    assert bundle.is_dir(), "bundle should be a directory of part*.parquet files"
    parts = list(bundle.glob("part*.parquet"))
    assert parts, "bundle has no parts"
    df = pd.read_parquet(str(bundle))
    # 20 phenos x 1500 variants = 30000 rows, 16 columns (the 14 from gwas.tsv plus threshold + significant)
    assert len(df) == 20 * 1500
    assert "threshold" in df.columns
    assert "significant" in df.columns
    assert set(df["Pheno"].unique()) == set(PHENO_NAMES)


def test_gwas_no_per_pheno_dirs(example_geno, example_pheno, outdir) -> None:
    """--bundle --no-per-pheno-dirs keeps only the bundle, the per-pheno tree is skipped"""
    args = _basic_gwas_args(example_geno, example_pheno, outdir) + [
        "--bundle", "--no-per-pheno-dirs"]
    _run(*args)
    assert (outdir / "gwas_bundle.parquet").is_dir()
    # none of the per-pheno dirs should have been written
    leaked = [p for p in PHENO_NAMES if (outdir / p).exists()]
    assert leaked == [], f"per-pheno dirs leaked with --no-per-pheno-dirs: {leaked}"


def test_gwas_pheno_idx(example_geno, example_pheno, outdir) -> None:
    """--pheno-idx 3 scans only the 4th pheno column, no other pheno dirs land"""
    args = _basic_gwas_args(example_geno, example_pheno, outdir) + ["--pheno-idx", "3"]
    _run(*args)
    # column index 3 in the pheno tsv (after the Strain column) is PHENO_NAMES[3]
    target = PHENO_NAMES[3]
    pheno_dirs = sorted(d.name for d in outdir.iterdir() if d.is_dir())
    assert pheno_dirs == [target], f"expected just {target}, got {pheno_dirs}"
    assert (outdir / target / "gwas.tsv").exists()


def test_gwas_pheno_range(example_geno, example_pheno, outdir) -> None:
    """--pheno-start 5 --pheno-end 10 covers exactly the 5 columns in [5, 10)"""
    args = _basic_gwas_args(example_geno, example_pheno, outdir) + [
        "--pheno-start", "5", "--pheno-end", "10"]
    _run(*args)
    expected = set(PHENO_NAMES[5:10])
    pheno_dirs = {d.name for d in outdir.iterdir() if d.is_dir()}
    assert pheno_dirs == expected, f"expected {expected}, got {pheno_dirs}"


def test_gwas_shard_then_concat(example_geno, example_pheno, outdir) -> None:
    """two --shard runs stage parts under .bundle_parts/, then fasterlmm concat gathers them into one bundle"""
    base = _basic_gwas_args(example_geno, example_pheno, outdir) + ["--bundle"]
    _run(*(base + ["--shard", "0/2"]))
    _run(*(base + ["--shard", "1/2"]))
    # before the concat each shard owns its own shard{i}.parquet dir under .bundle_parts/
    parts_dir = outdir / ".bundle_parts"
    assert parts_dir.is_dir()
    shard_dirs = sorted(d.name for d in parts_dir.iterdir() if d.is_dir())
    assert shard_dirs == ["shard0.parquet", "shard1.parquet"]
    # concat flattens them into the final bundle, single command, no per-pheno reread
    _run("concat", str(outdir))
    bundle = outdir / "gwas_bundle.parquet"
    assert bundle.is_dir()
    df = pd.read_parquet(str(bundle))
    assert len(df) == 20 * 1500
    assert set(df["Pheno"].unique()) == set(PHENO_NAMES)


def test_gwas_no_rint(example_geno, example_pheno, outdir, tmp_path) -> None:
    """--no-rint changes the pheno distribution, so the first-row PValue shifts relative to a --rint run"""
    # rint side gets its own outdir under tmp_path so the fixture's outdir is left to no-rint
    rint_dir = tmp_path / "rint_run"
    _run(*(_basic_gwas_args(example_geno, example_pheno, rint_dir) + ["--rint"]))
    _run(*(_basic_gwas_args(example_geno, example_pheno, outdir) + ["--no-rint"]))
    pheno = PHENO_NAMES[0]
    p_rint = pd.read_csv(rint_dir / pheno / "gwas.tsv", sep="\t").PValue.iloc[0]
    p_no = pd.read_csv(outdir / pheno / "gwas.tsv", sep="\t").PValue.iloc[0]
    # rint reshapes the pheno before the scan, so the smallest p-value cant land identical
    assert p_rint != pytest.approx(p_no), (
        f"RINT vs no-RINT first-row PValue match exactly ({p_rint}), one of the two flags didnt take")


def test_gwas_no_loco(example_geno, example_pheno, outdir) -> None:
    """--no-loco runs single-K, every per-pheno output lands and Nullh2 stays constant across the variant table"""
    args = _basic_gwas_args(example_geno, example_pheno, outdir) + ["--no-loco"]
    _run(*args)
    for pheno in PHENO_NAMES:
        assert (outdir / pheno / "gwas.tsv").exists()
    # the structural invariant: under single-K every variant in one pheno shares the same Nullh2 (one
    # null fit), under loco each chromosome refits so Nullh2 varies.  spot-checking one pheno is plenty
    df = pd.read_csv(outdir / PHENO_NAMES[0] / "gwas.tsv", sep="\t")
    assert df.Nullh2.nunique() == 1, (
        f"single-K should leave Nullh2 constant within a pheno, got {df.Nullh2.nunique()} distinct values")


def test_watch_help() -> None:
    """fasterlmm watch --help returns clean -- the tui itself blocks so the actual run isnt tested here"""
    r = _run("watch", "--help")
    assert r.returncode == 0
