"""
cli guard / error-path cover for the gwas + extreme entries
the bad-flag combinations and the shard bounds-check are cheap to drive by subprocess since they bail at
parse time, before any scan runs.  cpu-only, no gpu, no fastlmm
"""
from __future__ import annotations

import subprocess
import sys


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "fasterlmm", *argv],
                          capture_output=True, text=True, check=False)


def test_gwas_no_per_pheno_dirs_without_bundle_errors(example_geno, example_pheno, outdir):
    """--no-per-pheno-dirs without --bundle would write nothing, so the parser rejects it (exit 2)"""
    r = _run("gwas", "--geno", str(example_geno), "--pheno", str(example_pheno),
             "--outdir", str(outdir), "--device", "cpu", "--no-multi-gpu",
             "--no-per-pheno-dirs")
    assert r.returncode != 0
    assert "--bundle" in r.stderr


def test_extreme_no_per_pheno_dirs_without_bundle_errors(example_geno, example_pheno, outdir):
    """same guard on the extreme entry"""
    r = _run("extreme", "--geno", str(example_geno), "--pheno", str(example_pheno),
             "--outdir", str(outdir), "--device", "cpu", "--no-multi-gpu",
             "--no-per-pheno-dirs")
    assert r.returncode != 0
    assert "--bundle" in r.stderr


def test_gwas_bad_shard_index_errors(example_geno, example_pheno, outdir):
    """a shard index outside [0, N) trips the bounds-check in _parse_shard before the scan starts"""
    r = _run("gwas", "--geno", str(example_geno), "--pheno", str(example_pheno),
             "--outdir", str(outdir), "--device", "cpu", "--shard", "5/2", "--n-perm", "1")
    assert r.returncode != 0
    blob = r.stderr + r.stdout
    assert "shard" in blob or "ValueError" in blob


def test_unknown_subcommand_exits_2():
    """the umbrella rejects an unknown subcommand with exit 2 and reprints the banner"""
    r = _run("bogus")
    assert r.returncode == 2
    assert "unknown subcommand" in r.stderr
    assert "gwas" in r.stderr  # banner reprinted
