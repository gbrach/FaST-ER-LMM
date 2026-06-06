"""
Shared pytest fixtures
Two data tiers: data/example/ ships in the repo (tiny, used by unit + cli tests) and tests/_data/parity/ is built locally from starlight (real-but-small, used by the parity layer)
Each test gets a fresh per-test outdir under tests/_runs/<test-name>/ so reruns start clean
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "data" / "example"
RUNS_ROOT = REPO / "tests" / "_runs"
PARITY_DATA = REPO / "tests" / "_data" / "parity"

# the 14-column fastlmm single_snp schema the per-pheno gwas.tsv carries, in the exact order cli._write_pheno lays it down
GWAS_SCHEMA_COLS = ["sid_index", "SNP", "Chr", "GenDist", "ChrPos", "PValue",
                    "SnpWeight", "SnpWeightSE", "EffectSize", "SnpFractVarExpl",
                    "Mixing", "Nullh2", "Pheno", "PhenoCount"]

# the 20 pheno columns in data/example/example_pheno.tsv, in their tsv order
PHENO_NAMES = ["YAL001C", "YAL002W", "YAL003W", "YAL005C",
               "YBR001C", "YBR002C", "YBR003W", "YBR004C",
               "YGR001C", "YGR002C", "YGR003W", "YGR004W",
               "YLR001C", "YLR002C", "YLR003C", "YLR004C",
               "YPR001W", "YPR002W", "YPR003C", "YPR004C"]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def example_geno() -> Path:
    return EXAMPLE / "example"


@pytest.fixture(scope="session")
def example_pheno() -> Path:
    return EXAMPLE / "example_pheno.tsv"


@pytest.fixture(scope="session")
def example_covar() -> Path:
    return EXAMPLE / "example_covar.tab"


@pytest.fixture(scope="session")
def parity_geno() -> Path:
    """real-data parity fixture, built once via tests/fixtures/build_parity_fixture.py, gitignored"""
    p = PARITY_DATA / "parity"
    if not p.with_suffix(".bed").exists():
        pytest.skip(f"parity fixture missing at {PARITY_DATA}, run tests/fixtures/build_parity_fixture.py")
    return p


@pytest.fixture(scope="session")
def parity_pheno() -> Path:
    p = PARITY_DATA / "parity_pheno.tsv"
    if not p.exists():
        pytest.skip(f"parity fixture missing at {PARITY_DATA}, run tests/fixtures/build_parity_fixture.py")
    return p


@pytest.fixture(scope="session")
def parity_covar() -> Path:
    p = PARITY_DATA / "parity_covar.tab"
    if not p.exists():
        pytest.skip(f"parity fixture missing at {PARITY_DATA}, run tests/fixtures/build_parity_fixture.py")
    return p


@pytest.fixture(scope="session")
def parity_golden_dir() -> Path:
    """fastlmm reference outputs against the parity fixture, built via tests/fixtures/build_parity_golden.py"""
    d = PARITY_DATA / "golden"
    if not d.exists() or not list(d.glob("*.parquet")):
        pytest.skip(f"parity golden missing at {d}, run tests/fixtures/build_parity_golden.py")
    return d


@pytest.fixture
def outdir(request) -> Path:
    """per-test fresh writable outdir under tests/_runs/<test-name>/, wiped on each call"""
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    name = request.node.name.replace("/", "_").replace("[", "_").replace("]", "")
    d = RUNS_ROOT / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d
