"""
Builds the fastlmm golden reference against the parity fixture at tests/_data/parity/
For each of the 5 phenos and each of (no_covar, with_covar) variants, runs fastlmm.association.single_snp
in both LOCO and single-K (leave_out_one_chrom False) and saves each returned DataFrame to
tests/_data/parity/golden/<pheno>__<variant>.parquet (LOCO) and <pheno>__<variant>__singlek.parquet

Re-run from repo root after tests/fixtures/build_parity_fixture.py:
    python tests/fixtures/build_parity_golden.py
Idempotent unless --force is passed
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

from fastlmm.association import single_snp

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "tests" / "_data" / "parity"
BED_PREFIX = DATA / "parity"
PHENO_TSV = DATA / "parity_pheno.tsv"
COVAR_TAB = DATA / "parity_covar.tab"
OUT = DATA / "golden"


def write_phen(strain_to_y: dict, path: Path) -> None:
    """plink .phen, whitespace, no header, FID IID y"""
    with open(path, "w") as f:
        for s, y in strain_to_y.items():
            f.write(f"{s} {s} {y}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not BED_PREFIX.with_suffix(".bed").exists():
        sys.exit(f"parity fixture missing at {DATA}, run tests/fixtures/build_parity_fixture.py first")

    OUT.mkdir(parents=True, exist_ok=True)
    pheno_df = pd.read_csv(PHENO_TSV, sep="\t")
    phenos = list(pheno_df.columns[1:])
    print(f"phenos: {phenos}")

    tmp = Path(tempfile.mkdtemp(prefix="parity_golden_", dir=str(REPO / "tests" / "_runs")))
    try:
        for p in phenos:
            sub = pheno_df[["Strain", p]].dropna()
            phen_path = tmp / f"{p}.phen"
            write_phen(dict(zip(sub["Strain"], sub[p])), phen_path)
            for variant, covar in [("no_covar", None), ("with_covar", COVAR_TAB)]:
                # both the LOCO golden (the default) and the single-K one (one kinship over every
                # variant), tagged __singlek, so the --no-loco path gets its own parity check
                for loco_tag, loco in [("", True), ("__singlek", False)]:
                    out_path = OUT / f"{p}__{variant}{loco_tag}.parquet"
                    if out_path.exists() and not args.force:
                        print(f"  skip {out_path.name}")
                        continue
                    df = single_snp(test_snps=str(BED_PREFIX),
                                    pheno=str(phen_path),
                                    covar=str(covar) if covar is not None else None,
                                    leave_out_one_chrom=loco,
                                    count_A1=True,
                                    output_file_name=None)
                    df.to_parquet(out_path)
                    print(f"  wrote {out_path.name}  rows={len(df)}  top-p={df.PValue.min():.3g}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\ngolden ref at {OUT}")
    print(f"files: {len(list(OUT.glob('*.parquet')))}  total: {sum(f.stat().st_size for f in OUT.glob('*.parquet')) / 1024:.0f} KB")


if __name__ == "__main__":
    sys.exit(main() or 0)
