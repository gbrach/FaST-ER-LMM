"""
Bundle per-pheno output dirs into one parquet
The default cli layout is one subdir per pheno under outdir, each with gwas.tsv + perms.tsv + threshold.txt.  At thousands of phenos that's hard on the filesystem and R / python explorers crawl forever just listing files.
--bundle collapses the gwas.tsv files into a single parquet that pandas reads back in ~30s intead of the ~13min the tree reload takes
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def bundle_outdir(outdir: Path | str, out_path: Path | str | None = None) -> Path:
    """
    Scan outdir for per-pheno gwas.tsv files (one per subdir), concat into one parquet
    Adds a 'pheno' column from the parent dir name so downstream filterin is easy
    """
    outdir = Path(outdir)
    out_path = Path(out_path) if out_path else outdir / "gwas_bundle.parquet"
    rows = []
    for tsv in sorted(outdir.rglob("gwas.tsv")):
        df = pd.read_csv(tsv, sep="\t")
        df["pheno"] = tsv.parent.name
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"no gwas.tsv under {outdir}")
    big = pd.concat(rows, ignore_index=True)
    big.to_parquet(out_path, engine="pyarrow", compression="snappy")
    return out_path
