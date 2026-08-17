"""reading per-pheno top-K marginal SNPs from a fasterlmm gwas results dir.

input layout (matches what `fasterlmm gwas` writes):
  <marginal_dir>/<pheno>/gwas.tsv (legacy first_assoc files also accepted)
  <marginal_dir>/<pheno>/threshold.txt           (optional sidecar)

each per-pheno file is sorted by PValue ascending already (writer pool
in perms.first_assoc_dataframe_from_arrays does .sort_values('PValue')).
so the top-K is just the head; no re-sort needed

returns:
  per_pheno_top_k: {pheno_name: [snp_id_1, ..., snp_id_K]}
  union: [snp_id, ...] in stable per-pheno-encounter order

also exposes read_marginal_stats / read_marginal_thresholds for plumbing
the anchor's marginal beta + p-value + per-pheno threshold straight into
the tier-2 bundle. graceful degrade: missing SnpWeight column or missing
threshold.txt yields NaN, old slim_marginals dirs (SNP+PValue only) still
work — tier-2 just writes NaN into the marginal_*_i columns
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import math
import pandas as pd
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_parquet

from fasterlmm_lux.epi.manifest import (
    MANIFEST_FILENAME, read_manifest, validate_against)


def _read_first_assoc(path: Path, *, with_beta: bool = False) -> pd.DataFrame:
    """auto-detecting csv.gz vs parquet. pulling SNP + PValue, and
    SnpWeight too when with_beta=True. graceful degrade: with_beta=True
    on a file missing the SnpWeight column returns the column filled with
    NaN so callers don't have to special-case the slim format"""
    cols = ["SNP", "PValue"] + (["SnpWeight"] if with_beta else [])
    if path.suffix == ".parquet":
        schema_names = set(pa_parquet.read_schema(path).names)
        ask = [c for c in cols if c in schema_names]
        tbl = pa_parquet.read_table(path, columns=ask)
        df = tbl.to_pandas()
    elif path.suffixes[-2:] == [".txt", ".gz"]:
        with gzip.open(path, "rb") as f:
            tbl = pa_csv.read_csv(
                f,
                read_options=pa_csv.ReadOptions(use_threads=False),
                parse_options=pa_csv.ParseOptions(delimiter="\t"),
                convert_options=pa_csv.ConvertOptions(include_columns=cols))
        df = tbl.to_pandas()
    elif path.suffix in (".tsv", ".txt"):
        # uncompressed tsv — the clean core's per-pheno-dirs layout writes
        # <pheno>/gwas.tsv (bare 14-col fastlmm schema, has SNP/PValue/SnpWeight)
        tbl = pa_csv.read_csv(
            path,
            read_options=pa_csv.ReadOptions(use_threads=False),
            parse_options=pa_csv.ParseOptions(delimiter="\t"),
            convert_options=pa_csv.ConvertOptions(include_columns=cols))
        df = tbl.to_pandas()
    else:
        raise ValueError(f"unrecognised first_assoc format: {path}")
    if with_beta and "SnpWeight" not in df.columns:
        df["SnpWeight"] = float("nan")
    return df


def _find_first_assoc(pheno_dir: Path, pheno: str) -> Path:
    # prefer the lux/dev-native <pheno>.first_assoc.* (also what old associations
    # dirs use), then fall back to the clean core's per-pheno-dirs layout
    # (<pheno>/gwas.tsv from `fasterlmm gwas`). order keeps existing dirs reading
    # exactly as before — the fallback only fires when first_assoc.* is absent.
    for ext in ("first_assoc.parquet", "first_assoc.txt.gz"):
        p = pheno_dir / f"{pheno}.{ext}"
        if p.exists():
            return p
    core_native = pheno_dir / "gwas.tsv"
    if core_native.exists():
        return core_native
    raise FileNotFoundError(
        f"no first_assoc file in {pheno_dir} for pheno {pheno!r} "
        f"(looked for {pheno}.first_assoc.parquet/.txt.gz and core-native gwas.tsv)")


def read_marginals(
    marginal_dir: str | Path,
    pheno_names: list[str],
    top_k: int) -> dict[str, list[str]]:
    """per pheno, returning the top-K SNP IDs sorted by PValue ascending.
    raises FileNotFoundError if a pheno's first_assoc file is missing"""
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    base = Path(marginal_dir)
    out: dict[str, list[str]] = {}
    for pheno in pheno_names:
        pheno_dir = base / pheno
        path = _find_first_assoc(pheno_dir, pheno)
        df = _read_first_assoc(path)
        # files are pre-sorted by PValue ascending. defending against the
        # rare downstream re-write that loses the sort
        df = df.sort_values("PValue", kind="stable")
        out[pheno] = df["SNP"].head(top_k).tolist()
    return out


def read_marginal_stats(
    marginal_dir: str | Path,
    pheno_names: list[str],
    snp_ids: list[str] | None = None) -> dict[str, dict[str, tuple[float, float]]]:
    """per-pheno {snp_id -> (PValue, SnpWeight)} lookup for the marginal
    anchors. used by scan_tier2 to backfill marginal_PValue_i and
    marginal_SnpWeight_i into each tier-2 row.

    snp_ids (optional) — restrict the dict to anchors actually consumed
    by tier-2. saves memory when the slim has top-2000 but we anchor on
    top-100. missing files raise FileNotFoundError; missing SnpWeight
    silently fills NaN (slim_marginals.py pre-2026-05-13 didn't keep it)
    """
    base = Path(marginal_dir)
    want = set(snp_ids) if snp_ids is not None else None
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for pheno in pheno_names:
        pheno_dir = base / pheno
        path = _find_first_assoc(pheno_dir, pheno)
        df = _read_first_assoc(path, with_beta=True)
        if want is not None:
            df = df[df["SNP"].isin(want)]
        out[pheno] = {
            str(s): (float(p), float(b))
            for s, p, b in zip(df["SNP"], df["PValue"], df["SnpWeight"])
        }
    return out


def read_marginal_thresholds(
    marginal_dir: str | Path,
    pheno_names: list[str]) -> dict[str, float]:
    """per-pheno scalar threshold read from <pheno>/<pheno>.threshold.txt.

    file shape mirrors what perms.write_starlight_outputs writes: two
    lines, first "x", second the float threshold. missing sidecar yields
    NaN for that pheno — old slim_marginals dirs predating the sidecar
    still load, tier-2 just writes NaN into marginal_threshold_i
    """
    base = Path(marginal_dir)
    out: dict[str, float] = {}
    for pheno in pheno_names:
        pdir = base / pheno
        # lux/dev-native <pheno>.threshold.txt, else the clean core's threshold.txt
        p = next((c for c in (pdir / f"{pheno}.threshold.txt", pdir / "threshold.txt")
                  if c.exists()), None)
        if p is None:
            out[pheno] = float("nan")
            continue
        try:
            lines = p.read_text().splitlines()
            # accept either single-line scalar or two-line "x\n<val>"
            val = lines[1] if len(lines) >= 2 and lines[0].strip() == "x" else lines[0]
            out[pheno] = float(val)
        except (ValueError, IndexError):
            out[pheno] = float("nan")
    return out


def union_marginal_snps(per_pheno: dict[str, list[str]]) -> list[str]:
    """union of all per-pheno top-K lists. order: first-seen across
    phenos, stable for reproducibility"""
    seen: set[str] = set()
    union: list[str] = []
    for snps in per_pheno.values():
        for sid in snps:
            if sid not in seen:
                seen.add(sid)
                union.append(sid)
    return union


def check_marginal_manifest(
    marginal_dir: str | Path,
    *,
    geno_prefix: Path, n_strains: int,
    n_variants: int, maf_floor: float,
    rint: bool, covar_path: Path | None,
    rank_correct: bool) -> None:
    """compare the manifest (if any) against the current run's params.
    mismatches print one stderr warning per offending field; missing
    manifest prints one stderr warning and returns. never raises, the
    user might be intentionally re-using a marginal-dir under different
    settings (different MAF, RINT toggle for sensitivity)"""
    base = Path(marginal_dir)
    manifest_path = base / MANIFEST_FILENAME
    try:
        manifest = read_manifest(base)
    except FileNotFoundError:
        print(f"warning: no manifest at {manifest_path}; cannot validate "
              f"marginal-dir against current params",
              file=sys.stderr)
        return
    msgs = validate_against(
        manifest,
        geno_prefix=geno_prefix, n_strains=n_strains,
        n_variants=n_variants, maf_floor=maf_floor,
        rint=rint, covar_path=covar_path,
        rank_correct=rank_correct)
    for m in msgs:
        print(f"warning: marginal-dir manifest mismatch: {m}", file=sys.stderr)


def map_snp_ids_to_indices(
    snp_ids: list[str],
    snp_id_universe: list[str]) -> list[int]:
    """resolving SNP IDs to their column index in the genotype matrix.
    raises KeyError if any ID is not in the universe"""
    idx_by_id = {sid: i for i, sid in enumerate(snp_id_universe)}
    out: list[int] = []
    for sid in snp_ids:
        if sid not in idx_by_id:
            raise KeyError(f"marginal SNP {sid!r} not in genotype matrix")
        out.append(idx_by_id[sid])
    return out
