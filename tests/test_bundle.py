"""
parquet bundle assembly (pure pyarrow, no torch)
covers fasterlmm.bundle -- the streaming BundleWriter, the gwas.tsv fallback route
bundle_outdir, and the multi-gpu shard merge merge_bundle_parts.  all of it lands
the same 14-column gwas schema plus a threshold + significant flag, so downstream
code never has to know how the scan ran.  cpu-only, portable, no fastlmm needed
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from fasterlmm.bundle import (
    BUNDLE_FILENAME,
    BUNDLE_PARTS_DIRNAME,
    BundleWriter,
    bundle_outdir,
    merge_bundle_parts,
)

from tests.conftest import GWAS_SCHEMA_COLS as GWAS_COLUMNS


def _make_table(pheno_name, n_rows=5, *, dict_pheno=False):
    """small pyarrow table carrying all 14 gwas columns, Pheno a string (or dict-encoded) column"""
    rng = np.random.default_rng(abs(hash(pheno_name)) % (2**32))
    pheno_col = pa.array([pheno_name] * n_rows, type=pa.string())
    if dict_pheno:
        pheno_col = pheno_col.dictionary_encode()
    cols = {
        "sid_index": pa.array(np.arange(n_rows), type=pa.int64()),
        "SNP": pa.array([f"{pheno_name}_v{i}" for i in range(n_rows)], type=pa.string()),
        "Chr": pa.array(np.full(n_rows, 1.0), type=pa.float64()),
        "GenDist": pa.array(np.zeros(n_rows), type=pa.float64()),
        "ChrPos": pa.array(np.arange(n_rows, dtype=float) * 100.0, type=pa.float64()),
        "PValue": pa.array(rng.uniform(0, 1, n_rows), type=pa.float64()),
        "SnpWeight": pa.array(rng.standard_normal(n_rows), type=pa.float64()),
        "SnpWeightSE": pa.array(rng.uniform(0.1, 1, n_rows), type=pa.float64()),
        "EffectSize": pa.array(rng.standard_normal(n_rows), type=pa.float64()),
        "SnpFractVarExpl": pa.array(rng.uniform(0, 1, n_rows), type=pa.float64()),
        "Mixing": pa.array(np.zeros(n_rows), type=pa.float64()),
        "Nullh2": pa.array(rng.uniform(0, 1, n_rows), type=pa.float64()),
        "Pheno": pheno_col,
        "PhenoCount": pa.array(np.full(n_rows, 150), type=pa.int64()),
    }
    return pa.table(cols)


def _write_gwas_tsv(pheno_dir: Path, pheno_name, n_rows=5, *, thresh=0.05):
    """lay a per-pheno gwas.tsv (14-col, tab-sep) plus its sibling threshold.txt"""
    pheno_dir.mkdir(parents=True, exist_ok=True)
    tbl = _make_table(pheno_name, n_rows=n_rows)
    df = tbl.to_pandas()
    df = df[GWAS_COLUMNS]  # keep the canonical column order
    df.to_csv(pheno_dir / "gwas.tsv", sep="\t", index=False)
    (pheno_dir / "threshold.txt").write_text(str(thresh))
    return df


# ---- the constants ---------------------------------------------------------


def test_bundle_constants():
    """the filename + parts-dir contract are the literal strings the writer / merge share"""
    assert BUNDLE_FILENAME == "gwas_bundle.parquet"
    assert BUNDLE_PARTS_DIRNAME == ".bundle_parts"


# ---- BundleWriter ----------------------------------------------------------


def test_writer_append_close_roundtrips(outdir):
    """two phenos appended stream into a part dir that pandas reads back with every row + a Pheno col"""
    path = outdir / BUNDLE_FILENAME
    w = BundleWriter(path)
    w.append(_make_table("phenoA", n_rows=5))
    w.append(_make_table("phenoB", n_rows=7))
    assert w.bytes_on_disk() > 0  # parts are on disk before close
    final = w.close()

    assert final == path
    assert final.is_dir()
    parts = sorted(final.glob("part*.parquet"))
    assert parts, "close should leave at least one part*.parquet behind"

    df = pd.read_parquet(final)
    assert len(df) == 12  # 5 + 7
    assert "Pheno" in df.columns
    assert set(df["Pheno"].unique()) == {"phenoA", "phenoB"}
    # the full 14-col schema survives the round trip
    for c in GWAS_COLUMNS:
        assert c in df.columns


def test_writer_decodes_dictionary_pheno(outdir):
    """a dictionary-encoded Pheno column comes back as a plain string so it matches the tsv route"""
    path = outdir / BUNDLE_FILENAME
    w = BundleWriter(path)
    w.append(_make_table("dictPheno", n_rows=4, dict_pheno=True))
    final = w.close()

    import pyarrow.parquet as pq
    schema = pq.read_schema(sorted(final.glob("part*.parquet"))[0])
    # the on-disk Pheno type is a flat string, not dictionary-encoded, so it matches the tsv route
    pheno_type = schema.field("Pheno").type
    assert pa.types.is_string(pheno_type)
    assert not pa.types.is_dictionary(pheno_type)

    df = pd.read_parquet(final)
    assert len(df) == 4
    assert df["Pheno"].iloc[0] == "dictPheno"
    # comes back as a flat string column, not a pandas categorical
    assert not isinstance(df["Pheno"].dtype, pd.CategoricalDtype)


def test_writer_bytes_grow_with_appends(outdir):
    """bytes_on_disk reflects the streaming parts and is non-negative throughout"""
    w = BundleWriter(outdir / BUNDLE_FILENAME)
    assert w.bytes_on_disk() == 0  # nothing written yet
    w.append(_make_table("p0", n_rows=10))
    after_one = w.bytes_on_disk()
    assert after_one > 0
    w.close()


# ---- bundle_outdir ---------------------------------------------------------


def test_bundle_outdir_builds_from_tsv_tree(outdir):
    """walks a per-pheno gwas.tsv tree, adds threshold + significant, parquet keeps all rows"""
    df_a = _write_gwas_tsv(outdir / "phenoA", "phenoA", n_rows=5, thresh=0.05)
    df_b = _write_gwas_tsv(outdir / "phenoB", "phenoB", n_rows=6, thresh=0.5)

    out = bundle_outdir(outdir)
    assert out == outdir / BUNDLE_FILENAME

    df = pd.read_parquet(out)
    assert len(df) == 11  # 5 + 6
    assert "threshold" in df.columns
    assert "significant" in df.columns
    # significant is PValue < threshold, per pheno's own threshold
    a = df[df["Pheno"] == "phenoA"].sort_values("sid_index").reset_index(drop=True)
    expect_a = (df_a["PValue"].values < 0.05)
    assert np.array_equal(a["significant"].values, expect_a)
    # threshold column carries the per-pheno value
    assert np.allclose(df[df["Pheno"] == "phenoA"]["threshold"].values, 0.05)
    assert np.allclose(df[df["Pheno"] == "phenoB"]["threshold"].values, 0.5)


def test_bundle_outdir_custom_outpath(outdir):
    """an explicit out_path is honoured rather than the default gwas_bundle.parquet"""
    _write_gwas_tsv(outdir / "p0", "p0", n_rows=4)
    custom = outdir / "my_bundle.parquet"
    out = bundle_outdir(outdir, out_path=custom)
    assert out == custom
    df = pd.read_parquet(custom)
    assert len(df) == 4


def test_bundle_outdir_empty_raises(outdir):
    """no gwas.tsv anywhere under outdir -> FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        bundle_outdir(outdir)


# ---- merge_bundle_parts ----------------------------------------------------


def _write_shard(parts_dir: Path, shard_idx, pheno_names):
    """commit one shard{i}.parquet dir of parts via a BundleWriter pointed at it"""
    shard_path = parts_dir / f"shard{shard_idx}.parquet"
    w = BundleWriter(shard_path)
    for name in pheno_names:
        w.append(_make_table(name, n_rows=5))
    return w.close()


def test_merge_bundle_parts_gathers_shards(outdir):
    """two shard dirs of parts merge into one gwas_bundle.parquet dir, .bundle_parts is gone"""
    parts_dir = outdir / BUNDLE_PARTS_DIRNAME
    parts_dir.mkdir(parents=True)
    _write_shard(parts_dir, 0, ["s0_phA", "s0_phB"])
    _write_shard(parts_dir, 1, ["s1_phC"])

    out = merge_bundle_parts(outdir)
    assert out == outdir / BUNDLE_FILENAME
    assert out.is_dir()
    assert not parts_dir.exists()  # .bundle_parts dropped on the way out

    df = pd.read_parquet(out)
    assert len(df) == 15  # (2 * 5) + (1 * 5)
    assert set(df["Pheno"].unique()) == {"s0_phA", "s0_phB", "s1_phC"}


def test_merge_bundle_parts_custom_outpath(outdir):
    """an explicit out_path lands the merged dataset where asked"""
    parts_dir = outdir / BUNDLE_PARTS_DIRNAME
    parts_dir.mkdir(parents=True)
    _write_shard(parts_dir, 0, ["only"])
    custom = outdir / "merged.parquet"
    out = merge_bundle_parts(outdir, out_path=custom)
    assert out == custom
    df = pd.read_parquet(custom)
    assert len(df) == 5
    assert not parts_dir.exists()


def test_merge_bundle_parts_no_shards_raises(outdir):
    """no .bundle_parts shard dirs -> FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        merge_bundle_parts(outdir)


def test_merge_bundle_parts_empty_parts_dir_raises(outdir):
    """an empty .bundle_parts (dir exists, no shard*.parquet) still raises FileNotFoundError"""
    (outdir / BUNDLE_PARTS_DIRNAME).mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        merge_bundle_parts(outdir)
