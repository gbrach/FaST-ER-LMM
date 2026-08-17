"""tier-2 pairwise epistasis output writer.

per-pheno first_assoc table with the marginal SNP (i) and test SNP (j)
side-by-side, sorted by ascending PValue (already sorted by the heap
drain in scan.py). schema mirrors the single-SNP fastlmm-style table
from perms.py with i/j columns instead of one

bundle mode: optional run-level all_pairs.parquet that flattens every
per-pheno table into one streaming parquet, pheno + threshold + signif
baked in as columns, one row group per pheno. cuts the per-pheno-dir
fan-out to a single arrow::open_dataset() call downstream
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_parquet

from fasterlmm_lux.epi.scan import Tier2Result


def _marginal_arrays(result: Tier2Result, K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """marginal-i columns aligned to K. NaN-filled when scan_tier2 wasn't
    handed per_pheno_marginal_stats / per_pheno_marginal_threshold. used
    by both the per-pheno csv/parquet writer and the run-level bundle"""
    if result.marginal_pvalue_i is not None and result.marginal_pvalue_i.shape[0] == K:
        m_pv = np.asarray(result.marginal_pvalue_i, dtype=np.float64)
    else:
        m_pv = np.full(K, np.nan, dtype=np.float64)
    if result.marginal_beta_i is not None and result.marginal_beta_i.shape[0] == K:
        m_bt = np.asarray(result.marginal_beta_i, dtype=np.float64)
    else:
        m_bt = np.full(K, np.nan, dtype=np.float64)
    m_thr = float(result.marginal_threshold_pheno) if np.isfinite(result.marginal_threshold_pheno) else float("nan")
    m_thr_arr = np.full(K, m_thr, dtype=np.float64)
    return m_pv, m_bt, m_thr_arr


def _tier2_dataframe_from_arrays(
    result: Tier2Result,
    snp_id: list[str],
    chrom: np.ndarray,
    pos: np.ndarray) -> pd.DataFrame:
    """assembling the per-pheno tier-2 table from raw heap arrays.

    rank is 1-indexed in heap order (already ascending PValue from
    drain_sorted). empty result yields a header-only frame.

    marginal_{PValue,SnpWeight,threshold}_i columns are filled from
    Tier2Result.marginal_* (NaN-aligned when scan_tier2 didn't get the
    marginal stats — keeps old code paths working without changes)
    """
    K = result.snp_i_idx.shape[0]
    m_pv, m_bt, m_thr_arr = _marginal_arrays(result, K)
    if K == 0:
        return pd.DataFrame({
            "rank": np.empty((0,), dtype=np.int64),
            "SNP_i": pd.Series([], dtype=object),
            "Chr_i": np.empty((0,), dtype=chrom.dtype),
            "ChrPos_i": np.empty((0,), dtype=pos.dtype),
            "SNP_j": pd.Series([], dtype=object),
            "Chr_j": np.empty((0,), dtype=chrom.dtype),
            "ChrPos_j": np.empty((0,), dtype=pos.dtype),
            "PValue": np.empty((0,)),
            "SnpWeight": np.empty((0,)),
            "SnpWeightSE": np.empty((0,)),
            "marginal_PValue_i": np.empty((0,)),
            "marginal_SnpWeight_i": np.empty((0,)),
            "marginal_threshold_i": np.empty((0,)),
            "Pheno": pd.Series([], dtype=object),
        })
    i_idx = result.snp_i_idx
    j_idx = result.snp_j_idx
    return pd.DataFrame({
        "rank": np.arange(1, K + 1, dtype=np.int64),
        "SNP_i": [snp_id[i] for i in i_idx],
        "Chr_i": chrom[i_idx],
        "ChrPos_i": pos[i_idx],
        "SNP_j": [snp_id[j] for j in j_idx],
        "Chr_j": chrom[j_idx],
        "ChrPos_j": pos[j_idx],
        "PValue": result.pvalue,
        "SnpWeight": result.beta,
        "SnpWeightSE": result.se,
        "marginal_PValue_i": m_pv,
        "marginal_SnpWeight_i": m_bt,
        "marginal_threshold_i": m_thr_arr,
        "Pheno": result.pheno_name,
    })


def write_tier2_outputs(
    result: Tier2Result,
    snp_id: list[str],
    chrom: np.ndarray,
    pos: np.ndarray,
    outdir: str | Path,
    *,
    gzip_level: int = 1,
    output_format: str = "csv") -> None:
    """writing one pheno's tier-2 table to
    <outdir>/<pheno>/<pheno>.tier2.first_assoc.{txt.gz|parquet}

    output_format='csv' (default): pyarrow.csv.write_csv streamed through
    gzip at gzip_level. output_format='parquet' writes zstd-compressed
    parquet instead. empty results still create the dir + a header-only file
    """
    p = Path(outdir) / result.pheno_name
    p.mkdir(parents=True, exist_ok=True)
    df = _tier2_dataframe_from_arrays(result, snp_id, chrom, pos)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if output_format == "parquet":
        pa_parquet.write_table(
            table, p / f"{result.pheno_name}.tier2.first_assoc.parquet",
            compression="zstd")
    elif output_format == "csv":
        with gzip.open(p / f"{result.pheno_name}.tier2.first_assoc.txt.gz", "wb",
                        compresslevel=gzip_level) as f:
            pa_csv.write_csv(table, f, write_options=pa_csv.WriteOptions(
                include_header=True, delimiter="\t"))
    else:
        raise ValueError(f"output_format must be 'csv' or 'parquet', got {output_format!r}")


def write_threshold_and_signif(
    result: Tier2Result,
    snp_id: list[str],
    chrom: np.ndarray,
    pos: np.ndarray,
    outdir: str | Path) -> None:
    """writing perm-derived per-pheno significance bits next to first_assoc:

      <pheno>.threshold.txt — single-line "x\\n<value>\\n", same shape as
        perms.write_starlight_outputs (downstream R / awk consumers
        already grok this)
      <pheno>.signif_pairs.txt — TSV of pairs whose PValue < threshold,
        same column schema as the tier2 first_assoc table

    skipped silently when result.threshold is NaN (n_perm == 0 in the
    scan); keeps n_perm-off behaviour byte-identical to v1
    """
    if not np.isfinite(result.threshold):
        return
    p = Path(outdir) / result.pheno_name
    p.mkdir(parents=True, exist_ok=True)
    with open(p / f"{result.pheno_name}.threshold.txt", "w") as f:
        f.write(f"x\n{result.threshold}\n")
    df = _tier2_dataframe_from_arrays(result, snp_id, chrom, pos)
    sig = df[df["PValue"] < result.threshold]
    sig.to_csv(p / f"{result.pheno_name}.signif_pairs.txt",
               sep="\t", index=False)


def _write_one_for_pool(args):
    """top-level worker so ProcessPoolExecutor can pickle it.
    accepts either the legacy 7-tuple (result, snp_id, chrom, pos, outdir,
    gzip_level, output_format) or the 8-tuple with a trailing per_pheno_dirs
    bool. per_pheno_dirs=False skips the starlight-style per-folder tree,
    only valid alongside the run-level bundle (which the parent writes)"""
    if len(args) == 8:
        (result, snp_id, chrom, pos, outdir, gzip_level, output_format,
         per_pheno_dirs) = args
    else:
        result, snp_id, chrom, pos, outdir, gzip_level, output_format = args
        per_pheno_dirs = True
    if per_pheno_dirs:
        write_tier2_outputs(
            result, snp_id, chrom, pos, outdir,
            gzip_level=gzip_level, output_format=output_format)
        write_threshold_and_signif(result, snp_id, chrom, pos, outdir)


def _bundle_schema() -> pa.Schema:
    """parquet schema for the run-level bundle. one row per (pheno, anchor,
    test_snp) top-K entry. mirrors _tier2_dataframe_from_arrays + threshold
    and signif baked in. shard merges line up because the schema doesn't
    depend on cond / pheno-name layout"""
    return pa.schema([
        pa.field("Pheno", pa.string()),
        pa.field("rank", pa.int64()),
        pa.field("SNP_i", pa.string()),
        pa.field("Chr_i", pa.int32()),
        pa.field("ChrPos_i", pa.int64()),
        pa.field("SNP_j", pa.string()),
        pa.field("Chr_j", pa.int32()),
        pa.field("ChrPos_j", pa.int64()),
        pa.field("PValue", pa.float64()),
        pa.field("SnpWeight", pa.float64()),
        pa.field("SnpWeightSE", pa.float64()),
        pa.field("threshold", pa.float64()),
        pa.field("signif", pa.bool_()),
        pa.field("marginal_PValue_i", pa.float64()),
        pa.field("marginal_SnpWeight_i", pa.float64()),
        pa.field("marginal_threshold_i", pa.float64()),
    ])


class BundleWriter:
    """streaming pyarrow ParquetWriter wrapper. one row group per pheno;
    closes on context exit. opened by epistasis/cli.py when --bundle is on.
    threshold is NaN when n_perm == 0 (no perm calibration); signif is
    False everywhere in that case (NaN < x is always False)"""

    def __init__(self, path: Path, compression: str = "snappy"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = _bundle_schema()
        self._writer = pa_parquet.ParquetWriter(
            str(self.path), self.schema, compression=compression)
        self.n_phenos = 0

    def append_pheno(self, result: Tier2Result, snp_id: list[str],
                     chrom: np.ndarray, pos: np.ndarray) -> None:
        """append one pheno's top-K table as a row group, with Pheno +
        threshold + signif baked in. empty top-K (no marginals for this
        pheno) writes a zero-row table to keep the row-group-per-pheno
        bookkeeping consistent across shards"""
        K = result.snp_i_idx.shape[0]
        thr = float(result.threshold) if np.isfinite(result.threshold) else float("nan")
        m_pv, m_bt, m_thr_arr = _marginal_arrays(result, K)
        if K == 0:
            cols = {
                "Pheno": pa.array([], type=pa.string()),
                "rank": pa.array([], type=pa.int64()),
                "SNP_i": pa.array([], type=pa.string()),
                "Chr_i": pa.array([], type=pa.int32()),
                "ChrPos_i": pa.array([], type=pa.int64()),
                "SNP_j": pa.array([], type=pa.string()),
                "Chr_j": pa.array([], type=pa.int32()),
                "ChrPos_j": pa.array([], type=pa.int64()),
                "PValue": pa.array([], type=pa.float64()),
                "SnpWeight": pa.array([], type=pa.float64()),
                "SnpWeightSE": pa.array([], type=pa.float64()),
                "threshold": pa.array([], type=pa.float64()),
                "signif": pa.array([], type=pa.bool_()),
                "marginal_PValue_i": pa.array([], type=pa.float64()),
                "marginal_SnpWeight_i": pa.array([], type=pa.float64()),
                "marginal_threshold_i": pa.array([], type=pa.float64()),
            }
            table = pa.Table.from_pydict(cols, schema=self.schema)
            self._writer.write_table(table)
            self.n_phenos += 1
            return
        i_idx = result.snp_i_idx
        j_idx = result.snp_j_idx
        pvals = np.asarray(result.pvalue, dtype=np.float64)
        # signif: pair clears the perm-derived threshold. NaN < x is False so n_perm == 0 silently flips every row to False, which is what we want
        signif = np.asarray(pvals < thr, dtype=bool) if np.isfinite(thr) else np.zeros(K, dtype=bool)
        cols = {
            "Pheno": pa.array([result.pheno_name] * K, type=pa.string()),
            "rank": pa.array(np.arange(1, K + 1, dtype=np.int64), type=pa.int64()),
            "SNP_i": pa.array([snp_id[i] for i in i_idx], type=pa.string()),
            "Chr_i": pa.array(np.asarray(chrom[i_idx], dtype=np.int32), type=pa.int32()),
            "ChrPos_i": pa.array(np.asarray(pos[i_idx], dtype=np.int64), type=pa.int64()),
            "SNP_j": pa.array([snp_id[j] for j in j_idx], type=pa.string()),
            "Chr_j": pa.array(np.asarray(chrom[j_idx], dtype=np.int32), type=pa.int32()),
            "ChrPos_j": pa.array(np.asarray(pos[j_idx], dtype=np.int64), type=pa.int64()),
            "PValue": pa.array(pvals, type=pa.float64()),
            "SnpWeight": pa.array(np.asarray(result.beta, dtype=np.float64), type=pa.float64()),
            "SnpWeightSE": pa.array(np.asarray(result.se, dtype=np.float64), type=pa.float64()),
            "threshold": pa.array(np.full(K, thr, dtype=np.float64), type=pa.float64()),
            "signif": pa.array(signif, type=pa.bool_()),
            "marginal_PValue_i": pa.array(m_pv, type=pa.float64()),
            "marginal_SnpWeight_i": pa.array(m_bt, type=pa.float64()),
            "marginal_threshold_i": pa.array(m_thr_arr, type=pa.float64()),
        }
        table = pa.Table.from_pydict(cols, schema=self.schema)
        self._writer.write_table(table)
        self.n_phenos += 1

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "BundleWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
