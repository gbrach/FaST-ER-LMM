"""LUX file-based Parquet shard merging.

Pairwise bundles use one file per worker. This helper merges those files;
the core uses a separate directory-of-parts format.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq


def merge_bundle_shards(
    shard_paths: list[Path],
    final_path: Path,
    *,
    remove_shards: bool = True,
    compression: str = "snappy") -> Path:
    """stream-concatenate shard parquet files into one final bundle.

    row-group-streamed so memory stays bounded by the largest single
    row group (one gene / one pheno / one anchor) instead of the whole
    bundle. used by all three subcommands' main() after spawn join

    schema mismatch is a hard error: per-cond-name skew between shards
    or fp32 / fp64 promotions would silently corrupt the merged file
    """
    shard_paths = [Path(p) for p in shard_paths if Path(p).exists()]
    if not shard_paths:
        return Path(final_path)
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pq.ParquetFile(str(shard_paths[0])).schema_arrow
    writer = pq.ParquetWriter(str(final_path), schema, compression=compression)
    try:
        for sp in shard_paths:
            pf = pq.ParquetFile(str(sp))
            if pf.schema_arrow != schema:
                raise ValueError(
                    f"shard {sp} has a different schema than {shard_paths[0]}; "
                    "can't merge")
            for rg in range(pf.num_row_groups):
                writer.write_table(pf.read_row_group(rg))
    finally:
        writer.close()
    if remove_shards:
        for sp in shard_paths:
            try:
                sp.unlink()
            except FileNotFoundError:
                pass
    return final_path


def bundle_path_for_shard(outdir: Path, shard_rank: int, shard_n: int,
                           basename: str = "all_assoc") -> Path:
    """resolving where one shard's bundle lives. multi-shard runs land at
    <outdir>/<basename>.shard{r}.parquet so they can be merged later;
    single-shard runs go straight to <outdir>/<basename>.parquet"""
    if shard_n > 1:
        return Path(outdir) / f"{basename}.shard{shard_rank}.parquet"
    return Path(outdir) / f"{basename}.parquet"
