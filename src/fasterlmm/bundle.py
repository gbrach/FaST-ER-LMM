"""
Assemble a gwas scan into one parquet bundle
Per pheno the cli builds a 16-column table -- the 14 fastlmm single_snp columns plus a threshold and a significant flag.  BundleWriter streams those straight into the bundle while the scan is still running, so the bundle lands alongside the gpu work instead of in a slow read-everything-back pass afterwards
The bundle is a directory of parquet parts, one part per writer-pool thread, so the per-pheno compression runs fully parallel with no lock in the way.  A directory of parts is itself a valid dataset, pandas / pyarrow / duckdb all read it back exactly like one file.  A multi-gpu run drops a shard{i}.parquet directory of parts per worker under .bundle_parts/, merge_bundle_parts flattens every part into the gwas_bundle.parquet dataset.  bundle_outdir is the fallback route, it rebuilds the bundle from a tree of per-pheno gwas.tsv files when only that tree survived
every route lands the same 16-column bundle, so downstream code never has to care how the scan was run
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

# the parts dir and the bundle filename are the contract between the writer in cli.py, the merge
# here and the stale-cleanup -- spelling them once keeps those three from ever drifting apart
BUNDLE_PARTS_DIRNAME = ".bundle_parts"
BUNDLE_FILENAME = "gwas_bundle.parquet"
_PARQUET_COMPRESSION = "snappy"

# the per-pheno gwas.tsv is plain text, so reading it back needs the column types spelled out --
# csv inference on its own turns the all-zero Mixing column and the integer-valued Chr / ChrPos
# into ints, wich then wouldnt line up with the floats the streamed route keeps.  this map mirrors
# the table cli._write_pheno builds, the two have to move together
_GWAS_TSV_TYPES = {
    "sid_index": pa.int64(),
    "SNP": pa.string(),
    "Chr": pa.float64(),
    "GenDist": pa.float64(),
    "ChrPos": pa.float64(),
    "PValue": pa.float64(),
    "SnpWeight": pa.float64(),
    "SnpWeightSE": pa.float64(),
    "EffectSize": pa.float64(),
    "SnpFractVarExpl": pa.float64(),
    "Mixing": pa.float64(),
    "Nullh2": pa.float64(),
    "Pheno": pa.string(),
    "PhenoCount": pa.int64(),
}


class BundleWriter:
    """
    Per-process bundle writer, one parquet part per writer-pool thread
    The bundle is a directory of parquet parts, so handing each pool thread its own part lets every thread compress and write its phenos fully in parallel -- no lock around write_table, wich a single shared ParquetWriter would otherwise force since pyarrow's writer is not thread-safe.  Each thread's writer is built lazily on its first append, the only shared state left is the part counter
    Each finished pheno is appended as its own row group the moment its scan + write lands, so when the gpu work ends the bundle is already on disk -- no separate read-everything-back pass.  Parts stream into a .tmp directory wich is renamed into place on close, so a crashed run never leaves a half-built bundle behind.  Pheno arrives dictionary-encoded for a cheap table build and gets decoded back to plain string here so the bundle matches the bundle_outdir route
    """

    def __init__(self, path: Path | str) -> None:
        self._dir = Path(path)
        self._tmp = self._dir.with_name(self._dir.name + ".tmp")
        # a stale .tmp -- a staging dir from an interrupted run or an orphaned single-file .tmp from
        # an older crashed writer -- would block the fresh staging dir, so clear whichever it is
        if self._tmp.is_dir():
            shutil.rmtree(self._tmp)
        elif self._tmp.exists():
            self._tmp.unlink()
        self._tmp.mkdir(parents=True)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._writers: list[pq.ParquetWriter] = []
        self._next_part = 0

    def append(self, table: pa.Table) -> None:
        """Append one pheno's table as a row group on the calling thread's own part.  Lock-free apart from claiming a part the first time a given thread lands here"""
        # the dictionary cast is pure and thread-local, no sharing
        pheno_i = table.schema.get_field_index("Pheno")
        table = table.set_column(pheno_i, "Pheno", table.column("Pheno").cast(pa.string()))
        writer = getattr(self._local, "writer", None)
        if writer is None:
            # first append on this thread -- open its own part under the lock.  the lock guards only
            # this one-off setup, the counter and the writers list, all tiny next to write_table
            with self._lock:
                writer = pq.ParquetWriter(self._tmp / f"part{self._next_part}.parquet",
                                          table.schema, compression=_PARQUET_COMPRESSION)
                self._next_part += 1
                self._writers.append(writer)
            self._local.writer = writer
        # the compression + write itself runs free of the lock, so the threads genuinely overlap
        writer.write_table(table)

    def bytes_on_disk(self) -> int:
        """Total size of the streaming parts right now, feeds the write-throughput readout while the bundle drains"""
        total = 0
        for part in self._tmp.glob("*.parquet"):
            try:
                total += part.stat().st_size
            except OSError:
                pass
        return total

    def close(self) -> Path:
        """Close every per-thread writer and rename the .tmp directory into place.  Returns the final bundle path"""
        for writer in self._writers:
            writer.close()
        self._writers = []
        # a stale bundle from an earlier run, file or directory, would block the rename
        if self._dir.is_dir():
            shutil.rmtree(self._dir)
        elif self._dir.exists():
            self._dir.unlink()
        self._tmp.replace(self._dir)
        return self._dir


def bundle_outdir(outdir: Path | str, out_path: Path | str | None = None) -> Path:
    """
    Rebuild the bundle from a tree of per-pheno gwas.tsv files
    The fallback route, for when only the per-pheno-dirs tree survived and the streamed bundle didnt get written.  Walks outdir for the gwas.tsv files, reads the threshold.txt sitting next to each one, appends a threshold + a significant flag so the bundle matches a streamed one, and streams every table trough a single ParquetWriter so the rebuild never holds the whole thing in memory
    """
    outdir = Path(outdir)
    out_path = Path(out_path) if out_path else outdir / BUNDLE_FILENAME
    tsvs = sorted(outdir.rglob("gwas.tsv"))
    if not tsvs:
        raise FileNotFoundError(f"no gwas.tsv under {outdir}")
    # pin every column type (see _GWAS_TSV_TYPES) so the bundle lines up field-for-field with the
    # streamed route, instead of letting csv inference guess from the text
    parse = pacsv.ParseOptions(delimiter="\t")
    convert = pacsv.ConvertOptions(column_types=_GWAS_TSV_TYPES)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    writer = None
    try:
        for tsv in tsvs:
            table = pacsv.read_csv(tsv, parse_options=parse, convert_options=convert)
            thresh = float((tsv.parent / "threshold.txt").read_text())
            table = table.append_column("threshold", pa.array(np.full(table.num_rows, thresh)))
            table = table.append_column("significant", pc.less(table.column("PValue"), thresh))
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema, compression=_PARQUET_COMPRESSION)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    # rename only once the whole bundle is on disk, so a crash never leaves a half-written parquet
    tmp_path.replace(out_path)
    return out_path


def merge_bundle_parts(outdir: Path | str, out_path: Path | str | None = None) -> Path:
    """
    Gather the per-shard bundle parts a multi-gpu run streamed under .bundle_parts/ into the gwas_bundle.parquet dataset
    Each shard worker commits a shard{i}.parquet directory of per-thread parts, and a directory of parquet parts is itself a valid dataset -- pandas, pyarrow, duckdb and arrow all open read_parquet(directory) the same way they open one file.  So the merge just flattens every part into the bundle directory under a shard-tagged name, no rewrite of the tens of gigabytes they weigh.  Renames are same-filesystem so this lands well under a second whatever the bundle weighs.  Stages the moves in a .tmp directory and renames the whole thing into place at the end, so a reader -- a snakemake job downstream, say -- never catches a bundle thats only half its parts.  Drops the now-empty .bundle_parts on the way out
    """
    outdir = Path(outdir)
    parts_dir = outdir / BUNDLE_PARTS_DIRNAME
    out_path = Path(out_path) if out_path else outdir / BUNDLE_FILENAME
    # each shard committed a shard{i}.parquet directory, collect every part inside it, shard-tagged
    # so the flattened names stay unique once part0.parquet from two shards land side by side
    shard_dirs = sorted(d for d in parts_dir.glob("shard*.parquet") if d.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"no .bundle_parts shard dirs under {outdir}")
    tagged_parts = [(d.stem, p) for d in shard_dirs for p in sorted(d.glob("*.parquet"))]
    if not tagged_parts:
        raise FileNotFoundError(f"the shard dirs under {outdir} hold no parquet parts")
    # footer-only check on every part -- constructing a ParquetFile reads just the metadata, so this
    # catches a truncated or schema-drifted part for free, without streaming the row groups back
    # trough the cpu the way the old rewrite merge did
    schema = pq.ParquetFile(tagged_parts[0][1]).schema_arrow
    for _, part in tagged_parts[1:]:
        if not pq.ParquetFile(part).schema_arrow.equals(schema):
            raise ValueError(f"part schema mismatch: {part} does not line up with {tagged_parts[0][1]}")
    # stage every move in a .tmp directory, then rename that directory in one go -- a downstream read
    # never lands on a bundle thats missing some of its parts
    tmp_dir = out_path.with_name(out_path.name + ".tmp")
    # a stale .tmp could be either a leftover staging dir from an interrupted merge or an orphaned
    # single-file .tmp a crashed BundleWriter never renamed -- clear whichever kind it is, else the
    # mkdir below trips FileExistsError on it
    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)
    elif tmp_dir.exists():
        tmp_dir.unlink()
    tmp_dir.mkdir(parents=True)
    for shard_tag, part in tagged_parts:
        part.replace(tmp_dir / f"{shard_tag}_{part.name}")
    # a stale bundle from an earlier run, file or directory, would block the rename of the new one
    if out_path.is_dir():
        shutil.rmtree(out_path)
    elif out_path.exists():
        out_path.unlink()
    tmp_dir.replace(out_path)
    shutil.rmtree(parts_dir, ignore_errors=True)
    return out_path
