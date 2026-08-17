"""Manhattan PDFs from saved GWAS results; independent of the LMM scan math."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import os
from pathlib import Path
import re
import tempfile
import textwrap
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fasterlmm.bundle import BUNDLE_FILENAME, BUNDLE_PARTS_DIRNAME


REQUIRED = {"SNP", "Chr", "ChrPos", "PValue"}
COLUMNS = ["Pheno", "SNP", "Chr", "ChrPos", "PValue", "threshold"]


def chromosome_label(value) -> str:
    """Keep named chromosomes; normalize numeric labels such as 1.0 to 1."""
    label = str(value).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", label):
        return str(int(float(label)))
    return label


def _chromosome_key(label):
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold())
                 for part in re.split(r"(\d+)", label) if part)


def read_chrom_sizes(path: str | Path | None) -> dict[str, float] | None:
    """Read chromosome/length columns, without a header; .fai files also work."""
    if path is None:
        return None
    sizes = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError("chromosome sizes need two columns: chromosome length")
        chrom, length = chromosome_label(fields[0]), float(fields[1])
        if chrom in sizes or not np.isfinite(length) or length <= 0:
            raise ValueError(f"invalid or duplicate chromosome size: {line}")
        sizes[chrom] = length
    if not sizes:
        raise ValueError("chromosome size file is empty")
    return sizes


class Results:
    """Index phenotype row groups, keeping only one phenotype's data in memory.

    Native bundles store one phenotype per row group, so metadata is enough to
    build the index. For external Parquet files with mixed row groups, read only
    their Pheno column to identify the groups, then filter when loading a page.
    """

    def __init__(self, path: str | Path):
        source = Path(path)
        if source.is_dir() and (source / BUNDLE_FILENAME).exists():
            source = source / BUNDLE_FILENAME
        self.source = source
        self.groups = defaultdict(list)
        self.tsvs = {}
        if source.is_file() and source.suffix == ".tsv":
            self.tsvs[source.parent.name] = source
        elif source.is_dir() and (source / "gwas.tsv").exists():
            self.tsvs[source.name] = source / "gwas.tsv"
        elif source.is_dir() and source.name != BUNDLE_FILENAME and not list(source.glob("*.parquet")):
            self.tsvs = {p.parent.name: p for p in sorted(source.glob("*/gwas.tsv"))}
        else:
            parts = sorted(source.rglob("*.parquet")) if source.is_dir() else [source]
            for part in parts:
                if not part.is_file():
                    continue
                parquet = pq.ParquetFile(part)
                names = parquet.schema_arrow.names
                missing = (REQUIRED | {"Pheno"}) - set(names)
                if missing:
                    raise ValueError(f"{part}: missing GWAS columns {', '.join(sorted(missing))}")
                for i in range(parquet.num_row_groups):
                    group = parquet.metadata.row_group(i)
                    if not group.num_rows:
                        continue
                    stats = group.column(names.index("Pheno")).statistics
                    if (stats is not None and stats.has_min_max and stats.null_count == 0
                            and stats.min == stats.max):
                        phenos = [stats.min]
                    else:
                        phenos = parquet.read_row_group(i, columns=["Pheno"]).column(0).unique().to_pylist()
                    for pheno in phenos:
                        if pheno is None:
                            raise ValueError(f"{part}: missing phenotype name")
                        self.groups[str(pheno)].append((part, i))
        self.bundled = bool(self.groups)
        self.names = sorted(self.groups or self.tsvs)
        if not self.names:
            raise ValueError(f"no GWAS results found at {path}")

    def load(self, name: str) -> pd.DataFrame:
        if self.bundled:
            frames = []
            for part, index in self.groups[name]:
                parquet = pq.ParquetFile(part)
                columns = [c for c in COLUMNS if c in parquet.schema_arrow.names]
                frame = parquet.read_row_group(index, columns=columns).to_pandas()
                frames.append(frame.loc[frame["Pheno"].astype(str) == name])
            return pd.concat(frames, ignore_index=True)
        path = self.tsvs[name]
        frame = pd.read_csv(path, sep="\t", dtype={"SNP": str, "Chr": str},
                            usecols=lambda col: col in COLUMNS)
        if "Pheno" in frame and set(frame.Pheno.dropna().astype(str)) != {name}:
            raise ValueError(f"{path}: expected one phenotype named {name!r}")
        threshold = path.parent / "threshold.txt"
        if "threshold" not in frame and threshold.exists():
            frame["threshold"] = float(threshold.read_text().strip())
        return frame

    def default_output(self, name: str | None = None) -> Path:
        if name is not None and not self.bundled:
            return self.tsvs[name].parent / "manhattan.pdf"
        return (self.source.parent if self.bundled or self.source.is_file() else self.source) / "manhattan.pdf"


def prepare_data(frame: pd.DataFrame, sizes=None):
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"missing GWAS columns: {', '.join(sorted(missing))}")
    data = frame.copy()
    positions = pd.to_numeric(data.ChrPos, errors="coerce")
    valid_pos = data.Chr.notna() & np.isfinite(positions) & (positions >= 0)
    data = data.loc[valid_pos].copy()
    data["ChrPos"] = positions.loc[valid_pos]
    data["Chr"] = data.Chr.map(chromosome_label)
    observed = data.groupby("Chr").ChrPos.max().to_dict()
    if sizes is None:
        sizes = {chrom: max(1.0, observed[chrom]) for chrom in sorted(observed, key=_chromosome_key)}
    else:
        for chrom, maximum in observed.items():
            if chrom not in sizes or maximum > sizes[chrom]:
                raise ValueError(f"chromosome size missing or too short for {chrom}")
    pvalues = pd.to_numeric(data.PValue, errors="coerce")
    valid_p = np.isfinite(pvalues) & pvalues.between(0, 1)
    skipped = len(frame) - int(valid_p.sum())
    if skipped:
        warnings.warn(f"skipping {skipped} rows with invalid p-values or genomic positions", stacklevel=2)
    data = data.loc[valid_p].copy()
    data["PValue"] = pvalues.loc[valid_p]
    if data.empty:
        raise ValueError("no valid variants to plot")
    threshold = None
    if "threshold" in frame:
        values = pd.to_numeric(frame.threshold, errors="raise").dropna().unique()
        if len(values) > 1 or (len(values) == 1 and not (np.isfinite(values[0]) and 0 < values[0] <= 1)):
            raise ValueError("expected one finite permutation threshold in (0, 1] per phenotype")
        if len(values):
            threshold = float(values[0])
    offsets, centers, total = {}, [], 0.0
    for chrom, length in sizes.items():
        offsets[chrom] = total
        centers.append(total + length / 2)
        total += length
    data["x"] = data.ChrPos + data.Chr.map(offsets)
    data["y"] = -np.log10(data.PValue.clip(lower=np.nextafter(0.0, 1.0)))
    data["hit"] = data.PValue < threshold if threshold is not None else False
    return data, sizes, centers, total, threshold


def _label_hits(ax, hits, limit):
    """Place a bounded set of lead labels without overlap or clipping."""
    occupied = []
    canvas = ax.figure.canvas
    canvas.draw()
    for row in hits.sort_values(["PValue", "SNP"], kind="stable").head(limit).itertuples():
        label = "\n".join(textwrap.wrap(str(row.SNP), width=27))
        placed = False
        for dy in (10, 26, 42, 58, -20, -36, -52):
            for dx in (0, 35, -35, 70, -70, 105, -105):
                annotation = ax.annotate(label, (row.x, row.y), xytext=(dx, dy),
                                         textcoords="offset points", ha="center", va="bottom",
                                         fontsize=7, color="black", annotation_clip=False,
                                         bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1},
                                         arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.45})
                # Use the text box, not the connector, for collision detection.
                annotation.update_positions(canvas.get_renderer())
                annotation.update_bbox_position_size(canvas.get_renderer())
                bounds = annotation.get_bbox_patch().get_window_extent().expanded(1.04, 1.1)
                inside = ax.bbox.contains(bounds.x0, bounds.y0) and ax.bbox.contains(bounds.x1, bounds.y1)
                if inside and not any(bounds.overlaps(other) for other in occupied):
                    occupied.append(bounds)
                    placed = True
                    break
                annotation.remove()
            if placed:
                break


def manhattan_figure(frame, name, *, chrom_sizes=None, label_top=10):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    data, sizes, centers, total, threshold = prepare_data(frame, chrom_sizes)
    fig = Figure(figsize=(10, 6), facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.88)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.65)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, chrom in enumerate(sizes):
        rows = data.loc[data.Chr == chrom]
        ax.scatter(rows.x, rows.y, s=6, color=("skyblue", "navy")[i % 2],
                   alpha=0.6, edgecolors="none", rasterized=True)
    hits = data.loc[data.hit]
    ax.scatter(hits.x, hits.y, s=28, c="black", marker="^", edgecolors="none", zorder=3)
    high = float(data.y.max())
    if threshold is not None:
        line = -np.log10(threshold)
        ax.axhline(line, color="red", linestyle="--", linewidth=0.9)
        high = max(high, line)
    ax.set(xlim=(-total * 0.005, total * 1.005), ylim=(0, max(1, high) * 1.25 + 0.5),
           xlabel="chromosome", ylabel="-log10(p)")
    ax.set_xticks(centers, list(sizes))
    if len(sizes) > 24:
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(length=0, pad=6)
    ax.set_title("\n".join(textwrap.wrap(str(name), width=80)), fontweight="bold", fontsize=12, pad=14)
    _label_hits(ax, hits, label_top)
    footer = "Permutation threshold unavailable" if threshold is None else f"Permutation threshold: {threshold:.3g}"
    if len(hits) > label_top:
        footer += f"  |  up to {label_top} strongest hits labelled"
    if (data.PValue == 0).any():
        footer += "  |  p=0 clipped to smallest positive float"
    fig.text(0.09, 0.045, footer, fontsize=7, color="#666666")
    return fig


@contextmanager
def _pdf_output(path):
    from matplotlib import rc_context
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    if path.suffix.lower() != ".pdf":
        raise ValueError("plot output must end in .pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".manhattan-", suffix=".pdf", dir=path.parent)
    os.close(descriptor)
    try:
        with rc_context({"pdf.fonttype": 42, "font.family": "DejaVu Sans"}):
            with PdfPages(temporary, metadata={"Title": "Manhattan plots", "Creator": "FaST-ER-LMM"}) as pdf:
                yield pdf
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def plot_results(source, *, out=None, bundle=False, phenotypes=None, chrom_sizes=None, label_top=10):
    """Render saved results. An explicit output path always creates one PDF."""
    if label_top < 0:
        raise ValueError("--label-top must be >= 0")
    results = Results(source)
    names = results.names if phenotypes is None else sorted(set(phenotypes))
    unknown = set(names) - set(results.names)
    if unknown:
        raise ValueError(f"phenotypes not found: {', '.join(sorted(unknown))}")
    if not names:  # empty scan shard
        return []
    sizes = read_chrom_sizes(chrom_sizes)
    outputs = []
    groups = [(Path(out) if out else results.default_output(), names)] if (
        out is not None or bundle or results.bundled) else [(results.default_output(n), [n]) for n in names]
    for path, page_names in groups:
        if results.bundled and results.source.is_dir() and path.resolve().is_relative_to(results.source.resolve()):
            raise ValueError("write the PDF alongside the Parquet dataset, outside its directory")
        with _pdf_output(path) as pdf:
            for name in page_names:
                fig = manhattan_figure(results.load(name), name, chrom_sizes=sizes, label_top=label_top)
                pdf.savefig(fig, dpi=300)
                fig.clear()
        outputs.append(path)
    print(f"wrote {len(names)} Manhattan plot(s) to {len(outputs)} PDF(s)", flush=True)
    return outputs


def add_scan_arguments(parser):
    parser.add_argument("--manhattan", action="store_true", help="write Manhattan PDFs after scanning; --bundle makes one multipage PDF")
    parser.add_argument("--chrom-sizes", help="chromosome lengths: whitespace-delimited chromosome/length columns, no header (or .fai)")
    parser.add_argument("--label-top", type=int, default=10, help="maximum significant SNP labels per plot (0 disables labels)")


def validate_scan_arguments(parser, args):
    if args.label_top < 0:
        parser.error("--label-top must be >= 0")
    if args.manhattan:
        try:
            read_chrom_sizes(args.chrom_sizes)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))


def plot_scan(args, names, shard_i=None):
    if not getattr(args, "manhattan", False) or not names:
        return
    root = Path(args.outdir)
    if args.bundle:
        source = root / BUNDLE_FILENAME if shard_i is None else root / BUNDLE_PARTS_DIRNAME / f"shard{shard_i}.parquet"
        out = root / ("manhattan.pdf" if shard_i is None else f"manhattan.shard{shard_i}.pdf")
        plot_results(source, out=out, phenotypes=names, chrom_sizes=args.chrom_sizes, label_top=args.label_top)
    else:
        # Read only this run's phenotype folders, even if an older bundle exists.
        for name in names:
            plot_results(root / name / "gwas.tsv", chrom_sizes=args.chrom_sizes, label_top=args.label_top)


def main():
    parser = argparse.ArgumentParser(prog="fasterlmm plot", description="Plot saved GWAS TSVs or a Parquet bundle without rerunning the scan")
    parser.add_argument("input", help="run directory, phenotype directory, gwas.tsv, or Parquet file/dataset")
    add_scan_arguments(parser)
    parser.add_argument("--out", help="one output PDF (default: manhattan.pdf alongside the input)")
    parser.add_argument("--bundle", action="store_true", help="combine per-phenotype TSV plots into one PDF")
    parser.add_argument("--pheno", action="append", help="phenotype name to plot; repeat to select several")
    args = parser.parse_args()
    if not args.manhattan:
        parser.error("select --manhattan")
    validate_scan_arguments(parser, args)
    try:
        plot_results(args.input, out=args.out, bundle=args.bundle, phenotypes=args.pheno,
                     chrom_sizes=args.chrom_sizes, label_top=args.label_top)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
