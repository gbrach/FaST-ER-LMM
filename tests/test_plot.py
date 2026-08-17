"""Saved-result plotting, PDF output, and scan/dispatch integration on CPU."""

from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pypdf import PdfReader
import pytest

from fasterlmm.plot import Results, plot_results, prepare_data, read_chrom_sizes


def frame(name="trait_a"):
    return pd.DataFrame({"Pheno": [name] * 4, "SNP": ["s1", "s2", "s3", "s4"],
                         "Chr": [10.0, 2.0, 1.0, 1.0], "ChrPos": [50, 200, 100, 150],
                         "PValue": [0.2, 0.0001, 0.5, 0.03], "threshold": [0.01] * 4})


def write_bundle(path, names=("trait_a", "trait_b")):
    path.mkdir(parents=True)
    table = pa.Table.from_pandas(pd.concat([frame(n) for n in names]), preserve_index=False)
    pq.write_table(table, path / "part0.parquet", row_group_size=4)
    return path


def write_tsv(root, name="trait_a", threshold=True):
    path = root / name / "gwas.tsv"
    path.parent.mkdir(parents=True)
    frame(name).drop(columns="threshold").to_csv(path, sep="\t", index=False)
    if threshold:
        (path.parent / "threshold.txt").write_text("0.01\n")
    return path


def run_cli(*args):
    result = subprocess.run([sys.executable, "-m", "fasterlmm", *map(str, args)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_natural_chromosome_order_positions_and_hits():
    data, sizes, centers, total, threshold = prepare_data(frame())
    assert list(sizes) == ["1", "2", "10"]
    assert data.x.tolist() == [400, 350, 100, 150]
    assert centers == [75, 250, 375]
    assert total == 400
    assert threshold == 0.01
    assert data.loc[data.hit, "SNP"].tolist() == ["s2"]


def test_invalid_values_and_zero_p():
    data = frame()
    data["PValue"] = [0, np.nan, -1, np.inf]
    with pytest.warns(UserWarning, match="skipping 3"):
        plotted, *_ = prepare_data(data)
    assert len(plotted) == 1
    assert np.isfinite(plotted.y).all()
    assert plotted.y.iloc[0] > 300


def test_chromosome_size_file_preserves_order_and_validates_bounds(tmp_path):
    sizes = tmp_path / "chrom.fai"
    sizes.write_text("2\t500\t0\t60\t61\n1\t300\t0\t60\t61\n10\t100\t0\t60\t61\n")
    mapping = read_chrom_sizes(sizes)
    _, labels, _, total, _ = prepare_data(frame(), mapping)
    assert list(labels) == ["2", "1", "10"]
    assert total == 900
    with pytest.raises(ValueError, match="too short"):
        prepare_data(frame(), {"1": 100, "2": 500, "10": 100})


def test_inconsistent_thresholds_are_rejected():
    data = frame()
    data.loc[0, "threshold"] = 0.05
    with pytest.raises(ValueError, match="one finite permutation threshold"):
        prepare_data(data)


def test_parquet_dataset_writes_one_page_per_pheno(tmp_path):
    source = write_bundle(tmp_path / "gwas_bundle.parquet")
    outputs = plot_results(source)
    assert outputs == [tmp_path / "manhattan.pdf"]
    pages = PdfReader(outputs[0]).pages
    assert len(pages) == 2
    assert "trait_a" in pages[0].extract_text()
    assert "trait_b" in pages[1].extract_text()
    assert "s2" in pages[0].extract_text()
    assert not list(source.glob("*.pdf"))


def test_mixed_row_groups_and_split_pheno_are_gathered(tmp_path):
    source = tmp_path / "mixed.parquet"
    mixed = pd.concat([frame("b").iloc[:2], frame("a"), frame("b").iloc[2:]])
    pq.write_table(pa.Table.from_pandas(mixed, preserve_index=False), source, row_group_size=3,
                   write_statistics=False)
    results = Results(source)
    assert results.names == ["a", "b"]
    assert len(results.load("b")) == 4
    assert set(results.load("a").Pheno) == {"a"}
    output = plot_results(source, phenotypes=["b"])[0]
    assert len(PdfReader(output).pages) == 1
    assert "b" in PdfReader(output).pages[0].extract_text()


def test_tsvs_create_separate_pdfs_and_can_be_combined(tmp_path):
    write_tsv(tmp_path, "trait_a")
    write_tsv(tmp_path, "trait_b", threshold=False)
    outputs = plot_results(tmp_path)
    assert set(outputs) == {tmp_path / name / "manhattan.pdf" for name in ("trait_a", "trait_b")}
    assert "threshold unavailable" in PdfReader(tmp_path / "trait_b/manhattan.pdf").pages[0].extract_text()
    combined = plot_results(tmp_path, bundle=True)[0]
    assert len(PdfReader(combined).pages) == 2


def test_failures_preserve_existing_pdf(tmp_path):
    source = write_bundle(tmp_path / "gwas_bundle.parquet")
    destination = tmp_path / "manhattan.pdf"
    destination.write_bytes(b"previous output")
    with pytest.raises(ValueError, match="phenotypes not found"):
        plot_results(source, phenotypes=["absent"])
    bad = frame("trait_b")
    bad["threshold"] = -1.0
    pq.write_table(pa.Table.from_pandas(bad, preserve_index=False), source / "part1.parquet")
    with pytest.raises(ValueError, match="threshold"):
        plot_results(source)
    assert destination.read_bytes() == b"previous output"
    assert not list(tmp_path.glob(".manhattan-*.pdf"))


def test_reject_pdf_inside_parquet_dataset(tmp_path):
    source = write_bundle(tmp_path / "gwas_bundle.parquet")
    with pytest.raises(ValueError, match="outside its directory"):
        plot_results(source, out=source / "plots.pdf")


def test_plot_cli_accepts_run_dir_and_selection(tmp_path):
    write_bundle(tmp_path / "gwas_bundle.parquet")
    run_cli("plot", tmp_path, "--manhattan", "--pheno", "trait_b", "--label-top", "0")
    pdf = PdfReader(tmp_path / "manhattan.pdf")
    assert len(pdf.pages) == 1
    assert "trait_b" in pdf.pages[0].extract_text()
    assert "s2" not in pdf.pages[0].extract_text()


def scan_args(example_geno, example_pheno, out):
    return ["--geno", example_geno, "--pheno", example_pheno, "--outdir", out,
            "--device", "cpu", "--pheno-end", "2", "--n-perm", "2", "--write-workers", "1"]


@pytest.mark.parametrize("command", ["gwas", "extreme"])
def test_scan_bundle_only_writes_multipage_pdf(command, example_geno, example_pheno, tmp_path):
    extra = ["--grm-k", "20", "--float64"] if command == "extreme" else []
    run_cli(command, *scan_args(example_geno, example_pheno, tmp_path), *extra,
            "--bundle", "--no-per-pheno-dirs", "--manhattan")
    assert len(PdfReader(tmp_path / "manhattan.pdf").pages) == 2
    assert not (tmp_path / "YAL001C").exists()
    assert (tmp_path / "gwas_bundle.parquet").is_dir()


def test_scan_without_bundle_ignores_stale_bundle(example_geno, example_pheno, tmp_path):
    write_bundle(tmp_path / "gwas_bundle.parquet", names=["stale"])
    run_cli("gwas", *scan_args(example_geno, example_pheno, tmp_path), "--manhattan")
    for name in ("YAL001C", "YAL002W"):
        page = PdfReader(tmp_path / name / "manhattan.pdf").pages[0]
        assert name in "".join(page.extract_text().split())
    assert not (tmp_path / "manhattan.pdf").exists()


def test_shard_pdfs_and_concat(example_geno, example_pheno, tmp_path):
    for shard in (0, 1):
        run_cli("gwas", *scan_args(example_geno, example_pheno, tmp_path), "--bundle",
                "--no-per-pheno-dirs", "--manhattan", "--shard", f"{shard}/2")
        assert len(PdfReader(tmp_path / f"manhattan.shard{shard}.pdf").pages) == 1
    run_cli("concat", tmp_path, "--manhattan")
    assert len(PdfReader(tmp_path / "manhattan.pdf").pages) == 2


def test_dry_run_does_not_make_plots(example_geno, example_pheno, tmp_path):
    run_cli("gwas", *scan_args(example_geno, example_pheno, tmp_path), "--bundle", "--manhattan", "--dry-run")
    assert not list(tmp_path.rglob("*.pdf"))


@pytest.mark.parametrize("module_name,worker", [("cli", "_run_scan"), ("cli_extreme", "_run_extreme")])
def test_auto_gpu_parent_plots_after_merge(monkeypatch, tmp_path, module_name, worker):
    import importlib
    from fasterlmm.bundle import BundleWriter
    module = importlib.import_module(f"fasterlmm.{module_name}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 2)

    def fake_scan(args, shard_i, shard_n, device):
        assert not args.manhattan  # parent renders once; children only write data
        writer = BundleWriter(tmp_path / ".bundle_parts" / f"shard{shard_i}.parquet")
        writer.append(pa.Table.from_pandas(frame(f"trait_{shard_i}"), preserve_index=False))
        writer.close()

    class Process:
        exitcode = 0

        def __init__(self, target, args):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

        def join(self):
            pass

    monkeypatch.setattr(module, worker, fake_scan)
    monkeypatch.setattr(module.multiprocessing, "get_context", lambda _: SimpleNamespace(Process=Process))
    monkeypatch.setattr(sys, "argv", ["fasterlmm", "--geno", "unused", "--pheno", "unused",
                                      "--outdir", str(tmp_path), "--bundle", "--manhattan"])
    module.main()
    assert len(PdfReader(tmp_path / "manhattan.pdf").pages) == 2
    assert not list(tmp_path.glob("manhattan.shard*.pdf"))
