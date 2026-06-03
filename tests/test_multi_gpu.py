"""
multi-gpu dispatch tests
Exercising the auto-spawn path (one worker per visible cuda device) on the shipped 20-pheno example
Round-robin slicing means shard 0 gets phenos 0,2,4,...,18 and shard 1 gets 1,3,5,...,19, so the two shards write roughly balanced row counts to gwas_bundle.parquet
Skipped wholesale on login nodes (no cuda, or <2 gpus), the real run happens under srun -p gpu --gres=gpu:tesla:2
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="multi-GPU tests need >=2 CUDA devices")


def _run_fasterlmm(argv: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    """invoking the fasterlmm cli through the current python so the test always picks up the editable install"""
    cmd = [sys.executable, "-m", "fasterlmm", *argv]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)


def test_multi_gpu_dispatch(
    example_geno: Path,
    example_pheno: Path,
    outdir: Path
) -> None:
    """auto-dispatch with bare --device cuda fans out one worker per visible gpu, both write status.shard{i}.json and stream into .bundle_parts/shard{i}.parquet, the parent then folds those parts into gwas_bundle.parquet"""
    res = _run_fasterlmm([
        "gwas",
        "--geno", str(example_geno),
        "--pheno", str(example_pheno),
        "--outdir", str(outdir),
        "--device", "cuda",
        "--bundle",
        "--n-perm", "10",
        "--no-rint",
    ])
    assert res.returncode == 0, f"gwas failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    # per-worker status files land alongside the orchestrators aggregated status.json
    assert (outdir / "status.shard0.json").exists(), "missing status.shard0.json"
    assert (outdir / "status.shard1.json").exists(), "missing status.shard1.json"
    assert (outdir / "status.json").exists(), "missing aggregated status.json"

    # the merged bundle is a directory of parquet parts, pandas reads it as one dataset
    bundle = outdir / "gwas_bundle.parquet"
    assert bundle.is_dir(), f"expected gwas_bundle.parquet to be a directory of parts, got {bundle}"

    df = pd.read_parquet(bundle)
    phenos = set(df["Pheno"].unique())
    assert len(phenos) == 20, f"expected 20 phenos in bundle, found {len(phenos)}: {sorted(phenos)}"

    # round-robin slicing puts phenos 0,2,4,...,18 on shard 0 and 1,3,5,...,19 on shard 1, so each
    # shard contributes ~half the rows.  not exactly half if any pheno droped variants (NaN rows),
    # so we check the ratio sits within a loose 0.4..0.6 band
    pheno_cols = [c for c in pd.read_csv(example_pheno, sep="\t", nrows=0).columns if c != "Strain"]
    even = {pheno_cols[i] for i in range(0, len(pheno_cols), 2)}
    odd = {pheno_cols[i] for i in range(1, len(pheno_cols), 2)}
    n_even = df["Pheno"].isin(even).sum()
    n_odd = df["Pheno"].isin(odd).sum()
    total = n_even + n_odd
    assert total > 0, "bundle has zero rows"
    ratio = n_even / total
    assert 0.4 <= ratio <= 0.6, (
        f"shard row balance off, shard0 phenos got {n_even} rows / shard1 phenos got {n_odd} "
        f"(ratio {ratio:.3f}), expected ~0.5")


def test_multi_gpu_single_gpu_flag(
    example_geno: Path,
    example_pheno: Path,
    outdir: Path
) -> None:
    """opting out of the auto-dispatch with --no-multi-gpu collapses to one scan process, so no per-shard status files appear and all 20 phenos land in a single pane"""
    res = _run_fasterlmm([
        "gwas",
        "--geno", str(example_geno),
        "--pheno", str(example_pheno),
        "--outdir", str(outdir),
        "--device", "cuda",
        "--no-multi-gpu",
        "--bundle",
        "--n-perm", "10",
        "--no-rint",
    ])
    assert res.returncode == 0, f"gwas failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    assert (outdir / "status.json").exists(), "missing status.json"
    # no shard files when the dispatch is supressed
    assert not (outdir / "status.shard0.json").exists(), "status.shard0.json should not exist with --no-multi-gpu"
    assert not (outdir / "status.shard1.json").exists(), "status.shard1.json should not exist with --no-multi-gpu"

    bundle = outdir / "gwas_bundle.parquet"
    assert bundle.is_dir(), f"expected gwas_bundle.parquet to be a directory of parts, got {bundle}"
    df = pd.read_parquet(bundle)
    phenos = set(df["Pheno"].unique())
    assert len(phenos) == 20, f"expected 20 phenos in single-gpu bundle, found {len(phenos)}"


def test_multi_gpu_shard_array_then_concat(
    example_geno: Path,
    example_pheno: Path,
    outdir: Path
) -> None:
    """simulating a 2-task slurm array, two --shard invocations each pinned to one gpu via CUDA_VISIBLE_DEVICES, then fasterlmm concat folds the parts together"""
    base_env = os.environ.copy()

    for shard_i, gpu in enumerate(("0", "1")):
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        res = _run_fasterlmm([
            "gwas",
            "--geno", str(example_geno),
            "--pheno", str(example_pheno),
            "--outdir", str(outdir),
            "--device", "cuda",
            "--shard", f"{shard_i}/2",
            "--bundle",
            "--n-perm", "10",
            "--no-rint",
        ], env=env)
        assert res.returncode == 0, (
            f"shard {shard_i}/2 failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")
        assert (outdir / f"status.shard{shard_i}.json").exists(), f"missing status.shard{shard_i}.json"

    # both shards streamed into .bundle_parts/shard{i}.parquet, concat does the merge
    parts_dir = outdir / ".bundle_parts"
    assert parts_dir.is_dir(), f"missing parts dir {parts_dir}"
    assert (parts_dir / "shard0.parquet").exists(), "missing .bundle_parts/shard0.parquet"
    assert (parts_dir / "shard1.parquet").exists(), "missing .bundle_parts/shard1.parquet"

    res = _run_fasterlmm(["concat", str(outdir)])
    assert res.returncode == 0, f"concat failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    bundle = outdir / "gwas_bundle.parquet"
    assert bundle.is_dir(), f"expected gwas_bundle.parquet directory, got {bundle}"
    df = pd.read_parquet(bundle)
    phenos = set(df["Pheno"].unique())
    assert len(phenos) == 20, f"expected 20 phenos after concat, found {len(phenos)}"
