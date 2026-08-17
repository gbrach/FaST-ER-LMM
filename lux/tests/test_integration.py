"""Exercise installed commands against real outputs from the public core."""

from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from pysnptools.snpreader import Bed, SnpData


def run_command(command, *args):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    executable = Path(sys.executable).parent / command
    return subprocess.run([str(executable), *map(str, args)], env=env,
                          text=True, capture_output=True, timeout=90)


def success(result):
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("pairwise")
    rng = np.random.default_rng(839)
    n, m = 40, 24
    z = rng.binomial(2, 0.35, size=(n, m)).astype(float)
    ids = [f"sample_{i}" for i in range(n)]
    chrom = np.repeat([1, 2, 3], m // 3)
    pos = np.column_stack([chrom, np.zeros(m), np.arange(m) * 100_000 + 1])
    geno = root / "geno"
    Bed.write(str(geno), SnpData(iid=np.array([[s, s] for s in ids]),
                               sid=[f"rs{i}" for i in range(m)], pos=pos, val=z),
              count_A1=True)
    pheno = root / "pheno.tsv"
    pd.DataFrame({"Strain": ids,
                  "trait_a": 2 * z[:, 0] + z[:, 0] * z[:, 9] + rng.normal(size=n),
                  "trait_b": z[:, 10] + rng.normal(size=n)}).to_csv(pheno, sep="\t", index=False)
    covar = root / "covar.tab"
    c = rng.normal(size=n)
    # Duplicate covariates exercise Lux's adapter for the core input structure.
    pd.DataFrame({"FID": ids, "IID": ids, "c1": c, "c2": c}).to_csv(
        covar, sep="\t", index=False, header=False)
    common = ["--geno", geno, "--pheno", pheno, "--covar", covar, "--device", "cpu"]
    marginals = root / "marginals"
    success(run_command("fasterlmm", "gwas", *common, "--outdir", marginals, "--n-perm", "2"))
    return common, marginals


def pair_args(inputs, outdir):
    common, marginals = inputs
    return [*common, "--marginal-dir", marginals, "--outdir", outdir,
            "--marginal-top-k", "2", "--top-k", "10", "--rint",
            "--exclude-window-kb", "0", "--dtype", "float64", "--write-workers", "1"]


def test_gwas_to_pairwise_and_watch(inputs, tmp_path):
    success(run_command("gwas-epi", *pair_args(inputs, tmp_path), "--bundle", "--n-perm", "3"))
    pairs = pd.read_parquet(tmp_path / "all_pairs.parquet")
    assert set(pairs.Pheno) == {"trait_a", "trait_b"}
    assert pairs.groupby("Pheno").size().eq(10).all()
    assert pairs.PValue.between(0, 1).all()
    assert pairs.SNP_i.ne(pairs.SNP_j).all()
    assert np.isfinite(pairs.threshold).all()
    assert pairs.signif.eq(pairs.PValue < pairs.threshold).all()
    assert np.isfinite(pairs.marginal_PValue_i).all()
    for name in ("trait_a", "trait_b"):
        assert (tmp_path / name / f"{name}.tier2.first_assoc.txt.gz").exists()
        assert (tmp_path / name / f"{name}.threshold.txt").exists()
    status = tmp_path / ".status.json"
    assert json.loads(status.read_text())["stage_state"] == "done"
    success(run_command("epi-watch", status, "--once"))


def test_bundle_only_without_permutations(inputs, tmp_path):
    success(run_command("gwas-epi", *pair_args(inputs, tmp_path), "--bundle", "--no-per-pheno-dirs"))
    pairs = pd.read_parquet(tmp_path / "all_pairs.parquet")
    assert len(pairs) == 20
    assert pairs.threshold.isna().all()
    assert not pairs.signif.any()
    assert not any(p.is_dir() for p in tmp_path.iterdir())


def test_dry_run_does_not_touch_previous_results(inputs, tmp_path):
    status = tmp_path / ".status.json"
    status.write_text('{"previous_run": true}')
    success(run_command("gwas-epi", *pair_args(inputs, tmp_path), "--dry-run", "--bundle"))
    assert status.read_text() == '{"previous_run": true}'
    assert list(tmp_path.iterdir()) == [status]


def test_all_anchors_visits_unique_pairs(inputs, tmp_path):
    common, _ = inputs
    success(run_command("gwas-epi", *common, "--outdir", tmp_path,
                        "--all-anchors", "--top-k", "500", "--maf-floor", "0",
                        "--exclude-window-kb", "0", "--dtype", "float64",
                        "--bundle", "--no-per-pheno-dirs", "--write-workers", "1"))
    pairs = pd.read_parquet(tmp_path / "all_pairs.parquet")
    for _, group in pairs.groupby("Pheno"):
        keys = [tuple(sorted(p)) for p in zip(group.SNP_i, group.SNP_j)]
        assert len(keys) == len(set(keys)) == 24 * 23 // 2


def test_empty_pheno_shard_writes_empty_bundle(inputs, tmp_path):
    success(run_command("gwas-epi", *pair_args(inputs, tmp_path), "--bundle", "--shard", "3/4"))
    assert pd.read_parquet(tmp_path / "all_pairs.shard3.parquet").empty
    assert json.loads((tmp_path / ".status.json.shard3").read_text())["stage_state"] == "done"


def test_shard_worker_preserves_gpu_mask_and_failure(monkeypatch):
    from fasterlmm_lux.epi import cli

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaa,GPU-bbb")
    monkeypatch.setattr(cli, "_run_one", lambda *a, **kw: 2)
    with pytest.raises(SystemExit) as exc:
        cli._shard_entrypoint(1, 2, {})
    assert exc.value.code == 2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-bbb"


def test_all_anchors_rejects_sharding():
    from fasterlmm_lux.epi.cli import parse_args

    with pytest.raises(SystemExit) as exc:
        parse_args(["--geno", "unused", "--pheno", "unused", "--outdir", "unused",
                    "--all-anchors", "--shard", "0/2"])
    assert exc.value.code == 2


@pytest.mark.parametrize("fallback", [False, True])
def test_pair_masks_apply_to_both_scan_paths(monkeypatch, fallback):
    import torch
    from fasterlmm.io import standardise_columns
    from fasterlmm_lux.epi import scan

    if fallback:
        def force_fallback(*args, **kwargs):
            raise torch._C._LinAlgError("exercise fallback pair exclusions")
        monkeypatch.setattr(scan, "snp_wald_scan_batched", force_fallback)
    rng = np.random.default_rng(17)
    z = torch.from_numpy(standardise_columns(rng.binomial(2, 0.4, size=(32, 9)).astype(float)))
    snps = [f"s{i}" for i in range(9)]
    results = scan.scan_tier2(
        Z_std=z, Y=torch.from_numpy(rng.normal(size=(32, 1))),
        X=torch.ones((32, 1), dtype=torch.float64),
        chrom=np.repeat([1, 2, 3], 3), snp_id=snps, pheno_names=["trait"],
        per_pheno_marginals={"trait": snps}, top_k_pair=100,
        symmetric_pairs=True, device="cpu", dtype=torch.float64)
    r = results[0]
    pairs = [tuple(sorted(p)) for p in zip(r.snp_i_idx, r.snp_j_idx)]
    assert len(pairs) == len(set(pairs)) == 9 * 8 // 2
    assert np.all(r.snp_i_idx != r.snp_j_idx)
