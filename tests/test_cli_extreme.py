"""
tests for fasterlmm.cli_extreme, the big-N streamed LOCO entry point
covers the importable helpers (_pack_perms, _want_resident, the ram-headroom probes) plus a few subprocess runs of the extreme subcommand on the shipped data/example bed.  everything here is cpu-only and portable, no gpu and no fastlmm needed -- the subprocess runs pass --device cpu --no-multi-gpu and keep n_perm tiny so the whole file stays quick
the helper tests pin the perm-packing scheme against an independant numpy gather (mirroring perms.perm_threshold) and check the resident-vs-stream decision is a memory call that never changes the numbers
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fasterlmm.cli_extreme import (
    _meminfo_available_bytes, _pack_perms, _ram_headroom_bytes, _want_resident,
)


# the 14-column fastlmm schema the per-pheno gwas.tsv carries, exact order
GWAS_TSV_COLS = ["sid_index", "SNP", "Chr", "GenDist", "ChrPos", "PValue",
                 "SnpWeight", "SnpWeightSE", "EffectSize", "SnpFractVarExpl",
                 "Mixing", "Nullh2", "Pheno", "PhenoCount"]

# the 20 pheno columns in data/example/example_pheno.tsv, in their tsv order
PHENO_NAMES = ["YAL001C", "YAL002W", "YAL003W", "YAL005C",
               "YBR001C", "YBR002C", "YBR003W", "YBR004C",
               "YGR001C", "YGR002C", "YGR003W", "YGR004W",
               "YLR001C", "YLR002C", "YLR003C", "YLR004C",
               "YPR001W", "YPR002W", "YPR003C", "YPR004C"]


def _run_extreme(geno: Path, pheno: Path, outdir: Path, *extra: str) -> subprocess.CompletedProcess:
    """drive the extreme subcommand by subprocess, cpu-only, surface stderr on a bad exit"""
    argv = [sys.executable, "-m", "fasterlmm", "extreme",
            "--geno", str(geno),
            "--pheno", str(pheno),
            "--outdir", str(outdir),
            "--device", "cpu",
            "--no-multi-gpu",
            "--n-perm", "5",
            *extra]
    r = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert r.returncode == 0, (
        f"extreme {' '.join(extra)} exited {r.returncode}\n"
        f"--- stderr ---\n{r.stderr}\n--- stdout ---\n{r.stdout}")
    return r


# ----- _pack_perms ---------------------------------------------------------

def test_pack_perms_shape_and_real_block() -> None:
    """packed matrix is (N, B + B*n_perm) and its first B columns are the real phenos untouched"""
    rng = np.random.default_rng(19930909)
    N, B, n_perm = 30, 3, 4
    Y_real = torch.from_numpy(rng.standard_normal((N, B)))
    pheno_idx = [2, 7, 11]
    packed = _pack_perms(Y_real, pheno_idx, n_perm, seed=19930909)
    assert packed.shape == (N, B + B * n_perm)
    # the leading B columns must be the real phenos copied through bit for bit
    assert torch.equal(packed[:, :B], Y_real)


def test_pack_perms_matches_independent_gather() -> None:
    """each pheno's perm block matches an independant default_rng([seed, p]) argsort gather, the perms scheme"""
    rng = np.random.default_rng(123)
    N, n_perm = 25, 6
    pheno_idx = [4, 0, 9]
    B = len(pheno_idx)
    Y_real = torch.from_numpy(rng.standard_normal((N, B)))
    seed = 19930909
    packed = _pack_perms(Y_real, pheno_idx, n_perm, seed=seed)

    # rebuild the expected perm columns the same way perms.perm_threshold does it: pheno b_pos seeded
    # on its own original column index p, n_perm row-shuffles via argsort, gathered along the rows
    for b_pos, p in enumerate(pheno_idx):
        prng = np.random.default_rng([seed, int(p)])
        orders = prng.random((n_perm, N)).argsort(axis=1).T  # (N, n_perm)
        block = packed[:, B + b_pos * n_perm:B + (b_pos + 1) * n_perm]
        for j in range(n_perm):
            expected = Y_real[torch.from_numpy(orders[:, j]), b_pos]
            assert torch.equal(block[:, j], expected), f"pheno {p} perm {j} mismatch"


def test_pack_perms_deterministic_across_calls() -> None:
    """same (seed, pheno_idx) gives a byte-identical packed matrix every call"""
    rng = np.random.default_rng(7)
    N, n_perm = 20, 3
    pheno_idx = [1, 5]
    Y_real = torch.from_numpy(rng.standard_normal((N, len(pheno_idx))))
    a = _pack_perms(Y_real, pheno_idx, n_perm, seed=19930909)
    b = _pack_perms(Y_real, pheno_idx, n_perm, seed=19930909)
    assert torch.equal(a, b)


def test_pack_perms_seed_independent_of_batching() -> None:
    """a pheno's perm block is keyed on its own index, so it lands the same whether scanned alone or batched"""
    rng = np.random.default_rng(42)
    N, n_perm = 18, 4
    # the full two-pheno batch, then pheno index 5 scanned on its own
    cols = rng.standard_normal((N, 2))
    Y_both = torch.from_numpy(cols)
    packed_both = _pack_perms(Y_both, [3, 5], n_perm, seed=19930909)

    Y_solo = torch.from_numpy(cols[:, 1:2])  # just the second pheno, original index 5
    packed_solo = _pack_perms(Y_solo, [5], n_perm, seed=19930909)

    # in the batch pheno 5 is b_pos 1, its perm block sits after the 2 real cols + pheno 3's n_perm block
    block_in_batch = packed_both[:, 2 + 1 * n_perm:2 + 2 * n_perm]
    block_solo = packed_solo[:, 1:1 + n_perm]  # B==1 here, real col then the perm block
    assert torch.equal(block_in_batch, block_solo)


# ----- _want_resident + ram probes -----------------------------------------

def test_want_resident_off_always_false() -> None:
    """mode off never goes resident no matter how small the slab"""
    assert _want_resident("off", 0) is False
    assert _want_resident("off", 1 << 60) is False


def test_want_resident_on_always_true() -> None:
    """mode on forces resident even for an absurd slab size"""
    assert _want_resident("on", 0) is True
    assert _want_resident("on", 1 << 60) is True


def test_want_resident_auto_huge_slab_streams() -> None:
    """auto declines resident for a slab far bigger than any plausible headroom"""
    # 1 EiB will never fit under half the cgroup-capped headroom, so auto must stream
    assert _want_resident("auto", 1 << 60) is False


def test_want_resident_auto_zero_slab() -> None:
    """auto on a zero-byte slab goes resident when headroom probes positive, else streams safely"""
    head = _ram_headroom_bytes()
    got = _want_resident("auto", 0)
    if head is not None and head > 0:
        # 0 < 0.5 * positive headroom, so the empty slab always fits
        assert got is True
    else:
        # can't probe the headroom -> the conservative branch streams
        assert got is False


def test_ram_headroom_returns_int_or_none() -> None:
    """the three ram probes each hand back an int or None, never a float or a raise"""
    for fn in (_ram_headroom_bytes, _meminfo_available_bytes):
        v = fn()
        assert v is None or isinstance(v, int), f"{fn.__name__} returned {type(v)}"
        if isinstance(v, int):
            assert v >= 0


# ----- subprocess: the extreme cli on data/example -------------------------

def test_extreme_pheno_idx_single_dir(example_geno, example_pheno, outdir) -> None:
    """--pheno-idx writes exactly that one pheno dir with a 14-col gwas.tsv, p-values in [0, 1]"""
    _run_extreme(example_geno, example_pheno, outdir, "--pheno-idx", "0")
    target = PHENO_NAMES[0]
    pheno_dirs = sorted(d.name for d in outdir.iterdir() if d.is_dir())
    assert pheno_dirs == [target], f"expected just {target}, got {pheno_dirs}"
    tsv = outdir / target / "gwas.tsv"
    assert tsv.exists()
    df = pd.read_csv(tsv, sep="\t")
    assert list(df.columns) == GWAS_TSV_COLS
    assert len(df) == 1500
    assert df.PValue.between(0.0, 1.0).all()
    assert (df.Pheno == target).all()


def test_extreme_bundle(example_geno, example_pheno, outdir) -> None:
    """--bundle on a one-pheno run drops a gwas_bundle.parquet dir pandas can read as one table"""
    _run_extreme(example_geno, example_pheno, outdir, "--pheno-idx", "0", "--bundle")
    bundle = outdir / "gwas_bundle.parquet"
    assert bundle.is_dir(), "bundle should be a directory of part*.parquet files"
    df = pd.read_parquet(str(bundle))
    # one pheno x 1500 variants, the 14 gwas cols plus bundle's threshold + significant
    assert len(df) == 1500
    assert "threshold" in df.columns
    assert "significant" in df.columns
    assert set(df["Pheno"].unique()) == {PHENO_NAMES[0]}


def test_extreme_resident_on_off_same_numbers(example_geno, example_pheno, outdir, tmp_path) -> None:
    """--resident on and --resident off are just memory paths, the gwas.tsv numbers must come out identical"""
    on_dir = tmp_path / "resident_on"
    _run_extreme(example_geno, example_pheno, on_dir, "--pheno-idx", "1", "--resident", "on")
    _run_extreme(example_geno, example_pheno, outdir, "--pheno-idx", "1", "--resident", "off")
    target = PHENO_NAMES[1]
    df_on = pd.read_csv(on_dir / target / "gwas.tsv", sep="\t")
    df_off = pd.read_csv(outdir / target / "gwas.tsv", sep="\t")
    # the streamed and resident paths share the exact low-rank scan, only the genotype delivery differs
    assert list(df_on.columns) == list(df_off.columns)
    assert df_on["SNP"].tolist() == df_off["SNP"].tolist()
    for col in ("PValue", "SnpWeight", "SnpWeightSE", "EffectSize", "SnpFractVarExpl", "Nullh2"):
        np.testing.assert_array_equal(df_on[col].to_numpy(), df_off[col].to_numpy(),
                                      err_msg=f"resident vs stream diverged on {col}")
