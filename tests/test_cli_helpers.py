"""
cli helper + umbrella entry tests
Covers the small importable bits of fasterlmm.cli (shard parsing, writer-pool sizing, the resource snapshot) and drives the fasterlmm.cli_main umbrella by subprocess to check the banner routing.  cpu-only and portable -- no gpu, no fastlmm, nothing past stdlib + the package itself, so this runs on a fresh clone
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from fasterlmm.cli import _default_write_workers, _parse_shard, _resource_stats


# ---- _parse_shard --------------------------------------------------------

def test_parse_shard_valid() -> None:
    """a well-formed X/N parses to the integer pair"""
    assert _parse_shard("0/4") == (0, 4)
    assert _parse_shard("3/4") == (3, 4)
    assert _parse_shard("1/2") == (1, 2)


def test_parse_shard_index_equals_n_raises() -> None:
    """index has to be strictly less than n, so X/N with X==N is out of range"""
    with pytest.raises(ValueError):
        _parse_shard("4/4")


def test_parse_shard_index_above_n_raises() -> None:
    """an index past n is rejected"""
    with pytest.raises(ValueError):
        _parse_shard("5/4")


def test_parse_shard_negative_raises() -> None:
    """a negative shard index is out of range (lower bound is zero)"""
    with pytest.raises(ValueError):
        _parse_shard("-1/4")


# ---- _default_write_workers ---------------------------------------------

def test_default_write_workers_floor_is_one() -> None:
    """the pool size never drops below one, whatever the proc count"""
    assert _default_write_workers(1) >= 1
    assert _default_write_workers(4) >= 1
    # even a silly proc count larger than the core budget still leaves one writer
    assert _default_write_workers(100000) >= 1


def test_default_write_workers_shrinks_with_more_procs() -> None:
    """splitting the cores across more procs cant give a bigger slice than one proc gets the whole budget"""
    one = _default_write_workers(1)
    four = _default_write_workers(4)
    assert four <= one


# ---- _resource_stats -----------------------------------------------------

def test_resource_stats_cpu_has_rss() -> None:
    """on linux the cpu snapshot carries current + peak host rss, both finite and non-negative"""
    stats = _resource_stats("cpu")
    assert isinstance(stats, dict)
    # /proc/self/status exists on linux so both keys land, and a live process has used some memory
    assert "rss_mb" in stats
    assert "peak_rss_mb" in stats
    assert stats["rss_mb"] >= 0.0
    assert stats["peak_rss_mb"] >= stats["rss_mb"]


def test_resource_stats_cpu_no_gpu_keys() -> None:
    """a cpu device never injects the cuda gpu_* fields"""
    stats = _resource_stats("cpu")
    assert not any(k.startswith("gpu_") for k in stats)


# ---- umbrella entry (subprocess) ----------------------------------------

# the four subcommands the banner advertises, every banner print has to name all of them
_SUBCOMMANDS = ("gwas", "extreme", "watch", "concat")


def _run_module(*argv: str) -> subprocess.CompletedProcess:
    """invoke the umbrella as a module so we exercise the real __main__ dispatch, not an in-proc import"""
    return subprocess.run([sys.executable, "-m", "fasterlmm", *argv],
                          capture_output=True, text=True, check=False)


def test_umbrella_bare_prints_banner() -> None:
    """a bare invoke exits clean and lists every subcommand"""
    r = _run_module()
    assert r.returncode == 0
    for sub in _SUBCOMMANDS:
        assert sub in r.stdout, f"banner missing {sub}"


def test_umbrella_help_prints_banner() -> None:
    """--help exits clean and lists every subcommand"""
    r = _run_module("--help")
    assert r.returncode == 0
    for sub in _SUBCOMMANDS:
        assert sub in r.stdout, f"banner missing {sub}"
    # -h is the short form, same deal
    r2 = _run_module("-h")
    assert r2.returncode == 0
    for sub in _SUBCOMMANDS:
        assert sub in r2.stdout


def test_umbrella_unknown_subcommand_exits_2() -> None:
    """an unrecognised subcommand exits 2 and prints the banner to stderr"""
    r = _run_module("bogus")
    assert r.returncode == 2
    # the banner gets reprinted on stderr, with the unknown-subcommand line above it
    assert "unknown subcommand" in r.stderr
    for sub in _SUBCOMMANDS:
        assert sub in r.stderr, f"stderr banner missing {sub}"
