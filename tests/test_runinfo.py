"""
run record (run_info.json + run.log): command line, args, input files, outcome, per-shard names
cpu-only, no scan
"""
from __future__ import annotations

import argparse
import json
import sys

from fasterlmm import runinfo


def _args(tmp_path):
    (tmp_path / "p.tsv").write_text("Strain\ta\ns1\t1\n")
    return argparse.Namespace(outdir = str(tmp_path / "out"), geno = str(tmp_path / "missing"),
                              pheno = str(tmp_path / "p.tsv"), covar = None, seed = 7)


def test_begin_and_update_write_record_and_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fasterlmm gwas", "--seed", "7", "--outdir", "x y"])
    out, err = sys.stdout, sys.stderr
    try:
        path = runinfo.begin(_args(tmp_path), "gwas")
        print("hello log", file = sys.stderr)
    finally:
        sys.stdout, sys.stderr = out, err
    runinfo.update(tmp_path / "out", None, state = "done", N = 3)
    rec = json.loads(path.read_text())
    assert rec["state"] == "done" and rec["N"] == 3
    assert rec["command_line"] == "fasterlmm gwas --seed 7 --outdir 'x y'"
    assert rec["args"]["seed"] == 7
    roles = {e["role"]: e for e in rec["inputs"]}
    assert roles["pheno"]["size_bytes"] > 0
    assert roles["geno.bed"]["missing"] is True
    assert "covar" not in roles
    assert "hello log" in (tmp_path / "out" / "run.log").read_text()


def test_shard_names_and_update_creates_missing(tmp_path):
    assert runinfo.record_path(tmp_path, 2).name == "run_info.shard2.json"
    assert runinfo.log_path(tmp_path, 2).name == "run.shard2.log"
    runinfo.update(tmp_path, 2, state = "done")
    assert json.loads((tmp_path / "run_info.shard2.json").read_text())["state"] == "done"
