"""
lambda GC: calc_GIF.R definition, shard tables and their merge
cpu-only
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from fasterlmm.inflation import lambda_gc, merge_shard_tables, write_shard_table


def test_lambda_gc_matches_r_definition():
    rng = np.random.default_rng(0)
    p = rng.uniform(size = 5001)
    lam, n = lambda_gc(p)
    assert n == 5001
    assert np.isclose(lam, np.median(chi2.ppf(1 - p, 1)) / chi2.ppf(0.5, 1), rtol = 1e-12)


def test_lambda_gc_uniform_is_about_one_and_inflation_scales():
    rng = np.random.default_rng(1)
    z2 = rng.chisquare(1, size = 200_000)
    assert abs(lambda_gc(chi2.sf(z2, 1))[0] - 1) < 0.02
    assert abs(lambda_gc(chi2.sf(1.5 * z2, 1))[0] - 1.5) < 0.03


def test_lambda_gc_drops_nan_and_handles_empty():
    p = np.array([0.5, np.nan, 0.5])
    lam, n = lambda_gc(p)
    assert n == 2 and np.isclose(lam, 1.0)
    lam, n = lambda_gc(np.array([np.nan]))
    assert np.isnan(lam) and n == 0


def test_shard_tables_merge_in_pheno_order_and_remove_parts(tmp_path):
    idx = {"a": 0, "b": 1, "c": 2, "d": 3}
    write_shard_table(tmp_path, 1, [("d", 1.1, 10), ("b", 0.9, 10)], idx)
    write_shard_table(tmp_path, 0, [("c", 1.0, 10), ("a", 1.2, 10)], idx)
    out = merge_shard_tables(tmp_path)
    df = pd.read_csv(out, sep = "\t")
    assert df["Pheno"].tolist() == ["a", "b", "c", "d"]
    assert not list(tmp_path.glob("lambda_gc.shard*.tsv"))
    assert merge_shard_tables(tmp_path) is None and out.exists()


def test_unsharded_table_name(tmp_path):
    assert write_shard_table(tmp_path, None, [("a", 1.0, 5)], {"a": 0}).name == "lambda_gc.tsv"
