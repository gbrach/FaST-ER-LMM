"""unit tests for the marginal-dir reader, focused on the GWAS-first addition:
the reader must accept the clean core's per-pheno-dirs layout (<pheno>/gwas.tsv +
<pheno>/threshold.txt) in addition to the lux/dev-native <pheno>.first_assoc.* —
that's what `fasterlmm gwas` produces on cluster scratch in the fused per-shard mode.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fasterlmm_lux.epi.marginals import (
    read_marginals, read_marginal_stats, read_marginal_thresholds)


def _write_gwas_tsv(d, snps, pvals, betas):
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SNP": snps, "PValue": pvals, "SnpWeight": betas}).to_csv(
        d / "gwas.tsv", sep="\t", index=False)


def test_reader_accepts_core_native_gwas_tsv(tmp_path):
    _write_gwas_tsv(tmp_path / "P1", ["s3", "s1", "s2"], [0.3, 0.1, 0.2], [1.0, 2.0, 3.0])
    (tmp_path / "P1" / "threshold.txt").write_text("1.5e-05\n")  # clean-core name

    top = read_marginals(tmp_path, ["P1"], top_k=2)
    assert top["P1"] == ["s1", "s2"]          # sorted by PValue ascending

    thr = read_marginal_thresholds(tmp_path, ["P1"])
    assert abs(thr["P1"] - 1.5e-05) < 1e-12

    stats = read_marginal_stats(tmp_path, ["P1"], snp_ids=["s1"])
    assert stats["P1"]["s1"] == (0.1, 2.0)    # (PValue, SnpWeight)


def test_reader_prefers_first_assoc_over_gwas_tsv(tmp_path):
    # when both layouts are present the lux-native first_assoc wins, so existing
    # marginal dirs (and the epi parity fixture) read exactly as before
    d = tmp_path / "P1"
    d.mkdir()
    pq.write_table(pa.table({"SNP": ["a"], "PValue": [0.01], "SnpWeight": [9.0]}),
                   d / "P1.first_assoc.parquet")
    _write_gwas_tsv(d, ["z"], [0.5], [0.0])
    assert read_marginals(tmp_path, ["P1"], top_k=1)["P1"] == ["a"]


def test_missing_threshold_is_nan(tmp_path):
    _write_gwas_tsv(tmp_path / "P1", ["s1"], [0.1], [1.0])  # no threshold sidecar
    import math
    assert math.isnan(read_marginal_thresholds(tmp_path, ["P1"])["P1"])
