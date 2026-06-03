"""
unit cover for the watch TUI (src/fasterlmm/watch.py)
all of it is cpu-testable with synthetic status dicts -- the formatters, the per-shard aggregation, the
panel builders and the single-vs-multi render branch -- so nothing here spins up a rich Live or a subprocess.
the formatters get value checks, the panels get rendered to plain text and probed for the key fields
"""
from __future__ import annotations

import io
import json
import time

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from fasterlmm.watch import (
    _aggregate, _all_done, _build_overall_panel, _build_shard_panel,
    _discover_shards, _fmt_age, _fmt_eta, _fmt_huge, _fmt_mem, _fmt_rate,
    _read_payload, _render,
)


def _plain(renderable, width: int = 100) -> str:
    """render a rich object to plain (uncoloured) text so a test can grep the fields out of it"""
    buf = io.StringIO()
    Console(file=buf, width=width, no_color=True, legacy_windows=False).print(renderable)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# formatters
# ---------------------------------------------------------------------------

def test_fmt_eta_units():
    """h/m/s compact string, none / non-positive / nan all collapse to a dash"""
    assert _fmt_eta(None) == "-"
    assert _fmt_eta(0) == "-"
    assert _fmt_eta(-5) == "-"
    assert _fmt_eta(float("nan")) == "-"
    assert _fmt_eta(45) == "45s"
    assert _fmt_eta(90) == "1m30s"
    assert _fmt_eta(3661) == "1h01m01s"


def test_fmt_rate_per_s_and_per_min():
    """fast runs read pheno/s, slow ones fall back to pheno/min, dead/none is a dash"""
    assert _fmt_rate(None) == "-"
    assert _fmt_rate(0) == "-"
    assert _fmt_rate(2.0).endswith("pheno/s")
    out = _fmt_rate(0.5)
    assert out.endswith("pheno/min")
    assert "30" in out  # 0.5/s = 30/min


def test_fmt_mem_gb_mb():
    """GB above 1024 MB, MB below, dash when missing"""
    assert _fmt_mem(None) == "-"
    assert _fmt_mem(0) == "-"
    assert _fmt_mem(512).endswith("MB")
    assert _fmt_mem(2048).endswith("GB")


def test_fmt_huge_scaling():
    """compact K / M / B / T for the big Wald-op counters"""
    assert _fmt_huge(None) == "-"
    assert _fmt_huge(0) == "-"
    assert _fmt_huge(950) == "950"
    assert _fmt_huge(1500).endswith("K")
    assert _fmt_huge(3.4e6).endswith("M")
    assert _fmt_huge(5.7e9).endswith("B")
    assert _fmt_huge(9.0e12).endswith("T")


def test_fmt_age_freshness():
    """a fresh ts reads now, older ones read in seconds then minutes, none is a question mark"""
    assert _fmt_age(None) == "?"
    assert _fmt_age(time.time()) == "now"
    assert _fmt_age(time.time() - 30).endswith("s ago")
    assert _fmt_age(time.time() - 300).endswith("m ago")


# ---------------------------------------------------------------------------
# status snapshot reading + shard discovery
# ---------------------------------------------------------------------------

def test_read_payload_roundtrip_and_missing(tmp_path):
    """a written json reads back, a missing or half-written file comes back None not an exception"""
    p = tmp_path / "status.json"
    p.write_text(json.dumps({"state": "scanning", "phenos_done": 3}))
    assert _read_payload(p) == {"state": "scanning", "phenos_done": 3}
    assert _read_payload(tmp_path / "nope.json") is None
    (tmp_path / "half.json").write_text("{not valid json")
    assert _read_payload(tmp_path / "half.json") is None


def test_discover_shards_globs_and_skips_tmp(tmp_path):
    """status.shard*.json get picked up keyed by index, the .tmp atomic-rename siblings are skipped"""
    (tmp_path / "status.shard0.json").write_text(json.dumps({"state": "scanning"}))
    (tmp_path / "status.shard1.json").write_text(json.dumps({"state": "done"}))
    (tmp_path / "status.shard0.json.tmp").write_text("{half")
    (tmp_path / "status.json").write_text(json.dumps({"state": "dispatch"}))  # parent, not a shard
    shards = _discover_shards(tmp_path)
    assert set(shards) == {0, 1}
    assert shards[1]["state"] == "done"


def test_all_done_state_machine():
    """parent-done OR every-shard-done flips it, an empty set or a still-scanning shard does not"""
    assert _all_done({"state": "done"}, {}) is True
    assert _all_done(None, {0: {"state": "done"}, 1: {"state": "done"}}) is True
    assert _all_done(None, {0: {"state": "done"}, 1: {"state": "scanning"}}) is False
    assert _all_done({"state": "scanning"}, {}) is False
    assert _all_done(None, {}) is False


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def test_aggregate_rolls_shards_up():
    """phenos sum across shards, elapsed is the slowest shard, rss sums, rate is done over wall"""
    states = [
        {"phenos_done": 10, "phenos_total": 50, "elapsed_s": 5.0, "N": 100, "M": 200,
         "n_perm": 100, "loco": True, "device": "cuda:0", "rss_mb": 1000, "peak_rss_mb": 1200,
         "writes_pending": 2, "state": "scanning"},
        {"phenos_done": 6, "phenos_total": 50, "elapsed_s": 8.0, "N": 100, "M": 200,
         "n_perm": 100, "loco": True, "device": "cuda:1", "rss_mb": 900, "peak_rss_mb": 1100,
         "writes_pending": 1, "state": "scanning"},
    ]
    agg = _aggregate(states)
    assert agg["phenos_done"] == 16
    assert agg["phenos_total"] == 100
    assert agg["elapsed_s"] == 8.0  # slowest shard
    assert agg["rss_mb"] == 1900
    assert agg["peak_rss_mb"] == 2300
    assert agg["writes_pending"] == 3
    assert agg["N"] == 100 and agg["M"] == 200
    assert abs(agg["rate"] - 16 / 8.0) < 1e-9
    assert agg["eta_s"] is not None and agg["eta_s"] > 0
    assert agg["state"] == "scanning"


def test_aggregate_all_done_and_zero_elapsed():
    """state is done only when every shard is, and a zero-elapsed snapshot leaves rate/eta as None"""
    agg = _aggregate([{"phenos_done": 0, "phenos_total": 5, "elapsed_s": 0.0, "state": "done"},
                      {"phenos_done": 0, "phenos_total": 5, "elapsed_s": 0.0, "state": "done"}])
    assert agg["state"] == "done"
    assert agg["rate"] is None
    assert agg["eta_s"] is None


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

def test_overall_panel_renders_key_fields():
    """the headline panel carries the title, a progress bar, the phenos count and the Wald-op line"""
    agg = _aggregate([{"phenos_done": 16, "phenos_total": 100, "elapsed_s": 8.0, "N": 100,
                       "M": 200, "n_perm": 100, "loco": True, "device": "cuda:0",
                       "rss_mb": 1900, "state": "scanning"}])
    panel = _build_overall_panel(agg, n_shards=2, run_name="myrun", width=100)
    assert isinstance(panel, Panel)
    text = _plain(panel)
    assert "FaST-ER-LMM" in text
    assert "myrun" in text
    assert "progress" in text
    assert "phenos" in text
    assert "Wald ops" in text  # M and total and n_perm all present


def test_shard_panel_loco_strip_and_drain_and_gpu():
    """a scanning shard shows the loco dot strip, the gpu-memory line, and the drain readout when writing"""
    scanning = {"state": "scanning", "phenos_done": 5, "phenos_total": 20, "elapsed_s": 4.0,
                "device": "cuda:0", "chroms_total": 16, "chroms_done": 7,
                "gpu_alloc_mb": 8000, "gpu_total_mb": 32000, "gpu_peak_alloc_mb": 9000,
                "rss_mb": 1500, "peak_rss_mb": 1600, "ts": time.time()}
    text = _plain(_build_shard_panel(0, scanning, n_shards=2, panel_width=60))
    assert "GPU 0" in text
    assert "LOCO" in text and "chr 7/16" in text
    assert "GPU mem" in text

    writing = {"state": "writing", "phenos_done": 20, "phenos_total": 20, "elapsed_s": 30.0,
               "device": "cuda:0", "writes_pending": 4, "write_cpu_frac": 0.4,
               "write_mb_s": 120.0, "ts": time.time()}
    dtext = _plain(_build_shard_panel(1, writing, n_shards=2, panel_width=60))
    assert "drain" in dtext
    assert "MB/s" in dtext


# ---------------------------------------------------------------------------
# render branch
# ---------------------------------------------------------------------------

def test_render_multi_shard_is_group():
    """with discovered shards the frame is a Group (overall panel + the shard views)"""
    shards = {0: {"state": "scanning", "phenos_done": 5, "phenos_total": 20, "elapsed_s": 4.0,
                  "N": 100, "M": 200, "n_perm": 100, "device": "cuda:0"},
              1: {"state": "scanning", "phenos_done": 4, "phenos_total": 20, "elapsed_s": 4.0,
                  "N": 100, "M": 200, "n_perm": 100, "device": "cuda:1"}}
    out = _render(parent=None, shards=shards, run_name="r", width=140)
    assert isinstance(out, Group)


def test_render_single_pane_from_parent():
    """a single-GPU run (parent IS the worker, carries phenos_total) renders the one-pane Group"""
    parent = {"state": "scanning", "phenos_done": 2, "phenos_total": 10, "elapsed_s": 3.0,
              "N": 50, "M": 100, "n_perm": 10, "device": "cpu"}
    out = _render(parent=parent, shards={}, run_name="r", width=100)
    assert isinstance(out, Group)


def test_render_dispatch_and_waiting():
    """the dispatch placeholder is a Panel, and a totally empty state is the waiting Text"""
    out = _render(parent={"state": "dispatch", "n_gpu": 2}, shards={}, run_name="r", width=100)
    assert isinstance(out, Panel)
    waiting = _render(parent=None, shards={}, run_name="r", width=100)
    assert isinstance(waiting, Text)
