"""live TUI for an in-flight `gwas-epi` (tier-2 epistasis) run.

epi has different shape from single-K gwas / gxe: per-pheno top-K heaps
only drain at the very end, so phenos_done stays at 0 until the writer
fires; the real progress signal is inner_done / inner_total (anchor SNP
× test-block cells). 2D LDCO tile grid (chrom_m × chrom_j) replaces the
1D LOCO chrom strip the og + gxe watchers render

reads tile-done set from the .jsonl event log (the snapshot's `recent`
ring buffer only holds 32 events; at 256 tiles the early ones get
dropped). cached by mtime so polling cost stays flat

usage:
    epi-watch                         auto-discover newest .status.json (cwd, cwd/results, parent results/, ~/fasterlmm/results)
    epi-watch <status-file>           poll every second
    epi-watch <status-file> --once    one snapshot then exit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fasterlmm._tui import PALETTE, shard_color
from fasterlmm.watch import _fmt_eta, _fmt_huge

# the watch.* private helper layer + the _tui GPU helpers diverged out of the
# public core; reuse lux's vendored copies. PALETTE/shard_color + _fmt_eta/
# _fmt_huge still come straight from the public core. the dev import also pulled
# gpu_mem_panel/header_panel/rate_panel/progress_bar — all dead here (panels are
# built inline), dropped exactly as the gxe watcher did.
from fasterlmm_lux._watchkit import (
    _autodiscover_status,
    _build_recent_panel,
    _gradient_bar,
    _read_gpu_stats,
    _read_snapshot,
    _resolve_shard_paths,
    _route_dir_arg,
    _short_label,
    _stage_text,
)


def _fmt_cells_per_s(rate: float | None) -> str:
    if rate is None or rate <= 0:
        return "-"
    if rate >= 10:
        return f"{rate:,.1f} cell/s"
    if rate >= 0.1:
        return f"{rate:.2f} cell/s"
    return f"{rate*60:.1f} cell/min"


_TILE_CACHE: dict[str, tuple[float, set[tuple[int, int]], tuple[int, int] | None]] = {}


def _read_tiles_from_jsonl(
    status_path: Path
) -> tuple[set[tuple[int, int]], tuple[int, int] | None]:
    """parsing the .jsonl event log for the full tile-done set + the
    most-recent in-flight tile. snapshot's `recent` only carries 32
    events so at 256 tiles we'd lose the early ones; .jsonl has every
    event ever written. cached by mtime so we only re-read on change

    returns (set of (chrom_m, chrom_j) done, optional in_flight cell)
    """
    jsonl = status_path.with_suffix(status_path.suffix + ".jsonl")
    if not jsonl.exists():
        return set(), None
    try:
        mtime = jsonl.stat().st_mtime
    except (FileNotFoundError, OSError):
        return set(), None
    cached = _TILE_CACHE.get(str(jsonl))
    if cached and cached[0] == mtime:
        return cached[1], cached[2]

    done: set[tuple[int, int]] = set()
    last_running: tuple[int, int] | None = None
    try:
        with open(jsonl) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stage = ev.get("stage", "")
                state = ev.get("stage_state", "")
                if stage == "tier2_tile" and state == "done":
                    cm = ev.get("chrom_m")
                    cj = ev.get("chrom_j")
                    if cm is not None and cj is not None:
                        done.add((int(cm), int(cj)))
                elif stage.startswith("ldco_kinship_") and state == "running":
                    # stage names look like ldco_kinship_3.0_2.0; grab the trailing two coords
                    parts = stage[len("ldco_kinship_"):].split("_")
                    if len(parts) == 2:
                        try:
                            last_running = (int(float(parts[0])),
                                            int(float(parts[1])))
                        except ValueError:
                            pass
    except OSError:
        return set(), None

    # in-flight = last_running tile that hasn't fired its tier2_tile done yet
    in_flight = last_running if (last_running and last_running not in done) else None
    _TILE_CACHE[str(jsonl)] = (mtime, done, in_flight)
    return done, in_flight


def _aggregate_epi(states: list[dict[str, Any]]) -> dict[str, Any]:
    """aggregating per-shard epi snapshots. inner_done = anchor × test-block cells; this is the real progress lever (phenos_done stays 0 until the very end when heaps drain)"""
    inner_done = sum(int(s.get("inner_done", 0)) for s in states)
    inner_total = sum(int(s.get("inner_total", 0)) for s in states)
    phenos_total = sum(int(s.get("phenos_total", 0)) for s in states)
    elapsed = max((s.get("elapsed_s", 0) for s in states), default=0)
    started_at = min((s.get("started_at", time.time()) for s in states),
                     default=time.time())
    cells_per_s = (inner_done / elapsed) if (elapsed > 0 and inner_done > 0) else None
    eta = None
    if cells_per_s and inner_total > inner_done:
        eta = (inner_total - inner_done) / cells_per_s
    n_strains = next((s.get("n_strains") for s in states if s.get("n_strains")), None)
    n_variants = next((s.get("n_variants") for s in states if s.get("n_variants")), None)
    n_marginals = next((s.get("n_marginals_unique") for s in states
                        if s.get("n_marginals_unique") is not None), None)
    top_k_pair = next((s.get("top_k_pair") for s in states if s.get("top_k_pair")), None)
    p_threshold = next((s.get("p_threshold") for s in states
                        if s.get("p_threshold") is not None), None)
    all_anchors = next((s.get("all_anchors") for s in states
                        if s.get("all_anchors") is not None), None)
    n_perm = next((s.get("n_perm") for s in states if s.get("n_perm") is not None), None)
    perm_q = next((s.get("perm_quantile") for s in states
                   if s.get("perm_quantile") is not None), None)
    maf_floor = next((s.get("maf_floor") for s in states
                      if s.get("maf_floor") is not None), None)
    rint = next((s.get("rint") for s in states if s.get("rint") is not None), None)
    device = next((s.get("device") for s in states if s.get("device")), None)
    chrom_m_total = next((s.get("chrom_m_total") for s in states
                          if s.get("chrom_m_total")), None)
    chrom_j_total = next((s.get("chrom_j_total") for s in states
                          if s.get("chrom_j_total")), None)
    return {
        "inner_done": inner_done, "inner_total": inner_total,
        "phenos_total": phenos_total,
        "started_at": started_at, "elapsed_s": elapsed,
        "cells_per_s": cells_per_s, "eta_s": eta,
        "n_strains": n_strains, "n_variants": n_variants,
        "n_marginals": n_marginals, "top_k_pair": top_k_pair,
        "p_threshold": p_threshold, "all_anchors": all_anchors,
        "n_perm": n_perm, "perm_quantile": perm_q,
        "maf_floor": maf_floor, "rint": rint,
        "device": device,
        "chrom_m_total": chrom_m_total, "chrom_j_total": chrom_j_total}


def _build_overall_panel(agg: dict[str, Any], n_shards: int,
                         states: list[dict[str, Any]] | None = None,
                         width: int = 100,
                         run_name: str = "") -> Panel:
    inner_done = agg["inner_done"]
    inner_total = agg["inner_total"]
    phenos_total = agg["phenos_total"]
    elapsed = agg["elapsed_s"]
    eta = agg["eta_s"]
    rate = agg["cells_per_s"]
    n_strains = agg.get("n_strains")
    n_variants = agg.get("n_variants")
    n_marginals = agg.get("n_marginals")
    top_k_pair = agg.get("top_k_pair")
    p_threshold = agg.get("p_threshold")
    all_anchors = agg.get("all_anchors")
    n_perm = agg.get("n_perm")
    perm_q = agg.get("perm_quantile")
    maf_floor = agg.get("maf_floor")
    rint = agg.get("rint")
    device = agg.get("device")

    states = states or []
    s0 = states[0] if states else {}
    # the dispatcher passes run_name from the status file's outdir; fall
    # back to outdir/prefix from snapshot only if dispatcher didn't pass
    # anything. drop the body line: we already show run_name in the title
    if not run_name:
        prefix = s0.get("outdir") or s0.get("prefix")
        run_name = Path(str(prefix)).name if prefix else ""

    body = Text()

    # progress is anchor x test-block cells, NOT phenos. phenos_done stays at 0
    # till the writer drains the heaps so it makes a useless bar
    pct = (inner_done / inner_total) if inner_total else 0
    bar_w = max(20, min(96, width - 28))
    body.append("progress    ", style="dim")
    body.append_text(_gradient_bar(pct, width=bar_w))
    body.append(f"  {pct*100:5.1f}%", style="bold bright_white")
    body.append("\n\n")

    def _kv(label, val, val_style="bright_white", pad=12):
        body.append(label.ljust(pad), style="dim")
        body.append(val, style=val_style)

    cell_str = (f"{inner_done:,} / {inner_total:,}"
                if inner_total else f"{inner_done:,}")
    _kv("cells", cell_str)
    body.append("    ")
    _kv("phenos", f"{phenos_total:,}")
    body.append("\n")

    _kv("elapsed", _fmt_eta(elapsed) if elapsed else "-")
    body.append("    ")
    _kv("eta", _fmt_eta(eta), "bright_yellow")
    body.append("\n")

    _kv("rate", _fmt_cells_per_s(rate), "bright_green")
    body.append("    ")
    dataset_str = (f"{n_strains:,} × {n_variants:,}"
                   if n_strains and n_variants else "?")
    _kv("N × M", dataset_str)
    body.append("\n")

    if n_marginals:
        anchor_kind = "all-pairs" if all_anchors else "unique"
        buf_kind = (f"p<{p_threshold:g} unbounded"
                    if p_threshold is not None
                    else f"top-{top_k_pair} pair-heap"
                    if top_k_pair else "?")
        _kv("anchors", f"{n_marginals:,} {anchor_kind} × {buf_kind}")
        body.append("\n")

    # Wald ops at tier-2 scale: M_test * P_real * (1 + n_perm) per anchor SNP, summed over anchors. inner_total counts (anchor × chrom_j) cells, each cell scans len(j_block) test SNPs. without per-tile chrom_j sizes we approximate as (inner_done * mean_M_per_cell) but for the rate display we just scale by M / inner_total
    if n_variants and inner_total and inner_done and n_perm is not None:
        cols = 1 + n_perm
        # each cell scans (M / chrom_j_total) test SNPs on average against P_real * cols pheno-cols
        wald_total = n_variants * phenos_total * cols
        wald_done = int(wald_total * inner_done / inner_total)
        wald_rate = (wald_done / elapsed) if (elapsed and wald_done) else None
        _kv("Wald ops", f"{_fmt_huge(wald_done)} / {_fmt_huge(wald_total)}")
        body.append("    ")
        _kv("ops/s", _fmt_huge(wald_rate) + "/s" if wald_rate else "-",
            "bright_green")
        body.append("\n")

    config_bits: list[str] = []
    if n_perm is not None:
        if perm_q is not None and n_perm > 0:
            config_bits.append(f"{n_perm} perms · q={perm_q}")
        else:
            config_bits.append(f"{n_perm} perms")
    if maf_floor is not None and maf_floor > 0:
        config_bits.append(f"MAF≥{maf_floor}")
    if rint:
        config_bits.append("RINT")
    config_bits.append("LDCO")
    if device:
        config_bits.append(str(device))
    _kv("GPUs", str(n_shards))
    if config_bits:
        body.append("    ")
        _kv("config", "  ·  ".join(config_bits))

    title = Text("FaST-ER-LMM ", style=f"bold {PALETTE['primary']}")
    title.append("epi ", style=f"bold {PALETTE['accent']}")
    if run_name:
        title.append(f"· {run_name} ", style=PALETTE["label"])
    title.append("· live", style="dim")
    return Panel(body, title=title, border_style=PALETTE["primary"],
                 padding=(1, 2), title_align="left")


_STATE_PENDING = 0
_STATE_DONE = 1
_STATE_IN_FLIGHT = 2

# 9-entry glyph table for paired (top, bottom) cm cells. each char encodes
# two cm rows in one line via Unicode half-blocks: "▀" = top filled, "▄" =
# bottom filled, "█" = both filled. mixed colors (top vs bottom different)
# use "fg on bg" so the half-block's filled half is fg and the empty half
# is bg
_PAIR_GLYPH = {
    (_STATE_PENDING,   _STATE_PENDING):   ("·", "grey50"),
    (_STATE_PENDING,   _STATE_DONE):      ("▄", "green"),
    (_STATE_PENDING,   _STATE_IN_FLIGHT): ("▄", "yellow"),
    (_STATE_DONE,      _STATE_PENDING):   ("▀", "green"),
    (_STATE_DONE,      _STATE_DONE):      ("█", "green"),
    (_STATE_DONE,      _STATE_IN_FLIGHT): ("▄", "yellow on green"),
    (_STATE_IN_FLIGHT, _STATE_PENDING):   ("▀", "yellow"),
    (_STATE_IN_FLIGHT, _STATE_DONE):      ("▀", "yellow on green"),
    (_STATE_IN_FLIGHT, _STATE_IN_FLIGHT): ("█", "yellow"),
}


def _cell_state(cell: tuple[int, int] | None,
                done: set[tuple[int, int]],
                in_flight: tuple[int, int] | None) -> int:
    if cell is None:
        return _STATE_PENDING
    if cell == in_flight:
        return _STATE_IN_FLIGHT
    if cell in done:
        return _STATE_DONE
    return _STATE_PENDING


def _build_tile_grid(
    cm_values: list[int], cj_total: int,
    done: set[tuple[int, int]],
    in_flight: tuple[int, int] | None) -> Text:
    """2D heatmap of (chrom_m × chrom_j) tile state, half-height: pairs
    adjacent cm rows via Unicode half-blocks (▀ top, ▄ bottom, █ both,
    · neither). on yeast (16 × 16) this collapses the grid from 17 lines
    to ~9 — single-shard panels were dominated by the grid height before.

    wide spaced format only for tiny cj (≤ 6); otherwise compact 1-char
    cells with tick marks every 5 columns. cm rows that have no progress
    yet are dropped (cm_values is pre-filtered to chromosomes with done /
    in-flight tiles)
    """
    grid = Text()
    spaced = cj_total <= 6
    if spaced:
        header = "      " + " ".join(f"{j+1:>2}" for j in range(cj_total))
    else:
        # mark every 5th column with the digit, rest as fine dots, so the
        # eye can quickly find chrom 5/10/15
        header = "      " + "".join(
            str(j + 1) if (j + 1) % 5 == 0 else "·"
            for j in range(cj_total))
    grid.append(header + "\n", style="dim")

    # pair adjacent cm rows; odd count → last pair has bottom=None (top-only)
    cm_values = sorted(cm_values)
    pairs: list[tuple[int, int | None]] = []
    i = 0
    while i < len(cm_values):
        top = cm_values[i]
        bot = cm_values[i + 1] if i + 1 < len(cm_values) else None
        pairs.append((top, bot))
        i += 2

    for top, bot in pairs:
        label = f"{top}+{bot}" if bot is not None else f"{top}"
        grid.append(label.rjust(4) + "  ", style="dim")
        for j_idx in range(cj_total):
            cj = j_idx + 1
            top_state = _cell_state((top, cj), done, in_flight)
            bot_state = _cell_state((bot, cj) if bot is not None else None,
                                    done, in_flight)
            char, style = _PAIR_GLYPH[(top_state, bot_state)]
            if spaced:
                grid.append(" " + char, style=style)
                grid.append(" ", style="")
            else:
                grid.append(char, style=style)
        grid.append("\n")
    return grid


def _build_shard_panel(path: Path, s: dict[str, Any], idx: int, n: int,
                       panel_width: int | None = None) -> Panel:
    inner_done = int(s.get("inner_done", 0))
    inner_total = int(s.get("inner_total", 0))
    elapsed = s.get("elapsed_s") or 0
    rate = (inner_done / elapsed) if (elapsed > 0 and inner_done > 0) else None
    upd = s.get("updated_at", time.time())
    age = time.time() - upd
    age_str = "now" if age < 1 else (
        f"{int(age)}s ago" if age < 60 else f"{int(age/60)}m ago")
    last_tile_elapsed = s.get("tile_elapsed_s")
    last_cm = s.get("chrom_m")
    last_cj = s.get("chrom_j")

    shard_rank = s.get("shard_rank")
    # prefer GPU fields from the snapshot itself (recorded on the compute
    # node by gwas-epi); fall back to local nvidia-smi only when the snapshot
    # is silent and the watcher is running on the same box as the run
    snap_gpu_name = s.get("gpu_name")
    snap_gpu_total = s.get("gpu_total_mb")
    gpu_stats = None
    if snap_gpu_name and snap_gpu_total:
        from fasterlmm_lux._watchkit import GpuStats
        gpu_stats = GpuStats(name=snap_gpu_name, total_mb=float(snap_gpu_total))
    elif s.get("device") == "cuda" and shard_rank is not None:
        gpu_stats = _read_gpu_stats(int(shard_rank))

    body = Text()
    body.append_text(_stage_text(s.get("stage"), s.get("stage_state")))
    body.append("\n")

    bar_w = 18 if panel_width is None else max(8, min(50, panel_width - 52))
    pf = (inner_done / inner_total) if inner_total else 0
    body.append("cells    ", style="dim")
    body.append_text(_gradient_bar(pf, bar_w))
    body.append(f"  {inner_done:,}", style="bright_white")
    body.append(f"/{inner_total:,}", style="dim")
    if last_tile_elapsed:
        body.append(f"   last tile  {float(last_tile_elapsed):.1f}s",
                    style="dim")
    body.append("\n")

    # tile grid pulled from the .jsonl event log (snapshot.recent only holds 32 events, would lose early tiles at scale)
    done_set, in_flight = _read_tiles_from_jsonl(path)
    cj_total = int(s.get("chrom_j_total", 0) or 0)
    if cj_total and (done_set or in_flight or last_cm):
        cm_set = {cm for cm, _ in done_set}
        if in_flight:
            cm_set.add(in_flight[0])
        if last_cm is not None:
            try:
                cm_set.add(int(last_cm))
            except (TypeError, ValueError):
                pass
        cm_values = sorted(cm_set)
        body.append_text(_build_tile_grid(cm_values, cj_total, done_set, in_flight))
        if in_flight:
            body.append(f"in-flight tile (cm={in_flight[0]}, cj={in_flight[1]})",
                        style="bright_yellow")
            body.append("\n")
    else:
        body.append("(no tile events yet)\n", style="dim")

    eta_s = ((inner_total - inner_done) / rate
             if rate and inner_total and inner_total > inner_done else None)
    started_at = s.get("started_at")
    started_str = (time.strftime("%H:%M:%S", time.localtime(started_at))
                   if started_at else "-")
    jobid = s.get("slurm_job_id") or "-"

    body.append("rate ", style="dim")
    body.append(_fmt_cells_per_s(rate), style="bright_green")
    body.append("    eta ", style="dim")
    body.append(_fmt_eta(eta_s), style="bright_yellow")
    body.append("    last ", style="dim")
    body.append(age_str, style="dim")
    body.append("\n")
    body.append("job ", style="dim")
    body.append(str(jobid), style="bright_white")
    body.append("    started ", style="dim")
    body.append(started_str, style="bright_white")
    if gpu_stats is not None and gpu_stats.total_mb:
        body.append(f"    gpu{shard_rank} ", style="dim")
        name = (gpu_stats.name or "?").replace("NVIDIA ", "").replace("Tesla ", "")
        body.append(name, style="bright_white")
        body.append(f" ({gpu_stats.total_mb / 1024:.0f} GB)", style="dim")

    label = "shard 0" if n == 1 else _shard_label_from_path(path, idx)
    color = shard_color(idx)
    title = Text(label, style=f"bold {color}")
    return Panel(body, title=title, border_style=color,
                 padding=(0, 1), title_align="left")


def _shard_label_from_path(path: Path, idx: int) -> str:
    name = path.name
    if ".shard" in name:
        try:
            return f"shard {int(name.rsplit('.shard', 1)[1])}"
        except ValueError:
            pass
    return f"shard {idx}"


def _build_compact_table(
    states: list[tuple[Path, dict[str, Any]]], width: int) -> Panel:
    """one-row-per-shard table. swapped in when n >= 3 so the watcher fits
    on screen with many GPUs in flight. drops the tile grid and per-shard
    panel decoration; keeps progress / rate / GPU / eta / last-tile / age.
    eta is per-shard, derived from this shard's own cells/s vs remaining
    inner cells. RSS dropped — the pid is the worker-host pid, /proc reads
    here are on the watcher host so it never resolved"""
    tbl = Table.grid(padding=(0, 1), expand=True)
    tbl.add_column(justify="left", no_wrap=True)  # label
    tbl.add_column(justify="right", no_wrap=True) # jobid
    tbl.add_column(justify="left", no_wrap=True)  # gpu
    tbl.add_column(justify="left")                # bar
    tbl.add_column(justify="right", no_wrap=True) # cells
    tbl.add_column(justify="right", no_wrap=True) # rate
    tbl.add_column(justify="right", no_wrap=True) # eta
    tbl.add_column(justify="right", no_wrap=True) # started
    tbl.add_column(justify="right", no_wrap=True) # tile
    tbl.add_column(justify="right", no_wrap=True) # last
    bar_w = max(12, min(36, width - 110))

    # header
    head = lambda s: Text(s, style="dim")
    tbl.add_row(head("shard"), head("jobid"), head("gpu"), head("progress"),
                head("cells"), head("rate"), head("eta"),
                head("started"), head("tile"), head("last"))

    for i, (p, s) in enumerate(states):
        inner_done = int(s.get("inner_done", 0))
        inner_total = int(s.get("inner_total", 0))
        elapsed = s.get("elapsed_s") or 0
        rate = (inner_done / elapsed) if (elapsed > 0 and inner_done > 0) else None
        # per-shard eta: remaining cells / this-shard cells-per-s. None when
        # rate is missing or inner_total isn't known yet
        eta_s = ((inner_total - inner_done) / rate
                 if rate and inner_total and inner_total > inner_done else None)
        pf = (inner_done / inner_total) if inner_total else 0
        upd = s.get("updated_at", time.time())
        age = time.time() - upd
        age_str = "now" if age < 1 else (
            f"{int(age)}s" if age < 60 else
            f"{int(age/60)}m" if age < 3600 else
            f"{int(age/3600)}h")
        # stuck heuristic: heartbeat age > 5min flags red
        age_style = "red" if age > 300 else "dim"
        last_cm = s.get("chrom_m")
        last_cj = s.get("chrom_j")
        tile_str = (f"{int(last_cm)}:{int(last_cj)}"
                    if last_cm is not None and last_cj is not None else "-")

        snap_gpu_name = s.get("gpu_name") or "?"
        gpu_short = (str(snap_gpu_name)
                     .replace("NVIDIA ", "").replace("Tesla ", "")
                     .replace("-PCIE-", " "))
        if len(gpu_short) > 18:
            gpu_short = gpu_short[:18]

        label = _shard_label_from_path(p, i)
        color = shard_color(i)
        jobid = s.get("slurm_job_id") or "-"
        started_at = s.get("started_at")
        # HH:MM:SS local; lets the eye spot oldest runs without digging into
        # elapsed_s. matches the hub watcher convention
        started_str = (time.strftime("%H:%M", time.localtime(started_at))
                       if started_at else "-")
        tbl.add_row(
            Text(label, style=f"bold {color}"),
            Text(str(jobid), style="bright_white"),
            Text(gpu_short, style="bright_white"),
            _gradient_bar(pf, bar_w),
            Text(f"{inner_done:,}/{inner_total:,}", style="bright_white"),
            Text(_fmt_cells_per_s(rate), style="bright_green"),
            Text(_fmt_eta(eta_s), style="bright_yellow"),
            Text(started_str, style="dim"),
            Text(tile_str, style="bright_yellow"),
            Text(age_str, style=age_style))

    n = len(states)
    title = Text(f"{n} shards (compact)", style="dim")
    return Panel(tbl, title=title, border_style=PALETTE["primary"],
                 padding=(0, 1), title_align="left")


def _build_shard_view(states: list[tuple[Path, dict[str, Any]]],
                      console_width: int | None = None):
    n = len(states)
    width = console_width or 120
    # >= 3 shards: per-shard panels stop fitting on screen. swap to a
    # one-row-per-shard table (no tile grid; just progress + rate + last
    # heartbeat age). full panels stay for n <= 2
    if n >= 3:
        return _build_compact_table(states, width)
    # tile grid is wide; only side-by-side when terminal is really wide
    side_by_side = (n > 1) and (width >= 80 * n)
    per_panel_width = (width // n) - 4 if side_by_side else width - 4
    panels = [_build_shard_panel(p, s, i, n, panel_width=per_panel_width)
              for i, (p, s) in enumerate(states)]
    if n == 1:
        return panels[0]
    if side_by_side:
        return Columns(panels, expand=True, equal=True, padding=(0, 1))
    return Group(*panels)


def _render(arg: str) -> Group | Text:
    paths = _resolve_shard_paths(arg)
    states_raw = [(p, _read_snapshot(p)) for p in paths]
    states = [(p, s) for p, s in states_raw if s is not None]
    if not states:
        return Text(f"waiting for status file(s) at {arg!r}…", style="yellow")
    bare = [s for _, s in states]
    agg = _aggregate_epi(bare)
    width = Console().size.width
    rn = states[0][0].parent.name if states else ""
    overall = _build_overall_panel(agg, len(states), states=bare, width=width,
                                   run_name=rn)
    shard_view = _build_shard_view(states, console_width=width)
    recent = _build_recent_panel(bare)
    return Group(overall, shard_view, recent)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="epi-watch",
        description="Live TUI dashboard for an in-flight `gwas-epi` "
                    "(tier-2 epistasis) run.")
    p.add_argument("status_file", nargs="?",
                   help="path to the --status-file passed to gwas-epi. "
                        "auto-globs <path>.shard* for sharded runs. pass a "
                        "directory and the watcher rglobs from there for the "
                        "newest .status.json. omit to auto-discover the newest "
                        ".status.json by walking cwd, cwd/results, and any "
                        "parent dir's results/ up to $HOME")
    p.add_argument("--search-root", default=None,
                   help="where to look when status_file is omitted. defaults to walking cwd, cwd/results, then any parent dir's results/ up to $HOME")
    p.add_argument("--interval", type=float, default=1.0,
                   help="refresh interval in seconds (default 1.0)")
    p.add_argument("--once", action="store_true",
                   help="render one snapshot and exit")
    p.add_argument("--save-svg", default=None,
                   help="render one frame as SVG to PATH and exit (implies --once)")
    p.add_argument("--screenshot-width", type=int, default=200,
                   help="terminal width for --save-svg (default 200)")
    p.add_argument("--no-follow", dest="follow", action="store_false",
                   help="don't switch to a newer .status.json that appears mid-watch")
    p.add_argument("--rediscover-interval", type=float, default=5.0,
                   help="seconds between rediscovery checks (default 5.0)")
    args = p.parse_args(argv)

    if args.save_svg:
        console = Console(record=True, width=args.screenshot_width, force_terminal=True)
        args.once = True
    else:
        console = Console()
    dir_routed = _route_dir_arg(args)
    auto_discovered = args.status_file is None
    if auto_discovered:
        args.status_file = _autodiscover_status(args.search_root, console)
        if args.status_file is None:
            return 2
    auto_discovered = auto_discovered or dir_routed
    if args.once:
        console.print(_render(args.status_file))
        if args.save_svg:
            console.save_svg(args.save_svg, title="epi-watch")
        return 0

    follow = auto_discovered and args.follow
    try:
        with Live(_render(args.status_file), console=console,
                  refresh_per_second=max(1.0, 1.0 / args.interval),
                  screen=False) as live:
            last_rediscover = time.time()
            while True:
                time.sleep(args.interval)
                if follow and (time.time() - last_rediscover) >= args.rediscover_interval:
                    last_rediscover = time.time()
                    new_target = _autodiscover_status(args.search_root, console, quiet=True)
                    if new_target and new_target != args.status_file:
                        console.print(f"[dim]switching to {_short_label(new_target)}[/dim]")
                        args.status_file = new_target
                live.update(_render(args.status_file))
    except KeyboardInterrupt:
        console.print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
