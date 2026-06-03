"""
Live TUI dashboard for a running fasterlmm gwas scan
Point it at the outdir and it builds the multi-GPU view: one overall panel (aggregate progress, rate, ETA, Wald ops, RAM) plus one panel per shard (per-GPU progress bar, rate, the live loco sweep, GPU memory).  Point it at a single status.json and it shows the one-pane view
The watcher and the runner are decoupled, the runner drops json status snapshots and the watcher just polls them, so a shared filesystem is all it needs
usage:
  fasterlmm watch <outdir>              dashboard, one panel per GPU shard
  fasterlmm watch <outdir/status.json>  single pane
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from fasterlmm._tui import PALETTE, progress_bar, shard_color

_SHARD_RE = re.compile(r"^status\.shard(\d+)\.json$")


# ---- status snapshot reading -------------------------------------------------

def _read_payload(path: Path) -> dict | None:
    """returning None on a missing file or a mid-rename half-write, the caller decides what to render in the placeholer"""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _discover_shards(outdir: Path) -> dict[int, dict]:
    """
    Globbing status.shard*.json in outdir, skipping the .tmp atomic-rename siblings that flash into existance for a millisecond during write_status
    Returns {shard_idx: payload}
    """
    out: dict[int, dict] = {}
    for p in sorted(outdir.glob("status.shard*.json")):
        m = _SHARD_RE.match(p.name)
        if not m:
            continue
        payload = _read_payload(p)
        if payload is not None:
            out[int(m.group(1))] = payload
    return out


def _all_done(parent: dict | None, shards: dict[int, dict]) -> bool:
    """
    Flagging done when the parent status.json says done, OR when every discovered shard says done and at least one shard exists
    Parent-side covers auto-dispatch (the parent writes the finalizer after the spawn join), shard-side covers an explicit --shard slurm-array where noone writes the parent
    """
    if parent and parent.get("state") == "done":
        return True
    if shards and all(s.get("state") == "done" for s in shards.values()):
        return True
    return False


# ---- formatters --------------------------------------------------------------

def _fmt_eta(seconds: float | None) -> str:
    """compact h/m/s string. None / non-positive / nan all collapse to a dash"""
    if seconds is None or seconds <= 0 or seconds != seconds:
        return "-"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _fmt_rate(rate: float | None) -> str:
    """phenos per second, falling back to per-minute when the run is slow"""
    if rate is None or rate <= 0:
        return "-"
    if rate >= 1.0:
        return f"{rate:,.1f} pheno/s"
    return f"{rate * 60:.1f} pheno/min"


def _fmt_mem(mb: float | None) -> str:
    """GB / MB from megabytes. a dash means the field wasn't in the snapshot (older runner, non-linux)"""
    if mb is None or mb <= 0:
        return "-"
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def _fmt_huge(n: float | None) -> str:
    """compact 1.2K / 3.4M / 5.7B / 9.0T formatter for the big Wald-op counters"""
    if n is None or n <= 0:
        return "-"
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f}{unit}"
    return f"{n:.0f}"


def _fmt_age(ts: float | None) -> str:
    """how stale the snapshot is. a shard that stops updating is the first sign somethings wrong"""
    if ts is None:
        return "?"
    age = time.time() - ts
    if age < 2:
        return "now"
    if age < 60:
        return f"{int(age)}s ago"
    return f"{int(age / 60)}m ago"


# ---- aggregation -------------------------------------------------------------

def _aggregate(states: list[dict[str, Any]]) -> dict[str, Any]:
    """rolling the per-shard snapshots up into one view. phenos sum across shards, elapsed is the slowest shard, rate is real phenos over wall time"""
    done = sum(int(s.get("phenos_done", 0)) for s in states)
    total = sum(int(s.get("phenos_total", 0)) for s in states)
    elapsed = max((s.get("elapsed_s", 0) or 0 for s in states), default=0)
    rate = (done / elapsed) if (elapsed > 0 and done > 0) else None
    eta = ((total - done) / rate) if (rate and total > done) else None

    def _first(key):
        return next((s.get(key) for s in states if s.get(key) is not None), None)

    def _sum(key):
        vals = [s.get(key) for s in states if s.get(key) is not None]
        return sum(vals) if vals else None

    return {
        "phenos_done": done, "phenos_total": total,
        "elapsed_s": elapsed, "rate": rate, "eta_s": eta,
        "N": _first("N"), "M": _first("M"),
        "n_perm": _first("n_perm"), "loco": _first("loco"), "device": _first("device"),
        "rss_mb": _sum("rss_mb"), "peak_rss_mb": _sum("peak_rss_mb"),
        "writes_pending": _sum("writes_pending"),
        "mode": _first("mode"),  # gwas leaves this unset, the title defaults it back to gwas
        "state": ("done" if states and all(s.get("state") == "done" for s in states)
                  else _first("state"))}


# ---- panels ------------------------------------------------------------------

def _build_overall_panel(agg: dict[str, Any], n_shards: int,
                         run_name: str, width: int) -> Panel:
    """the headline panel: aggregate progress bar plus the run-wide key numbers"""
    done = agg["phenos_done"]
    total = agg["phenos_total"]
    N, M = agg.get("N"), agg.get("M")
    n_perm = agg.get("n_perm")

    body = Text()
    frac = (done / total) if total else 0.0
    bar_w = max(20, min(96, width - 30))
    body.append("progress  ", style=PALETTE["muted"])
    body.append_text(progress_bar(done, total, bar_w))
    body.append(f"  {frac * 100:5.1f}%", style=f"bold {PALETTE['label']}")
    body.append("\n\n")

    def _kv(label, val, val_style=None, pad=10):
        body.append(label.ljust(pad), style=PALETTE["muted"])
        body.append(str(val), style=val_style or PALETTE["label"])

    _kv("phenos", f"{done:,} / {total:,}" if total else f"{done:,}")
    body.append("    ")
    _kv("N x M", f"{N:,} x {M:,}" if (N and M) else "?")
    body.append("\n")
    _kv("elapsed", _fmt_eta(agg["elapsed_s"]) if agg["elapsed_s"] else "-")
    body.append("    ")
    _kv("eta", _fmt_eta(agg["eta_s"]), PALETTE["warn"])
    body.append("\n")
    _kv("rate", _fmt_rate(agg["rate"]), PALETTE["success"])

    # Wald ops: every (variant, pheno-col) pair is one beta/SE solve, times (1 + n_perm) cols. the number lands in the trillions, feels-fast in a way pheno/s doesn't
    if M and total and n_perm is not None:
        cols = 1 + n_perm
        wald_done = M * done * cols
        wald_total = M * total * cols
        body.append("    ")
        _kv("Wald ops", f"{_fmt_huge(wald_done)} / {_fmt_huge(wald_total)}")
    body.append("\n")

    # RAM + writes lines stay unconditional so the panel keeps a fixed height -- a key the snapshot
    # doesnt carry yet just renders a dash rather than dropping the whole line out
    rss, peak = agg.get("rss_mb"), agg.get("peak_rss_mb")
    _kv("RAM", _fmt_mem(rss))
    body.append("    ")
    _kv("peak RAM", _fmt_mem(peak), PALETTE["warn"])
    if n_shards > 1:
        body.append(f"  (sum across {n_shards} shards)", style=PALETTE["muted"])
    body.append("\n")

    wp = agg.get("writes_pending")
    _kv("writes", f"{int(wp):,} pending" if wp else "-",
        PALETTE["warn"] if wp else PALETTE["muted"])
    body.append("\n")

    cfg = []
    if n_perm is not None:
        cfg.append(f"{n_perm} perms")
    loco = agg.get("loco")
    if loco is True:
        cfg.append("LOCO")
    elif loco is False:
        cfg.append("single-K")
    if agg.get("device"):
        cfg.append(str(agg["device"]))
    _kv("GPUs", n_shards)
    if cfg:
        body.append("    ")
        _kv("config", "  ·  ".join(cfg))

    title = Text("FaST-ER-LMM ", style=f"bold {PALETTE['primary']}")
    title.append(agg.get("mode") or "gwas", style=f"bold {PALETTE['accent']}")  # gwas (default) or extreme
    if run_name:
        title.append(f"  ·  {run_name}", style=PALETTE["label"])
    title.append("  ·  done" if agg.get("state") == "done" else "  ·  live", style=PALETTE["muted"])
    return Panel(body, title=title, border_style=PALETTE["primary"],
                 padding=(1, 2), title_align="left")


_STATE_GLYPH: dict[str, tuple[str, str]] = {
    "loading":  ("·", "muted"),
    "scanning": ("▶", "warn"),
    "writing":  ("▸", "warn"),
    "dry-run":  ("·", "muted"),
    "done":     ("✓", "success"),
}


def _build_shard_panel(idx: int, s: dict[str, Any], n_shards: int,
                       panel_width: int) -> Panel:
    """one panel per GPU shard: state, per-shard progress bar, rate, GPU memory"""
    state = s.get("state", "?")
    done = int(s.get("phenos_done", 0))
    total = int(s.get("phenos_total", 0))
    elapsed = s.get("elapsed_s") or 0
    rate = (done / elapsed) if (elapsed > 0 and done > 0) else None

    body = Text()
    glyph, gstyle = _STATE_GLYPH.get(state, ("·", "muted"))
    body.append(glyph + " ", style=PALETTE[gstyle])
    body.append(state, style=f"bold {PALETTE[gstyle]}")
    body.append("   ")
    body.append(str(s.get("device", "")), style=PALETTE["muted"])
    body.append("\n\n")

    bar_w = max(10, min(48, panel_width - 30))
    body.append("phenos  ", style=PALETTE["muted"])
    body.append_text(progress_bar(done, total, bar_w))
    body.append(f"  {done:,}", style=PALETTE["label"])
    body.append(f"/{total:,}", style=PALETTE["muted"])
    body.append("\n")

    # loco dot strip -- one dot per chromosome, filled as each leave-one-out fold lands.  drawn for
    # the whole of a loco run so the panel height stays fixed, it just reads full once the scan tips
    # into the write drain
    chroms_total = s.get("chroms_total")
    if chroms_total:
        chroms_total = int(chroms_total)
        chroms_done = (min(int(s.get("chroms_done", 0)), chroms_total)
                       if state == "scanning" else chroms_total)
        body.append("LOCO    ", style=PALETTE["muted"])
        body.append("●" * chroms_done, style=PALETTE["success"])
        body.append("·" * (chroms_total - chroms_done), style=PALETTE["muted"])
        body.append(f"  chr {chroms_done}/{chroms_total}", style=PALETTE["label"])
        body.append("\n")

    body.append("rate    ", style=PALETTE["muted"])
    body.append(_fmt_rate(rate), style=PALETTE["success"])
    body.append("    eta ", style=PALETTE["muted"])
    body.append(_fmt_eta(((total - done) / rate) if (rate and total > done) else None),
                style=PALETTE["warn"])
    body.append("    upd ", style=PALETTE["muted"])
    body.append(_fmt_age(s.get("ts")), style=PALETTE["muted"])

    # one fixed line for the writer side -- the drain readout while the bundle writes still land,
    # the pending count while scanning, a dash otherwise.  always present so nothing below it
    # jumps as the run crosses from scanning into the trailing write drain
    body.append("\n")
    cpu_frac = s.get("write_cpu_frac")
    pending = s.get("writes_pending")
    if cpu_frac is not None:
        # drain readout: throughput + cpu-busy fraction.  cpu-time over wall-time below one full
        # core, writes still pending, means the drain waits on the filesystem not on compute
        body.append("drain   ", style=PALETTE["muted"])
        mb_s = s.get("write_mb_s")
        if mb_s is not None:
            body.append(f"{mb_s:,.0f} MB/s", style=PALETTE["label"])
            body.append("   ")
        io_bound = cpu_frac < 1.0
        body.append(f"cpu {cpu_frac * 100:.0f}%",
                    style=PALETTE["warn"] if io_bound else PALETTE["success"])
        body.append("  (i/o bound)" if io_bound else "  (cpu bound)", style=PALETTE["muted"])
    elif pending:
        body.append("writes  ", style=PALETTE["muted"])
        body.append(f"{int(pending):,} pending", style=PALETTE["warn"])
    else:
        body.append("writes  ", style=PALETTE["muted"])
        body.append("-", style=PALETTE["muted"])

    # GPU memory: alloc / total + peak, straight from torch in the worker
    g_alloc = s.get("gpu_alloc_mb")
    g_total = s.get("gpu_total_mb")
    g_peak = s.get("gpu_peak_alloc_mb")
    if g_alloc is not None or g_total is not None:
        body.append("\n")
        body.append("GPU mem ", style=PALETTE["muted"])
        body.append(_fmt_mem(g_alloc), style=PALETTE["label"])
        body.append(" / ", style=PALETTE["muted"])
        body.append(_fmt_mem(g_total), style=PALETTE["muted"])
        if g_total and g_alloc:
            pct = min(1.0, g_alloc / g_total) * 100
            col = (PALETTE["fail"] if pct > 90
                   else PALETTE["warn"] if pct > 70 else PALETTE["success"])
            body.append(f"  {pct:.0f}%", style=col)
        if g_peak is not None:
            body.append("   peak ", style=PALETTE["muted"])
            body.append(_fmt_mem(g_peak), style=PALETTE["warn"])

    rss = s.get("rss_mb")
    if rss is not None:
        body.append("\n")
        body.append("RAM     ", style=PALETTE["muted"])
        body.append(_fmt_mem(rss), style=PALETTE["label"])
        if s.get("peak_rss_mb") is not None:
            body.append("   peak ", style=PALETTE["muted"])
            body.append(_fmt_mem(s["peak_rss_mb"]), style=PALETTE["warn"])

    color = shard_color(idx)
    title = Text(f"GPU {idx}", style=f"bold {color}")
    return Panel(body, title=title, border_style=color,
                 padding=(0, 1), title_align="left")


def _render(parent: dict | None, shards: dict[int, dict],
            run_name: str, width: int) -> Group | Panel | Text:
    """assembling the live frame from the parent manifest + discovered shards"""
    if shards:
        states = [shards[i] for i in sorted(shards)]
        agg = _aggregate(states)
        overall = _build_overall_panel(agg, len(states), run_name, width)
        n = len(states)
        side_by_side = n > 1 and width >= 58 * n
        per_w = (width // n - 4) if side_by_side else width - 6
        panels = [_build_shard_panel(i, shards[i], n, per_w) for i in sorted(shards)]
        shard_view = panels[0] if n == 1 else (
            Columns(panels, expand=True, equal=True) if side_by_side else Group(*panels))
        return Group(overall, shard_view)

    # no shard files: either a single-GPU run (parent IS the worker) or still dispatching
    if parent and ("phenos_total" in parent or parent.get("state") in
                   ("loading", "scanning", "done")):
        agg = _aggregate([parent])
        return Group(_build_overall_panel(agg, 1, run_name, width),
                     _build_shard_panel(0, parent, 1, width - 6))
    if parent and parent.get("state") == "dispatch":
        return Panel(Text("workers spawning, waiting for the first shard snapshot…",
                          style=PALETTE["warn"]),
                     title=Text("FaST-ER-LMM gwas  ·  live", style=f"bold {PALETTE['primary']}"),
                     border_style=PALETTE["primary"], padding=(1, 2), title_align="left")
    return Text("waiting for status file…", style=PALETTE["warn"])


def main() -> None:
    """
    Poll the status files under the given path and redraw the live frame til the run reports done
    A directory (or a not-yet-existing path whose parent already holds shard files) is the multi-GPU dashboard, a plain status.json is the single pane
    """
    parser = argparse.ArgumentParser(
        prog="fasterlmm watch",
        description="live TUI dashboard for a running fasterlmm gwas job, single or multi-GPU")
    parser.add_argument("path", help="outdir (multi-GPU dashboard) or a status.json file (single pane)")
    args = parser.parse_args()

    path = Path(args.path)
    # multi mode if path is a directory, or if it doesn't exist yet but its parent already has shard files (watcher launched before the parent finalizer wrote status.json)
    if path.is_dir():
        outdir, multi = path, True
    elif not path.exists() and path.parent.is_dir() and any(path.parent.glob("status.shard*.json")):
        outdir, multi = path.parent, True
    else:
        outdir, multi = path.parent, False
    run_name = outdir.name

    console = Console()
    with Live(refresh_per_second=2, console=console, screen=False) as live:
        while True:
            width = console.size.width
            if multi:
                shards = _discover_shards(outdir)
                parent = _read_payload(outdir / "status.json")
                live.update(_render(parent, shards, run_name, width))
                if _all_done(parent, shards):
                    time.sleep(0.6)  # one extra beat so the final numbers stay on screen
                    break
            else:
                payload = _read_payload(path)
                live.update(_render(payload, {}, run_name, width))
                if payload and payload.get("state") == "done":
                    time.sleep(0.6)
                    break
            time.sleep(1.0)  # poll the status files once a second


if __name__ == "__main__":
    main()
