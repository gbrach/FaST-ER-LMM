"""LUX snapshot discovery and dashboard helpers, imported from the sibling project."""

from __future__ import annotations

import glob
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# reuse the clean core's shared progress bar; _gradient_bar just wraps it
from fasterlmm._tui import progress_bar


def _read_snapshot(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


_FRESH_S = 300.0  # files with mtime older than this drop out (matches hub HEARTBEAT_STALE_S)


def _filter_fresh(paths: list[Path]) -> list[Path]:
    """drop paths whose mtime is older than _FRESH_S. when ALL matches are
    stale (e.g. between runs), keep them so the watcher isn't empty. skips
    .jsonl files because their mtime is the event log, not the snapshot"""
    import time
    now = time.time()
    def _mtime(p):
        try:
            return p.stat().st_mtime
        except (FileNotFoundError, OSError):
            return -1.0
    fresh = [p for p in paths if (now - _mtime(p)) < _FRESH_S]
    return fresh if fresh else paths


def _shard_idx_key(p: Path) -> tuple[int, str]:
    """numeric sort by .shardN suffix so shard10 comes after shard9, not shard1. unsharded paths fall through to lexicographic."""
    name = p.name
    if ".shard" in name:
        try:
            return (int(name.rsplit(".shard", 1)[1]), "")
        except ValueError:
            pass
    return (0, name)


def _resolve_shard_paths(arg: str) -> list[Path]:
    """taking a literal path, a glob, or the base path of a sharded run (auto-discovers .shard0, .shard1, ...). returns existing files only; empty list while waiting for the first write. stale files (mtime > 5min) are dropped so left-over status files from prior runs don't pollute the panel"""
    p = Path(arg)
    if any(ch in arg for ch in "*?["):
        matches = sorted((Path(m) for m in glob.glob(arg) if not m.endswith(".jsonl")),
                         key=_shard_idx_key)
        return _filter_fresh(matches)
    matches = sorted((Path(m) for m in glob.glob(f"{arg}.shard*") if not m.endswith(".jsonl")),
                     key=_shard_idx_key)
    if matches:
        return _filter_fresh(matches)
    return [p]


_STAGE_COLOR = {
    "read_plink": "blue", "read_phen": "blue", "read_covar": "blue",
    "rint_phen": "blue",
    "standardise_genotypes": "cyan", "align_inputs": "cyan", "rotate": "cyan",
    "main": "bright_white", "chunk": "bright_white",
    "permute_phenotypes": "magenta",
    "scan": "yellow", "loco_scan_compat": "yellow", "snp_wald_scan": "yellow",
    "results_assemble": "green",
    "drain_writes": "bright_green", "write_outputs": "bright_green",
}

_STATE_GLYPH = {
    "running": ("⏵", "yellow"),
    "starting": ("…", "yellow"),
    "done": ("✓", "green"),
    "chrom_done": ("✓", "green"),
    "error": ("✗", "red"),
}


def _stage_color(stage: str | None) -> str:
    return _STAGE_COLOR.get(stage or "", "white")


def _stage_text(stage: str | None, state: str | None) -> Text:
    glyph, gcol = _STATE_GLYPH.get(state or "", ("·", "dim"))
    t = Text()
    t.append(glyph + " ", style=gcol)
    t.append(stage or "?", style=f"bold {_stage_color(stage)}")
    if state:
        t.append("  ")
        t.append(state, style="dim")
    return t


def _gradient_bar(frac: float, width: int = 56) -> Text:
    """deprecated alias for fasterlmm._tui.progress_bar; kept so older callers
    that imported _gradient_bar from this module still work. delegates to the
    shared bar so the chrome stays unified across watchers"""
    frac = max(0.0, min(1.0, frac))
    # shared bar takes (done, total); rebuild from frac at width resolution
    done = int(round(width * frac))
    return progress_bar(done, width, width=width, style="primary")


def _build_recent_panel(states: list[dict[str, Any]], n: int = 8) -> Panel:
    """pulling recent events from every shard snapshot, deduping when shards emit the same event in lockstep (LOCO chrom rotations on multi-gpu often fire at the same ts/chrom)"""
    events: list[dict[str, Any]] = []
    for s in states:
        for ev in s.get("recent", []):
            events.append(ev)
    events.sort(key=lambda e: e.get("ts", 0))
    seen: set[tuple] = set()
    deduped: list[dict[str, Any]] = []
    for ev in events:
        # 10s bucket: shard events for the same chrom/chunk drift by a few hundred ms, the visible list should merge them anyway
        key = (int(float(ev.get("ts", 0)) // 10),
               ev.get("stage"), ev.get("stage_state"), ev.get("chrom"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ev)
    events = deduped[-n:]
    if not events:
        return Panel(Text("(no events yet)", style="dim"), title="recent",
                     border_style="grey50", title_align="left")
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True, width=8)
    table.add_column(no_wrap=False)
    for ev in events:
        ts = ev.get("ts", time.time())
        when = time.strftime("%H:%M:%S", time.localtime(ts))
        line = _stage_text(ev.get("stage"), ev.get("stage_state"))
        extras: list[str] = []
        if "chrom" in ev:
            try:
                extras.append(f"chr {int(ev['chrom']):>2}")
            except (TypeError, ValueError):
                pass
        if "chunk_idx" in ev:
            extras.append(f"chunk {ev['chunk_idx']}/{ev.get('chunks_total','?')}")
        if "chrom_elapsed_s" in ev:
            extras.append(f"{ev['chrom_elapsed_s']:.1f}s")
        if "last_chunk_elapsed_s" in ev:
            extras.append(f"{ev['last_chunk_elapsed_s']:.1f}s")
        if "last_chunk_cols_per_s" in ev and ev["last_chunk_cols_per_s"]:
            extras.append(f"{ev['last_chunk_cols_per_s']:,.0f} pheno-cols/s")
        if extras:
            line.append("   ")
            line.append("  ".join(extras), style="dim")
        table.add_row(when, line)
    title = Text("recent events", style="bold")
    return Panel(table, title=title, border_style="grey50", title_align="left")


def _resolve_search_roots(search_root: str | None) -> list[Path]:
    """deciding where to look when --search-root wasn't given. tries cwd/results, cwd, then walks up the parent chain looking for any results/ dir. also peeks at sibling-project conventional output dirs (cwd-parent/<sibling>/data/GWAS/associations) so snakemake-driven runs that write status under a sibling repo's tree get discovered without --search-root. starlight is the canonical example: snakemake writes <starlight>/data/GWAS/associations/<cond>/.status.json"""
    if search_root is not None:
        return [Path(search_root)]
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    out: list[Path] = []
    seen: set[Path] = set()
    def _add(p: Path) -> None:
        if p in seen or not p.exists():
            return
        seen.add(p)
        out.append(p)
    _add(cwd / "results")
    _add(cwd)
    # walking up from cwd toward $HOME, peeking at each parent for a results/ dir
    for parent in cwd.parents:
        if parent == home or parent == Path("/"):
            break
        _add(parent / "results")
    _add(home / "fasterlmm" / "results")
    # sibling-project association trees. only adding the narrow GWAS/associations subtree so rglob stays bounded (the parent project's full data/ can be huge with raw genotype matrices); these are a few hundred files at most. matches the snakemake convention used in starlight's `data/GWAS/associations/<cond>/.status.json`
    for sib in (home.iterdir() if home.exists() else []):
        if not sib.is_dir() or sib == cwd:
            continue
        if sib.name.startswith(".") or sib.name in {"miniforge3", "anaconda3"}:
            continue
        for sub in ("data/GWAS/associations",
                     "data/GWAS/associations_proteomics",
                     "data/GWAS/associations_GxE",
                     "data/GWAS/associations_GxE_kronecker"):
            _add(sib / sub)
    return out


def _autodiscover_status(
    search_root: str | None, console,
    *,
    quiet: bool = False) -> str | None:
    """finding the most recently touched .status.json across one or more search roots. picks by mtime so a live run wins over stale snapshots. quiet=True skips printing, used by the re-discovery loop. when search_root is None, walks cwd / cwd-parents / $HOME/fasterlmm looking for results/"""
    roots = _resolve_search_roots(search_root)
    if not roots:
        if not quiet:
            console.print(
                f"[red]no candidate search roots found "
                f"(cwd={Path.cwd()}, no results/ in cwd or any parent up to $HOME)[/red]")
        return None
    # any json with 'status' in its name (covers .status.json, status.json, real_glucose_2gpu.status.json.shard0, etc.) but skip rolling .jsonl event logs. files can vanish between rglob and stat, tolerating that. merges hits across ALL search roots so a fresh run in ~/starlight beats a stale one in ~/fasterlmm/results
    candidates: list[Path] = []
    for root in roots:
        for p in root.rglob("*status*"):
            if "status" not in p.name:
                continue
            if not (p.name.endswith(".json") or ".json.shard" in p.name):
                continue
            if p.name.endswith(".jsonl"):
                continue
            try:
                if p.stat().st_size > 0:
                    candidates.append(p)
            except (FileNotFoundError, OSError):
                continue
    if not candidates:
        if not quiet:
            tried = "\n  ".join(str(r) for r in roots)
            console.print(
                f"[yellow]no status.json found. tried:\n  {tried}\n"
                f"pass an explicit path or --search-root DIR[/yellow]")
        return None
    # candidates can vanish between rglob() and the max() stat() during atomic
    # rename (writer drops .tmp -> real path). protect the key fn the same way
    # the rglob loop above already protects its stat
    def _safe_mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except (FileNotFoundError, OSError):
            return -1.0
    # active-run filter: a snapshot whose top-level stage is "main" + stage_state "done" is a finished run; skip those in favour of anything still mid-stage. fall back to the full set when no actives so the watcher still shows something on a quiet machine
    def _is_active(p: Path) -> bool:
        try:
            with open(p) as f:
                blob = json.load(f)
        except (FileNotFoundError, OSError, ValueError):
            return True  # unreadable / mid-rename → tentatively keep
        # writer marks the run finished by setting stage='main' + stage_state='done' as the last write (cli.py:387). everything else is still in flight
        return not (blob.get("stage") == "main"
                    and blob.get("stage_state") == "done")
    actives = [p for p in candidates if _is_active(p)]
    pool = actives if actives else candidates
    newest = max(pool, key=_safe_mtime)
    if _safe_mtime(newest) < 0:
        return None
    age = time.time() - _safe_mtime(newest)
    tag = "live" if age < 60 else (f"{age/60:.0f}m old" if actives
                                     else f"{age/60:.0f}m old, finished")
    # if the newest is a shard, build a glob across its siblings (the stem before .shardN) so multi-gpu runs show every shard
    if ".shard" in newest.name:
        stem = newest.name.split(".shard", 1)[0]
        target = str(newest.parent / f"{stem}.shard*")
    else:
        target = str(newest)
    if not quiet:
        console.print(f"[dim]watching {_short_label(target)} ({tag})[/dim]")
    return target


def _route_dir_arg(args) -> bool:
    """status_file passed as a directory → treat as --search-root and let the
    autodiscover walk find the newest .status.json under it. mirrors what the
    umbrella `fasterlmm watch` dispatcher does, just inlined per sub-watcher
    so `fasterlmm epi-watch <dir>` / `fasterlmm gxe-watch <dir>` / direct
    `fasterlmm watch <dir>` all behave the same. returns True when the swap
    fired so the caller knows to invoke autodiscover"""
    sf = getattr(args, "status_file", None)
    if sf is None:
        return False
    if not Path(sf).is_dir():
        return False
    if getattr(args, "search_root", None) is None:
        args.search_root = sf
    args.status_file = None
    return True


def _short_label(target: str) -> str:
    """compressing a status-file path or shard glob to <cell>/<mode> for display. sweep paths look like .../<cell>/<mode>/.status.json[.shard*]; falling back to the full path when the layout doesn't match (single ad-hoc run)"""
    p = Path(target)
    sharded = ".shard" in p.name
    parent = p.parent
    if parent.parent != parent and parent.parent.name and parent.name:
        label = f"{parent.parent.name}/{parent.name}"
    else:
        label = str(p)
    return f"{label} (multi-GPU)" if sharded else label


# ── vendored from dev fasterlmm._tui: the nvidia-smi GPU-stats helper the epi
# watcher uses. the public _tui only ships PALETTE/shard_color/progress_bar; the
# gxe watcher didn't need GPU panels, so these weren't vendored before. the epi
# watcher prefers the snapshot's recorded gpu_name/gpu_total_mb and only falls
# back to a live nvidia-smi read via _read_gpu_stats when running on the compute
# box itself. watcher-only / cosmetic — outside the parity surface.

_GPU_STATS_CACHE: dict[int, tuple[float, "GpuStats"]] = {}


@dataclass
class GpuStats:
    """one-shot nvidia-smi snapshot for one device. None on any field that
    didn't parse (e.g. driver missing util.gpu support)"""
    used_mb: int | None = None
    total_mb: int | None = None
    name: str | None = None
    util_pct: int | None = None


def _read_gpu_stats(gpu_idx: int | None) -> GpuStats:
    """nvidia-smi for one device: memory + name + util in one call. cached
    2s. returns empty GpuStats when nvidia-smi missing / index bad / query
    fails. used by every panel that wants to render gpu state"""
    if gpu_idx is None or gpu_idx < 0:
        return GpuStats()
    now = time.time()
    cached = _GPU_STATS_CACHE.get(gpu_idx)
    if cached and (now - cached[0] < 2.0):
        return cached[1]
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,name,utilization.gpu",
             "--format=csv,noheader,nounits",
             "-i", str(gpu_idx)],
            capture_output=True, text=True, timeout=2.0, check=False)
        if out.returncode != 0 or not out.stdout.strip():
            stats = GpuStats()
        else:
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            # parts: [mem_used, mem_total, name, util]. util may be "[N/A]"
            # on older cards / drivers; left as None then
            used = int(parts[0]) if parts[0].isdigit() else None
            total = int(parts[1]) if parts[1].isdigit() else None
            name = parts[2] if len(parts) > 2 and parts[2] else None
            util = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
            stats = GpuStats(used_mb=used, total_mb=total, name=name, util_pct=util)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
        stats = GpuStats()
    _GPU_STATS_CACHE[gpu_idx] = (now, stats)
    return stats
