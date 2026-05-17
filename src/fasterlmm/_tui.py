"""
Render helpers for the live watcher TUI
"""

from __future__ import annotations

import time
from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _fmt_ts(ts: float | None) -> str:
    """Localtime HH:MM:SS, or "?" if ts is missing"""
    if ts is None:
        return "?"
    return time.strftime("%H:%M:%S", time.localtime(ts))


def render_status(payload: dict[str, Any]) -> Panel:
    """
    Build the rich Panel for a status payload
    Keys the renderer knows about: state, pheno_idx, N, M, n_perm, perm_done, thresh_05, n_signif, ts
    Anything else in the payload just gets ignored, no schema to fight
    """
    state = payload.get("state", "?")
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()

    table.add_row("state", state)
    if "pheno_idx" in payload:
        table.add_row("pheno", str(payload["pheno_idx"]))
    if "N" in payload and "M" in payload:
        table.add_row("shape", f"N={payload['N']}  M={payload['M']}")
    if "n_perm" in payload:
        done = payload.get("perm_done", 0)
        total = payload["n_perm"]
        table.add_row("perms", f"{done}/{total}")
    if "thresh_05" in payload:
        table.add_row("p<0.05", f"{payload['thresh_05']:.3e}")
    if "n_signif" in payload:
        table.add_row("n_signif", str(payload["n_signif"]))
    table.add_row("ts", _fmt_ts(payload.get("ts")))

    title = Text(f"fasterlmm watch ({state})", style="bold")
    return Panel(table, title=title, border_style="cyan")


def _shard_row(idx: int, payload: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    """One row in the rollup, fields picked so the row keeps the same shape across loading -> scanning -> done"""
    state = payload.get("state", "?")
    device = payload.get("device", "?")
    pheno_n = payload.get("n_pheno")
    pheno_idx = payload.get("pheno_idx")
    pheno_str = f"{pheno_idx}/{pheno_n}" if pheno_idx is not None and pheno_n else (str(pheno_n) if pheno_n is not None else "-")
    if "n_perm" in payload:
        perm_str = f"{payload.get('perm_done', 0)}/{payload['n_perm']}"
    else:
        perm_str = "-"
    return (str(idx), state, device, pheno_str, perm_str, _fmt_ts(payload.get("ts")))


def render_multi_status(parent: dict[str, Any] | None, shards: dict[int, dict[str, Any]]) -> Panel:
    """
    Per-shard rollup. One flat Table.grid as the Panel body, Group-in-Panel breaks rich Lives height measurement so dont nest
    Header rows for the parent manifest, blank separator, then one row per discovered shard
    Rendering even before any shard has reported so the user sees something while torch.multiprocessing is still spawing workers
    """
    parent = parent or {}
    parent_state = parent.get("state", "waiting")
    n_gpu = parent.get("n_gpu", len(shards) or "?")

    g = Table.grid(padding=(0, 2))
    # 6 columns to fit the widest shard row, header rows pad the unused cols with empty strings
    for _ in range(6):
        g.add_column()
    g.add_row("[bold cyan]state[/]", str(parent_state), "", "", "", "")
    g.add_row("[bold cyan]n_gpu[/]", str(n_gpu), "", "", "", "")
    if "bundle" in parent:
        g.add_row("[bold cyan]bundle[/]", str(parent["bundle"]), "", "", "", "")
    g.add_row("[bold cyan]ts[/]", _fmt_ts(parent.get("ts")), "", "", "", "")
    g.add_row("", "", "", "", "", "")
    g.add_row("[bold]shard[/]", "[bold]state[/]", "[bold]device[/]", "[bold]pheno[/]", "[bold]perms[/]", "[bold]ts[/]")
    if shards:
        for i in sorted(shards):
            g.add_row(*_shard_row(i, shards[i]))
    else:
        g.add_row("-", "no shard status files yet", "-", "-", "-", "-")

    title = Text(f"fasterlmm watch  multi-shard ({parent_state})", style="bold")
    return Panel(g, title=title, border_style="cyan")
