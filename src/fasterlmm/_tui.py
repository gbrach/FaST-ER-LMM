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
