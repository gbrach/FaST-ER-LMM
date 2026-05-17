"""
live TUI for a running fasterlmm scan
pointing it at the outdir gets the per-shard rollup (auto-fanout / slurm-array), pointing at a single status.json gets the legacy one-pane view
dir mode globs status.shard*.json siblings + the top-level status.json as they apear, so the panel fills in as workers come online
usage:
  fasterlmm watch <outdir>             rollup, one row per shard
  fasterlmm watch <outdir/status.json> single-pane
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live

from fasterlmm._tui import render_multi_status, render_status


_SHARD_RE = re.compile(r"^status\.shard(\d+)\.json$")


def _read_payload(path: Path) -> dict | None:
    """returning None on a missing file or a mid-rename half-write, caller deciding what to render in the placeholer"""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _discover_shards(outdir: Path) -> dict[int, dict]:
    """
    globbing status.shard*.json in outdir, skipping the .tmp atomic-rename siblings that flash into existance for a millisecond during write_status
    returns {shard_idx: payload}
    """
    out: dict[int, dict] = {}
    for p in sorted(outdir.glob("status.shard*.json")):
        if p.suffix == ".tmp" or p.name.endswith(".json.tmp"):
            continue
        m = _SHARD_RE.match(p.name)
        if not m:
            continue
        payload = _read_payload(p)
        if payload is not None:
            out[int(m.group(1))] = payload
    return out


def _all_done(parent: dict | None, shards: dict[int, dict]) -> bool:
    """
    flagging done when the parent status.json says done, OR when every discovered shard says done and at least one shard exists
    parent-side covers auto-fanout (the parent writes the finalizer after spawn join), shard-side covers explicit --shard slurm-array where noone writes the parent
    """
    if parent and parent.get("state") == "done":
        return True
    if shards and all(s.get("state") == "done" for s in shards.values()):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(prog="fasterlmm watch",
                                     description="live TUI for a running fasterlmm gwas job, single or multi-shard")
    parser.add_argument("path", help="outdir (multi-shard rollup) or status.json file (legacy single-shard)")
    parser.add_argument("--poll-sec", type=float, default=0.5, help="seconds between polls")
    args = parser.parse_args()

    path = Path(args.path)
    # multi mode if path is a directory, or if path doesnt exist yet but its parent already has shard files lying around (watcher launched before the parent finalizer wrote status.json)
    if path.is_dir():
        outdir = path
        multi = True
    elif not path.exists() and path.parent.is_dir() and any(path.parent.glob("status.shard*.json")):
        outdir = path.parent
        multi = True
    else:
        outdir = path.parent
        multi = False

    console = Console()
    with Live(refresh_per_second=4, console=console) as live:
        while True:
            if multi:
                shards = _discover_shards(outdir)
                parent = _read_payload(outdir / "status.json")
                live.update(render_multi_status(parent, shards))
                if _all_done(parent, shards):
                    time.sleep(0.5)  # one extra beat so the final numbers stay on screen before exit
                    break
            else:
                payload = _read_payload(path) or {"state": "waiting for status file"}
                live.update(render_status(payload))
                if payload.get("state") == "done":
                    time.sleep(0.5)  # same extra beat as above so done state is visible
                    break
            time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
