"""
Live TUI for a running fasterlmm scan
Tails the status.json that cli.py writes and re-renders a rich Panel every poll interval.
Usage: fasterlmm-watch <status.json> [--poll-sec 0.5]
status.json is written atomically by progress.write_status so reading at any time is safe
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live

from fasterlmm._tui import render_status


def main() -> None:
    parser = argparse.ArgumentParser(prog="fasterlmm-watch",
                                     description="live TUI for a running fasterlmm-gwas job")
    parser.add_argument("status_file", help="path to the status.json written by fasterlmm-gwas")
    parser.add_argument("--poll-sec", type=float, default=0.5, help="seconds betwen polls")
    args = parser.parse_args()

    path = Path(args.status_file)
    console = Console()
    with Live(refresh_per_second=4, console=console) as live:
        while True:
            try:
                payload = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                # file might not exist yet if the scan hasnt written its first status, or might be mid-rename.  Showing a placeholder and trying again
                payload = {"state": "waiting for status file"}
            live.update(render_status(payload))
            if payload.get("state") == "done":
                time.sleep(0.5)  # one extra beat so the final numbers stay on screen before exit
                break
            time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
