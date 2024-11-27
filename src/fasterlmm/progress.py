"""
Status JSON writer for long-running scans
A separate watcher process tails the json and shows what's happenig (comes in a later commit).
Keeping it minimal here, no schema.  The watcher just reads whatever keys are in the payload and decides what to display
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def write_status(path: str | Path, payload: dict) -> None:
    """
    Atomic status write: dump to a .tmp sibling, then rename in place
    Atomicity matters becuase a watcher could be reading at the same momment and a half-written json crashes the parser.
    Rename-on-same-filesystem is atomic on linux wich is the only OS that runs here
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "ts": time.time()}
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(path)


def pbar(iterable: Iterable[T],
         *,
         desc: str = "",
         total: int | None = None,
         leave: bool = True) -> Iterator[T]:
    """
    tqdm wrapper, falls back to a no-op iterator if tqdm is not installed
    The watcher reads status.json for richer infos anyway, this is just so the terminal isn't completely dead when running interactively
    """
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, leave=leave)
    except ImportError:
        return iter(iterable)
