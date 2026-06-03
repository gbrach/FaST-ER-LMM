"""
cpu-only unit tests for fasterlmm.progress
covers the atomic status writer (write_status) and the tqdm-or-not iterator wrapper (pbar).
nothing here spawns a subprocess or touches a gpu, every test writes into pytest's tmp_path so reruns start clean.
the watcher process that reads the status json lives elsewhere, here the contract is just: json round-trips plus a numeric ts, second call overwrites, missing parent dirs get created.
"""
from __future__ import annotations

import json

import pytest

from fasterlmm.progress import pbar, write_status


# ---------------------------------------------------------------------------
# write_status
# ---------------------------------------------------------------------------


def test_write_status_roundtrips_payload_plus_ts(tmp_path):
    """payload comes back via json.loads, with an extra numeric ts key injected"""
    path = tmp_path / "status.json"
    payload = {"stage": "scan", "done": 7, "total": 42, "label": "chr1"}
    write_status(path, payload)

    loaded = json.loads(path.read_text())
    # every original key survives untouched
    for k, v in payload.items():
        assert loaded[k] == v
    # the writer injects ts and it is a real number
    assert "ts" in loaded
    assert isinstance(loaded["ts"], (int, float))
    # write_status must not mutate the caller's dict
    assert "ts" not in payload


def test_write_status_overwrites_on_second_call(tmp_path):
    """a second write replaces the file contents wholesale, no merging of old keys"""
    path = tmp_path / "status.json"
    write_status(path, {"step": 1, "msg": "first"})
    write_status(path, {"step": 2})

    loaded = json.loads(path.read_text())
    assert loaded["step"] == 2
    # the stale key from the first call is gone, overwrite is not a merge
    assert "msg" not in loaded
    assert "ts" in loaded


def test_write_status_ts_is_monotonic_nondecreasing(tmp_path):
    """ts reflects wall-clock at write time so a later call is not earlier than an earlier one"""
    path = tmp_path / "status.json"
    write_status(path, {"n": 1})
    first = json.loads(path.read_text())["ts"]
    write_status(path, {"n": 2})
    second = json.loads(path.read_text())["ts"]
    assert second >= first


def test_write_status_mkdirs_missing_parent(tmp_path):
    """a status path several levels deep gets its parent tree created on the fly"""
    path = tmp_path / "deep" / "nested" / "dir" / "status.json"
    assert not path.parent.exists()
    write_status(path, {"ok": True})
    assert path.exists()
    assert json.loads(path.read_text())["ok"] is True


def test_write_status_accepts_str_path(tmp_path):
    """path can be a plain string, not just a Path object"""
    path = tmp_path / "as_str.json"
    write_status(str(path), {"via": "string"})
    assert json.loads(path.read_text())["via"] == "string"


def test_write_status_leaves_no_tmp_sibling(tmp_path):
    """the .tmp scratch file is renamed away, so only the final json remains"""
    path = tmp_path / "status.json"
    write_status(path, {"clean": 1})
    tmp = path.with_suffix(path.suffix + ".tmp")
    assert path.exists()
    assert not tmp.exists()


# ---------------------------------------------------------------------------
# pbar
# ---------------------------------------------------------------------------


def test_pbar_yields_input_items_in_order():
    """pbar is a pass-through iterator, same items same order whether tqdm is around or not"""
    items = list(range(11))
    out = list(pbar(items, desc="scan", total=len(items)))
    assert out == items


def test_pbar_return_is_iterable_and_lazy():
    """the return value is iterable, pulling from it yields the items one at a time in order
    tqdm wraps the iterable and is iterable but not itself an iterator, so go through iter() not next() directly"""
    it = iter(pbar(["a", "b", "c"]))
    assert next(it) == "a"
    assert next(it) == "b"
    assert list(it) == ["c"]


def test_pbar_works_on_a_generator_source():
    """a non-list iterable like a generator still flows through unchanged"""
    src = (x * x for x in range(5))
    assert list(pbar(src)) == [0, 1, 4, 9, 16]


def test_pbar_empty_iterable_yields_nothing():
    """an empty input produces an empty pass-through, no errors"""
    assert list(pbar([])) == []
