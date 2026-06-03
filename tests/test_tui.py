"""
cpu-only unit tests for the watcher visual primitives in fasterlmm._tui
just the pastel palette, the per-shard colour cycle and the flat unicode
progress bar.  nothing here spawns a Live or a subprocess so it stays portable
and fast on any clone, no gpu and no rich rendering loop needed
"""
from __future__ import annotations

import re

from rich.text import Text

from fasterlmm._tui import (
    PALETTE,
    SHARD_PALETTE_KEYS,
    progress_bar,
    shard_color,
)


# the documented palette keys, every one must be present
_EXPECTED_KEYS = {
    "primary",
    "accent",
    "success",
    "warn",
    "fail",
    "muted",
    "label",
    "lavender",
}

# rich rgb string like rgb(126,184,212)
_RGB_RE = re.compile(r"^rgb\(\d{1,3},\d{1,3},\d{1,3}\)$")


# ---------------------------------------------------------------------------
# PALETTE
# ---------------------------------------------------------------------------


def test_palette_has_documented_keys():
    """every documented palette key is present"""
    assert _EXPECTED_KEYS <= set(PALETTE)


def test_palette_values_are_rgb_strings():
    """each palette entry is a well-formed rgb(...) string in range"""
    for key, val in PALETTE.items():
        assert isinstance(val, str)
        assert _RGB_RE.match(val), f"{key} -> {val} is not an rgb string"
        channels = [int(c) for c in val[len("rgb("):-1].split(",")]
        assert len(channels) == 3
        assert all(0 <= c <= 255 for c in channels)


# ---------------------------------------------------------------------------
# SHARD_PALETTE_KEYS + shard_color
# ---------------------------------------------------------------------------


def test_shard_palette_keys_length_five():
    """the shard cycle is exactly five keys long"""
    assert len(SHARD_PALETTE_KEYS) == 5


def test_shard_palette_keys_all_in_palette():
    """every shard key resolves to a real palette entry"""
    for key in SHARD_PALETTE_KEYS:
        assert key in PALETTE


def test_shard_color_returns_palette_value():
    """shard_color hands back an actual palette colour string"""
    palette_vals = set(PALETTE.values())
    for idx in range(12):
        assert shard_color(idx) in palette_vals


def test_shard_color_cycles_period_five():
    """shard_color repeats with period 5, so colour(i) == colour(i+5)"""
    for idx in range(15):
        assert shard_color(idx) == shard_color(idx + 5)


def test_shard_color_distinct_within_one_cycle():
    """the five colours inside a single cycle are all different"""
    cycle = [shard_color(i) for i in range(5)]
    assert len(set(cycle)) == 5


# ---------------------------------------------------------------------------
# progress_bar
# ---------------------------------------------------------------------------


def test_progress_bar_returns_text():
    """progress_bar hands back a rich Text object"""
    bar = progress_bar(3, 10)
    assert isinstance(bar, Text)


def test_progress_bar_plain_length_matches_width():
    """for a positive total the plain string is exactly width chars long"""
    for width in (10, 40, 73):
        bar = progress_bar(3, 10, width=width)
        assert len(bar.plain) == width


def test_progress_bar_zero_total_is_all_muted():
    """total <= 0 renders width muted blocks, no filled glyphs"""
    width = 40
    bar = progress_bar(0, 0, width=width)
    assert len(bar.plain) == width
    # nothing filled when the total is unknown
    assert "█" not in bar.plain
    assert set(bar.plain) == {"░"}
    # the single span is the muted colour
    assert all(span.style == PALETTE["muted"] for span in bar.spans)


def test_progress_bar_negative_total_is_all_muted():
    """a negative total falls in the same unknown-total branch"""
    width = 25
    bar = progress_bar(5, -1, width=width)
    assert len(bar.plain) == width
    assert set(bar.plain) == {"░"}


def test_progress_bar_full_length_matches_width():
    """a completed bar still has a plain string of exactly width chars"""
    width = 40
    bar = progress_bar(10, 10, width=width)
    assert len(bar.plain) == width
    assert bar.plain == "█" * width


def test_progress_bar_done_exceeds_total_is_full():
    """done past total clamps to a full bar of width filled blocks"""
    width = 30
    bar = progress_bar(50, 10, width=width)
    assert len(bar.plain) == width
    assert bar.plain == "█" * width


def test_progress_bar_completed_uses_success_colour():
    """a finished bar tints the filled blocks with the success colour"""
    bar = progress_bar(10, 10, width=40)
    span_styles = {span.style for span in bar.spans}
    assert PALETTE["success"] in span_styles


def test_progress_bar_partial_uses_style_colour_not_success():
    """a half-finished bar uses the requested style for the filled part, not success"""
    bar = progress_bar(5, 10, width=40, style="accent")
    span_styles = {span.style for span in bar.spans}
    # filled part carries the accent colour, the rest is muted
    assert PALETTE["accent"] in span_styles
    assert PALETTE["muted"] in span_styles
    # well short of done so the success flip should not have fired
    assert PALETTE["success"] not in span_styles


def test_progress_bar_unknown_style_falls_back_to_primary():
    """an unrecognised style keyword falls back to the primary colour"""
    bar = progress_bar(5, 10, width=40, style="not-a-real-style")
    span_styles = {span.style for span in bar.spans}
    assert PALETTE["primary"] in span_styles


def test_progress_bar_default_width_is_forty():
    """the default bar width is 40 columns"""
    bar = progress_bar(2, 10)
    assert len(bar.plain) == 40
