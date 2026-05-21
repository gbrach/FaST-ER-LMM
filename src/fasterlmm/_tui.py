"""
Visual primitives for the live watcher TUI
A pastel palette mirroring the matplotlib figures (sky / peach / mint / dusty rose), one progress-bar helper and a per-shard colour cycle.  Panel bodies stay flat Text everywhere, Rich Live can't measure a Group nested inside a Panel and renders it below instead of in place
"""

from __future__ import annotations

from rich.text import Text

# pastel palette, same hues as the figs: sky / peach / mint / yellow / dusty rose
PALETTE: dict[str, str] = {
    "primary":  "rgb(126,184,212)",  # pastel sky, the fasterlmm core colour
    "accent":   "rgb(240,168,120)",  # pastel peach
    "success":  "rgb(161,217,155)",  # pastel mint, done
    "warn":     "rgb(255,217,128)",  # pastel yellow, in-flight
    "fail":     "rgb(224,140,140)",  # dusty rose, failed
    "lavender": "rgb(196,180,234)",  # extra shard colour
    "muted":    "rgb(170,170,170)",  # dim secondary text
    "label":    "rgb(206,206,206)",  # near-white labels
}

# per-shard colour cycle. peach + mint lead (no blue, that's the header colour)
SHARD_PALETTE_KEYS: tuple[str, ...] = ("accent", "success", "lavender", "warn", "fail")


def shard_color(idx: int) -> str:
    """colour for one gpu shard panel, cycling the pastel set so each shard reads distinct"""
    return PALETTE[SHARD_PALETTE_KEYS[idx % len(SHARD_PALETTE_KEYS)]]


def progress_bar(done: float, total: float,
                 width: int = 40,
                 *,
                 style: str = "primary") -> Text:
    """
    Flat unicode bar tinted from the palette
    An empty or unknown total renders all light blocks; a bar that reaches the end flips to the success colour so a finished shard reads diferent at a glance
    """
    color = PALETTE.get(style, PALETTE["primary"])
    bar = Text()
    if total <= 0:
        bar.append("░" * width, style=PALETTE["muted"])
        return bar
    frac = max(0.0, min(1.0, done / total))
    filled = int(round(width * frac))
    if filled > 0:
        bar.append("█" * filled, style=PALETTE["success"] if frac >= 0.999 else color)
    if filled < width:
        bar.append("░" * (width - filled), style=PALETTE["muted"])
    return bar
