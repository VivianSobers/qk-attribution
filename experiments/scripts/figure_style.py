"""Shared look for the experiment figures, so every chart in the README reads as one set.

Colours are the reference categorical palette in its fixed slot order, stepped separately for the
light and dark surfaces and validated as a set against each. Series take slots in order and never
skip one, so a model keeps its colour from one figure to the next.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#d8d7d3",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#3a3a38",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
    },
}
MODES = ("light", "dark")


def style_axis(axis, theme: dict, *, grid_axis: str = "y") -> None:
    """Recessive axes: no top or right spine, muted ticks, a faint grid on one axis."""
    axis.set_facecolor(theme["surface"])
    axis.tick_params(colors=theme["muted"], labelsize=9)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(theme["grid"])
    if grid_axis:
        axis.grid(axis=grid_axis, color=theme["grid"], linewidth=0.6, alpha=0.5, zorder=0)


def direct_labels(axis, x: float, ys: list[float], labels: list[str], theme: dict) -> None:
    """Label line ends in text ink, nudged apart vertically so no two labels overlap.

    Positions are resolved in display space, so this works on linear and log axes alike.
    """
    figure = axis.figure
    figure.canvas.draw()
    to_display = axis.transData.transform
    # Heights in points, so the spacing survives saving at a different dpi from the canvas.
    points_per_pixel = 72 / figure.dpi
    points = sorted(
        (
            (float(to_display((x, y))[1]) * points_per_pixel, y, label)
            for y, label in zip(ys, labels, strict=True)
        ),
        key=lambda item: item[0],
    )
    gap = 11.0
    placed: list[float] = []
    for height, _, _ in points:
        placed.append(max(height, placed[-1] + gap) if placed else height)
    for (height, y, label), target in zip(points, placed, strict=True):
        axis.annotate(
            label,
            xy=(x, y),
            xytext=(8, target - height),
            textcoords="offset points",
            va="center",
            color=theme["muted"],
            fontsize=9,
            annotation_clip=False,
        )


def finish(figure, theme: dict, title: str, out_dir: Path, stem: str, mode: str) -> Path:
    """Title, layout and save one mode of a figure."""
    figure.suptitle(title, color=theme["text"], fontsize=13, y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}-{mode}.png"
    figure.savefig(path, dpi=200, facecolor=theme["surface"])
    plt.close(figure)
    return path
