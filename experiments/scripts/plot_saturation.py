"""Draw the saturation curves experiment 013 measures.

The tables in the write-up give two crossings of a curve whose shape is the actual finding, so this
draws the curve. Heads differ in pool size, so the horizontal axis is the fraction of the pool
ablated rather than a raw count; each head's curve is interpolated onto a shared grid in log space
before the median and the interquartile band are taken.

One panel per model rather than one axis with two scales, and the vertical axis is shared, so the
two panels can be compared by eye without a second scale to read.

Written for light and dark surfaces from the same data, since a research README is read in both.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical slots 1 and 2 of the reference palette, validated as a pair against both surfaces.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#d8d7d3",
        "ranked": "#2a78d6",
        "random": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#3a3a38",
        "ranked": "#3987e5",
        "random": "#d95926",
    },
}

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("results", nargs="+", type=Path)
parser.add_argument("--out-dir", type=Path, default=Path("experiments/figures"))
parser.add_argument("--stem", default="013-saturation")
args = parser.parse_args()

GRID = np.logspace(math.log10(0.001), 0.0, 60)


def curve_on_grid(curve: dict[str, float], ks: list[int], full: float, pool: int) -> np.ndarray:
    """Interpolate one head's normalised curve onto the shared share-of-pool grid."""
    shares = np.array([k / pool for k in ks])
    values = np.array([min(curve[str(k)] / full, 1.0) for k in ks])
    return np.interp(np.log10(GRID), np.log10(shares), values, left=np.nan, right=values[-1])


by_model: dict[str, list[dict]] = {}
for path in args.results:
    payload = json.loads(path.read_text())
    by_model.setdefault(payload["model"], []).extend(payload["rows"])

order = sorted(by_model, key=lambda name: len(by_model[name]), reverse=True)
stacks = {}
for model in order:
    rows = by_model[model]
    stacks[model] = {
        arm: np.vstack(
            [curve_on_grid(r[arm], r["ks"], r["full_movement"], r["n_features"]) for r in rows]
        )
        for arm in ("top", "random")
    }


def draw(mode: str) -> Path:
    theme = THEMES[mode]
    figure, axes = plt.subplots(
        1, len(order), figsize=(5.1 * len(order), 4.3), sharey=True, facecolor=theme["surface"]
    )
    axes = np.atleast_1d(axes)

    for panel, (axis, model) in enumerate(zip(axes, order, strict=True)):
        axis.set_facecolor(theme["surface"])
        for level in (0.5, 0.9):
            axis.axhline(level, color=theme["grid"], linewidth=1, linestyle=(0, (4, 4)), zorder=1)
        for arm, key, label in (
            ("top", "ranked", "ranked by attribution"),
            ("random", "random", "random order"),
        ):
            stack = stacks[model][arm]
            median = np.nanmedian(stack, axis=0)
            low = np.nanpercentile(stack, 25, axis=0)
            high = np.nanpercentile(stack, 75, axis=0)
            axis.fill_between(GRID, low, high, color=theme[key], alpha=0.16, linewidth=0, zorder=2)
            axis.plot(
                GRID,
                median,
                color=theme[key],
                linewidth=2,
                solid_capstyle="round",
                zorder=3,
                label=label,
            )
            # Direct labels, so identity is never carried by colour alone. Anchored where the
            # two curves are furthest apart rather than at the right edge, where they converge.
            anchor = int(np.argmin(np.abs(GRID - 0.02)))
            if panel == 0:
                axis.annotate(
                    label,
                    xy=(GRID[anchor], median[anchor]),
                    xytext=(4, 9 if arm == "top" else -17),
                    textcoords="offset points",
                    ha="left",
                    color=theme["muted"],
                    fontsize=9,
                    zorder=4,
                )

        axis.set_xscale("log")
        axis.set_xlim(GRID[0], 1.0)
        axis.set_ylim(0, 1.02)
        axis.set_title(
            f"{model.split('/')[-1]}  ({len(by_model[model])} heads)",
            color=theme["text"],
            fontsize=11,
            pad=8,
        )
        axis.set_xlabel(
            "share of the attributed features ablated", color=theme["muted"], fontsize=9
        )
        axis.tick_params(colors=theme["muted"], labelsize=9)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(theme["grid"])
        axis.grid(axis="y", color=theme["grid"], linewidth=0.6, alpha=0.5, zorder=0)

    axes[0].set_ylabel("share of the full-ablation movement", color=theme["muted"], fontsize=9)
    axes[0].annotate(
        "half",
        xy=(GRID[0], 0.5),
        xytext=(2, 4),
        textcoords="offset points",
        color=theme["muted"],
        fontsize=8,
    )
    axes[0].annotate(
        "nine tenths",
        xy=(GRID[0], 0.9),
        xytext=(2, 4),
        textcoords="offset points",
        color=theme["muted"],
        fontsize=8,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncols=2,
        frameon=False,
        fontsize=9,
        labelcolor=theme["muted"],
    )
    figure.suptitle(
        "The ranking front-loads the movement",
        color=theme["text"],
        fontsize=13,
        y=0.985,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.87))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.stem}-{mode}.png"
    figure.savefig(path, dpi=200, facecolor=theme["surface"])
    plt.close(figure)
    return path


for mode in ("light", "dark"):
    print("wrote", draw(mode))
