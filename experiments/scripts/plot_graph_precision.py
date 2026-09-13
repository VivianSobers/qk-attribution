"""Draw how far each validated edge sits from its graph's adjacency entry, by graph dtype.

Experiment 015 checks the head-loadings branch against bfloat16 and float32 graphs of the same
prompts, and 017 follows the worst bfloat16 edges into the float32 graphs. The tables give ranges;
the finding is the shape, that disagreement shrinks with edge size in bfloat16 and vanishes in
float32, so this draws every edge.

Both axes are logarithmic. Some float32 edges agree exactly, which a log axis cannot place, so
deviations below the floor are drawn on it and the tick says so. The matched outlier edges from 017
are joined from their bfloat16 position to their float32 position.

One panel per model with a shared vertical axis, written for light and dark surfaces.
"""

from __future__ import annotations

import argparse
import json
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
        "float32": "#2a78d6",
        "bfloat16": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#3a3a38",
        "float32": "#3987e5",
        "bfloat16": "#d95926",
    },
}
FLOOR = 1e-6
MODELS = ("0.6b", "1.7b")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--results", type=Path, default=Path("experiments/results"))
parser.add_argument("--out-dir", type=Path, default=Path("experiments/figures"))
parser.add_argument("--stem", default="015-graph-precision")
args = parser.parse_args()


def edges(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for graph in json.loads(path.read_text())["graphs"] for row in graph["rows"]]
    size = np.array([abs(row["adjacency"]) for row in rows])
    miss = np.array([abs(row["ratio"] - 1) for row in rows])
    return size, np.maximum(miss, FLOOR)


def point(record: dict) -> tuple[float, float]:
    return abs(record["adjacency"]), max(abs(record["ratio"] - 1), FLOOR)


data = {
    model: {
        "bfloat16": edges(args.results / f"015-head-loadings-branch-{model}.json"),
        "float32": edges(args.results / f"015-head-loadings-branch-{model}-float32-graphs.json"),
        "matched": [
            (point(outlier["reference"]), point(outlier["float32"]))
            for outlier in (
                json.loads(path.read_text())
                for path in sorted(args.results.glob(f"017-outlier-{model}-*.json"))
            )
        ],
    }
    for model in MODELS
}


def draw(mode: str) -> Path:
    theme = THEMES[mode]
    figure, axes = plt.subplots(
        1, len(MODELS), figsize=(10.2, 4.5), sharey=True, facecolor=theme["surface"]
    )

    for panel, (axis, model) in enumerate(zip(axes, MODELS, strict=True)):
        axis.set_facecolor(theme["surface"])
        axis.axhline(0.1, color=theme["grid"], linewidth=1, linestyle=(0, (4, 4)), zorder=1)
        for dtype, label in (("bfloat16", "bfloat16 graphs"), ("float32", "float32 graphs")):
            size, miss = data[model][dtype]
            axis.scatter(
                size,
                miss,
                s=30,
                color=theme[dtype],
                edgecolors=theme["surface"],
                linewidths=0.8,
                alpha=0.85,
                zorder=3,
                label=f"{label} ({len(size)} edges)",
            )
        for start, end in data[model]["matched"]:
            axis.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": theme["text"],
                    "linewidth": 1.1,
                    "shrinkA": 6,
                    "shrinkB": 6,
                },
                zorder=4,
            )
            axis.scatter(
                *zip(start, end, strict=True),
                s=90,
                facecolors="none",
                edgecolors=theme["text"],
                linewidths=1.2,
                zorder=5,
            )
        # Direct labels in the empty band between the small and the strongest edges, so identity
        # never rests on colour alone.
        axis.text(2.5, 2e-2, "bfloat16", color=theme["muted"], fontsize=9, ha="center")
        axis.text(2.5, 1e-5, "float32", color=theme["muted"], fontsize=9, ha="center")
        if panel == 0:
            axis.annotate(
                "10% off",
                xy=(1e2, 0.1),
                xytext=(0, 4),
                textcoords="offset points",
                ha="right",
                color=theme["muted"],
                fontsize=8,
            )
            axis.text(
                2.5e-3,
                1.3,
                "ringed: the worst edges, followed\ninto the float32 graph (017)",
                color=theme["muted"],
                fontsize=8,
                va="center",
            )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylim(FLOOR / 2, 3)
        axis.set_title(f"Qwen3-{model.upper()}", color=theme["text"], fontsize=11, pad=8)
        axis.set_xlabel("|adjacency entry|", color=theme["muted"], fontsize=9)
        axis.tick_params(colors=theme["muted"], labelsize=9)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(theme["grid"])
        axis.grid(axis="y", color=theme["grid"], linewidth=0.6, alpha=0.5, zorder=0)

    ticks = [FLOOR, 1e-4, 1e-2, 1]
    axes[0].set_yticks(ticks, ["≤1e-6", "1e-4", "1e-2", "1"])
    axes[0].set_ylabel("|edge effect / adjacency − 1|", color=theme["muted"], fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        [label.split(" (")[0] for label in labels],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncols=2,
        frameon=False,
        fontsize=9,
        labelcolor=theme["muted"],
    )
    figure.suptitle(
        "Small edges are misstated in bfloat16 graphs and agree in float32",
        color=theme["text"],
        fontsize=13,
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.stem}-{mode}.png"
    figure.savefig(path, dpi=200, facecolor=theme["surface"])
    plt.close(figure)
    return path


for mode in ("light", "dark"):
    print("wrote", draw(mode))
