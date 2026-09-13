"""Draw the four-model comparison of experiment 009: feature share by depth, and compressibility.

The share table has sixteen cells and the rank table twenty-eight, and the pattern across models
is easier to see as lines. The panels measure different things, so each keeps its own vertical
axis and label; nothing is plotted against a second scale. A model keeps its colour in both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from figure_style import MODES, THEMES, direct_labels, finish, style_axis

MODELS = (
    ("0.6b", "0.6B, low L0"),
    ("1.7b", "1.7B, low L0"),
    ("4b", "4B, high L0"),
    ("8b", "8B, high L0"),
)
RANKS = ("1", "2", "4", "8", "16", "32", "64")
DEPTHS = (1 / 5, 2 / 5, 3 / 5, 4 / 5)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--results", type=Path, default=Path("experiments/results"))
parser.add_argument("--out-dir", type=Path, default=Path("experiments/figures"))
parser.add_argument("--stem", default="009-models")
args = parser.parse_args()


def load(tag: str, measure: str) -> dict:
    return json.loads((args.results / f"009-qwen3-{tag}-{measure}.json").read_text())


shares = {
    tag: [row["share"]["feature_feature"] for row in load(tag, "measure_completeness")["rows"]]
    for tag, _ in MODELS
}
truncation = {}
for tag, _ in MODELS:
    rows = load(tag, "measure_feature_rank")["rows"]
    truncation[tag] = [
        float(np.median([row["truncation"][rank] for row in rows if rank in row["truncation"]]))
        for rank in RANKS
    ]


def draw(mode: str) -> Path:
    theme = THEMES[mode]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.6), facecolor=theme["surface"])
    labels = [label for _, label in MODELS]

    style_axis(left, theme)
    for slot, (tag, label) in enumerate(MODELS):
        left.plot(
            DEPTHS,
            shares[tag],
            color=theme["series"][slot],
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.5,
            zorder=3,
            label=label,
        )
    left.set_xticks(DEPTHS, ["1/5", "2/5", "3/5", "4/5"])
    left.set_xlim(0.15, 1.12)
    left.set_ylim(0, 1)
    left.set_title("feature-pair share of the score", color=theme["text"], fontsize=11, pad=8)
    left.set_xlabel("sampled depth (one head per layer)", color=theme["muted"], fontsize=9)
    left.set_ylabel("share of score norm", color=theme["muted"], fontsize=9)
    direct_labels(left, DEPTHS[-1], [shares[tag][-1] for tag, _ in MODELS], labels, theme)

    style_axis(right, theme)
    ranks = [int(rank) for rank in RANKS]
    for slot, (tag, _) in enumerate(MODELS):
        right.plot(
            ranks,
            truncation[tag],
            color=theme["series"][slot],
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.2,
            zorder=3,
        )
    right.set_xscale("log", base=2)
    right.set_xticks(ranks, RANKS)
    right.set_xlim(0.8, 260)
    right.set_ylim(0, 1)
    right.set_title(
        "feature-pair block truncated to rank r", color=theme["text"], fontsize=11, pad=8
    )
    right.set_xlabel("rank r", color=theme["muted"], fontsize=9)
    right.set_ylabel("median relative error", color=theme["muted"], fontsize=9)
    direct_labels(right, ranks[-1], [truncation[tag][-1] for tag, _ in MODELS], labels, theme)

    handles, legend_labels = left.get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncols=len(MODELS),
        frameon=False,
        fontsize=9,
        labelcolor=theme["muted"],
    )
    return finish(
        figure,
        theme,
        "Higher-L0 transcoders carry more of the score, and compress no better",
        args.out_dir,
        args.stem,
        mode,
    )


for mode in MODES:
    print("wrote", draw(mode))
