"""Draw predicted against actual attention-score changes when a feature is removed.

Experiment 011 deletes the top attributed feature from the residual stream and measures how the
head's score moves. With normalisation scales frozen, as attribution graphs freeze them, the
prediction should be exact; with the norms free to respond it should not. Two panels on the same
scale make both claims visible at once. Every point is one head, from all four prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from figure_style import MODES, THEMES, finish, style_axis

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--results", type=Path, default=Path("experiments/results"))
parser.add_argument("--out-dir", type=Path, default=Path("experiments/figures"))
parser.add_argument("--stem", default="011-interventions")
args = parser.parse_args()

paths = [
    args.results / "011-intervention-0.6b.json",
    args.results / "011-intervention-1.7b.json",
    *sorted((args.results / "sweep").glob("*-intervention.json")),
]
by_model: dict[str, list[dict]] = {}
for path in paths:
    payload = json.loads(path.read_text())
    by_model.setdefault(payload["model"], []).extend(payload["rows"])
models = sorted(by_model, key=lambda name: float(name.split("-")[-1].rstrip("B")))
n_heads = sum(len(rows) for rows in by_model.values())

PANELS = (
    ("actual_score_change", "normalisation frozen, as in the graph"),
    ("actual_score_change_unfrozen", "normalisation free to respond"),
)


def pooled_r(key: str) -> float:
    predicted = [row["predicted_score_change"] for rows in by_model.values() for row in rows]
    actual = [row[key] for rows in by_model.values() for row in rows]
    return float(np.corrcoef(predicted, actual)[0, 1])


def draw(mode: str) -> Path:
    theme = THEMES[mode]
    figure, axes = plt.subplots(
        1, 2, figsize=(10.2, 4.9), sharex=True, sharey=True, facecolor=theme["surface"]
    )
    low, high = -7.0, 3.5
    for axis, (key, title) in zip(axes, PANELS, strict=True):
        style_axis(axis, theme, grid_axis="")
        axis.plot([low, high], [low, high], color=theme["grid"], linewidth=1, zorder=1)
        axis.axhline(0, color=theme["grid"], linewidth=0.6, zorder=0)
        axis.axvline(0, color=theme["grid"], linewidth=0.6, zorder=0)
        for slot, model in enumerate(models):
            rows = by_model[model]
            axis.scatter(
                [row["predicted_score_change"] for row in rows],
                [row[key] for row in rows],
                s=30,
                color=theme["series"][slot],
                edgecolors=theme["surface"],
                linewidths=0.8,
                alpha=0.9,
                zorder=3,
                label=f"{model.split('/')[-1]} ({len(rows)} heads)",
            )
        axis.text(
            0.04,
            0.94,
            f"pooled r = {pooled_r(key):.3f}",
            transform=axis.transAxes,
            color=theme["text"],
            fontsize=10,
            va="top",
        )
        axis.annotate(
            "exact prediction",
            xy=(2.2, 2.2),
            xytext=(-4, 8),
            textcoords="offset points",
            ha="right",
            color=theme["muted"],
            fontsize=8,
        )
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal")
        axis.set_title(title, color=theme["text"], fontsize=11, pad=8)
        axis.set_xlabel("predicted change in attention score", color=theme["muted"], fontsize=9)
    axes[0].set_ylabel("measured change", color=theme["muted"], fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncols=len(models),
        frameon=False,
        fontsize=9,
        labelcolor=theme["muted"],
    )
    return finish(
        figure,
        theme,
        f"Removing a feature moves the score by the attributed amount ({n_heads} heads)",
        args.out_dir,
        args.stem,
        mode,
    )


for mode in MODES:
    print("wrote", draw(mode))
