"""Draw how well each candidate score form reproduces Qwen3-0.6B's attention scores, head by head.

Experiment 003 reports medians. The finding is that the plain weight product fails on nearly every
head while the corrected form succeeds on every one, which a cumulative distribution over all 448
heads shows directly: each curve says what fraction of heads fall under a given error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from figure_style import MODES, THEMES, finish, style_axis

FORMS = (
    ("derived", "corrected form"),
    ("plain", "plain W_Q W_Kᵀ"),
    ("no_rope", "rotary left out"),
)
# Where each direct label sits: the share of heads to anchor at, and which side of the curve.
LABEL_AT = {"derived": (0.62, "right"), "plain": (0.25, "right"), "no_rope": (0.82, "left")}

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--result", type=Path, default=Path("experiments/results/003-qk-form-qwen3-0.6b.json")
)
parser.add_argument("--out-dir", type=Path, default=Path("experiments/figures"))
parser.add_argument("--stem", default="003-score-form")
args = parser.parse_args()

rows = json.loads(args.result.read_text())["rows"]
errors = {form: np.sort([row[form]["rel_fro"] for row in rows]) for form, _ in FORMS}


def draw(mode: str) -> Path:
    theme = THEMES[mode]
    figure, axis = plt.subplots(figsize=(8.2, 4.4), facecolor=theme["surface"])
    style_axis(axis, theme, grid_axis="x")
    axis.axvline(1.0, color=theme["grid"], linewidth=1, linestyle=(0, (4, 4)), zorder=1)
    axis.annotate(
        "error as large as\nthe score itself",
        xy=(1.0, 0.04),
        xytext=(6, 0),
        textcoords="offset points",
        color=theme["muted"],
        fontsize=8,
    )
    for slot, (form, label) in enumerate(FORMS):
        values = errors[form]
        share = np.arange(1, len(values) + 1) / len(values)
        colour = theme["series"][slot]
        axis.step(values, share, where="post", color=colour, linewidth=2, zorder=3, label=label)
        median = float(np.median(values))
        at, side = LABEL_AT[form]
        axis.annotate(
            f"{label}\nmedian {median:.2g}",
            xy=(float(np.quantile(values, at)), at),
            xytext=(10 if side == "right" else -10, 0),
            textcoords="offset points",
            ha="left" if side == "right" else "right",
            va="center",
            color=theme["muted"],
            fontsize=9,
        )
    axis.set_xscale("log")
    axis.set_xlim(3e-8, 5e2)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel(
        "relative error against the model's own attention scores, per head",
        color=theme["muted"],
        fontsize=9,
    )
    axis.set_ylabel(f"share of the {len(rows)} heads", color=theme["muted"], fontsize=9)
    figure.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncols=3,
        frameon=False,
        fontsize=9,
        labelcolor=theme["muted"],
    )
    return finish(
        figure,
        theme,
        "The corrected score form holds on every head; the plain product fails on most",
        args.out_dir,
        args.stem,
        mode,
    )


for mode in MODES:
    print("wrote", draw(mode))
