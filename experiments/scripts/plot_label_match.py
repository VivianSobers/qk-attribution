"""Draw how often an explanation's top key feature fires on the token it points to.

Experiment 010 checks, for each head, whether the strongest key-side feature's top activating tokens
include the token at the position it points to, against a random other position in the same prompt.
Bars per model, pooled over prompts, with the counts printed so no rate stands without its
denominator.
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
parser.add_argument("--stem", default="010-label-match")
args = parser.parse_args()

counts: dict[str, dict[str, int]] = {}
prompts: dict[str, int] = {}
for path in sorted(args.results.glob("010-labelmatch-*.json")):
    payload = json.loads(path.read_text())
    tally = counts.setdefault(payload["model"], {"match": 0, "control": 0, "live": 0})
    tally["match"] += payload["n_match"]
    tally["control"] += payload["n_control_match"]
    tally["live"] += payload["n_live"]
    prompts[payload["model"]] = prompts.get(payload["model"], 0) + 1
models = sorted(counts, key=lambda name: float(name.split("-")[-1].rstrip("B")))

SERIES = (("match", "position the explanation points to"), ("control", "random other position"))


def draw(mode: str) -> Path:
    theme = THEMES[mode]
    figure, axis = plt.subplots(figsize=(8.2, 4.4), facecolor=theme["surface"])
    style_axis(axis, theme)
    centres = np.arange(len(models))
    width = 0.34
    for slot, (key, label) in enumerate(SERIES):
        offset = (slot - 0.5) * (width + 0.02)
        rates = [counts[model][key] / counts[model]["live"] for model in models]
        bars = axis.bar(
            centres + offset,
            rates,
            width=width,
            color=theme["series"][slot],
            edgecolor=theme["surface"],
            linewidth=2,
            zorder=3,
            label=label,
        )
        for bar, model in zip(bars, models, strict=True):
            axis.annotate(
                f"{counts[model][key]}/{counts[model]['live']}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                color=theme["muted"],
                fontsize=8,
            )
    axis.set_xticks(
        centres,
        [
            f"{model.split('/')[-1]}\n{prompts[model]} prompt{'s' if prompts[model] > 1 else ''}"
            for model in models
        ],
    )
    axis.set_ylim(0, 1.05)
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    axis.set_ylabel(
        "heads whose top key feature fires\non the token at that position",
        color=theme["muted"],
        fontsize=9,
    )
    figure.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncols=2,
        frameon=False,
        fontsize=9,
        labelcolor=theme["muted"],
    )
    return finish(
        figure,
        theme,
        "Top key features fire on the token they point to, far above a control",
        args.out_dir,
        args.stem,
        mode,
    )


for mode in MODES:
    print("wrote", draw(mode))
