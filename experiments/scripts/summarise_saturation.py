"""Pool the saturation curves from one or more runs into the table experiment 013 reports.

Kept separate from the measurement so the write-up's numbers can be regenerated from the JSON on
disk without a GPU, and so the choice of summary statistic can change without re-running anything.

``k50`` and ``k90`` are first crossings of a curve that is not monotone, which makes them sensitive
to a single noisy point. The area under the normalised curve against log k is the robust companion:
it is 1.0 for a ranking that puts all the movement in the first feature and 0.0 for one that puts it
all in the last, and it does not care where any individual point lands.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("results", nargs="+", type=Path)
args = parser.parse_args()


def area(curve: dict[str, float], ks: list[int], full: float) -> float:
    """Trapezoidal area under the normalised movement curve, against log10 k."""
    xs = [math.log10(k) for k in ks]
    ys = [min(curve[str(k)] / full, 1.0) for k in ks]
    total = sum((xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) / 2 for i in range(len(xs) - 1))
    return total / (xs[-1] - xs[0])


def share(row: dict, field: str) -> float | None:
    value = row[field]
    return None if value is None else value / row["n_features"]


pooled: list[dict] = []
for path in args.results:
    payload = json.loads(path.read_text())
    for row in payload["rows"]:
        pooled.append({**row, "model": payload["model"], "source": path.name})

if not pooled:
    raise SystemExit("no head curves in the given results")

top_areas = [area(r["top"], r["ks"], r["full_movement"]) for r in pooled]
random_areas = [area(r["random"], r["ks"], r["full_movement"]) for r in pooled]
wins = sum(r["top"][str(k)] > r["random"][str(k)] for r in pooled for k in r["ks"][:-1])
points = sum(len(r["ks"]) - 1 for r in pooled)
drops = sum(
    r["top"][str(r["ks"][i + 1])] < r["top"][str(r["ks"][i])]
    for r in pooled
    for i in range(len(r["ks"]) - 1)
)

print(f"heads pooled: {len(pooled)} from {len({r['source'] for r in pooled})} runs")
print(f"models: {', '.join(sorted({r['model'] for r in pooled}))}")
print()
for field in ("k50", "k90"):
    shares = [s for r in pooled if (s := share(r, field)) is not None]
    counts = [r[field] for r in pooled if r[field] is not None]
    exhausted = sum(1 for r in pooled if r[field] == r["n_features"])
    print(
        f"{field}: median {int(statistics.median(counts))} features "
        f"({statistics.median(shares):.1%} of the pool), "
        f"range {min(shares):.2%} to {max(shares):.1%}, "
        f"{exhausted}/{len(pooled)} heads needed the whole pool"
    )
    random_counts = [r[f"random_{field}"] for r in pooled if r[f"random_{field}"] is not None]
    print(f"  random order needs a median of {int(statistics.median(random_counts))} features")
print()
ratio = statistics.median(t / r for t, r in zip(top_areas, random_areas, strict=True))
print(
    f"area under the normalised curve: ranked {statistics.median(top_areas):.3f}, "
    f"random {statistics.median(random_areas):.3f}, per-head ratio {ratio:.2f}x"
)
print(f"ranked beats random at {wins}/{points} = {wins / points:.1%} of (head, k) points")
print(f"the ranked curve steps downward {drops}/{points} times, so it is far from monotone")
