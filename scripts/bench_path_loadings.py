"""Time path_loadings against splitting one attention layer at a time.

Runs on CPU against the test stub, so it needs no weights and no GPU. The stub's dimensions are
tiny, so the ratio mostly reflects how many attention steps each approach takes rather than what a
real model would see in wall-clock time. Agreement is reported relative to the values, since the
stub's unnormalised weights make absolute magnitudes grow several-fold per layer.

Usage: PYTHONPATH=src:. python scripts/bench_path_loadings.py [--depths 8 16 28] [--reps 5]
"""

from __future__ import annotations

import argparse
import time

import torch

from qk_attribution.loadings import edge_loadings, path_loadings
from tests.stubs import D_MODEL, make_model
from tests.test_propagate import make_cache

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--depths", type=int, nargs="+", default=[8, 16, 28])
parser.add_argument("--reps", type=int, default=5)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

torch.set_grad_enabled(False)
print(f"seed {args.seed}, {args.reps} repetitions, float32, CPU, stub d_model={D_MODEL}")
for depth in args.depths:
    model, cache = make_model(n_layers=depth), make_cache(n_layers=depth)
    generator = torch.Generator().manual_seed(args.seed)
    source = torch.randn(D_MODEL, generator=generator)
    reader = torch.randn(D_MODEL, generator=generator)
    layers = range(1, depth)

    start = time.perf_counter()
    for _ in range(args.reps):
        loop = [edge_loadings(model, cache, source, 1, 0, reader, 4, depth - 1, a) for a in layers]
    per_layer = (time.perf_counter() - start) / args.reps

    start = time.perf_counter()
    for _ in range(args.reps):
        swept = path_loadings(model, cache, source, 1, 0, reader, 4, depth - 1)
    together = (time.perf_counter() - start) / args.reps

    worst = 0.0
    for a in layers:
        reference = torch.cat([loop[a - 1].per_head, loop[a - 1].bypass.reshape(1)])
        got = torch.cat([swept.at(a).per_head, swept.at(a).bypass.reshape(1)])
        worst = max(worst, float((got - reference).abs().max() / reference.abs().max()))
    print(
        f"depth {depth:3d}: one layer at a time {per_layer * 1e3:8.2f} ms, "
        f"path_loadings {together * 1e3:6.2f} ms, {per_layer / together:5.1f}x, "
        f"worst relative difference {worst:.1e}"
    )
