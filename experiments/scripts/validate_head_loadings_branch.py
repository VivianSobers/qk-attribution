"""Validate the circuit-tracer head-loadings branch against real attribution graphs.

Experiment 014 checked an earlier version of the port on one graph. The branch has changed since:
tighter guards and a single-sweep ``path_head_loadings``. This checks the code as it stands on the
branch, over several graphs, and measures three things per edge.

1. Agreement with the graph's own adjacency entry, which circuit-tracer computed by a different
   route (a backward pass), so it is an independent reference.
2. The partition: at every attention layer on the edge's path, the per-head parts plus the bypass
   must reproduce the edge effect the same module computes.
3. The single sweep against splitting one layer at a time, for agreement and for wall-clock time on
   a real model rather than the CPU stub.

The module is imported from a copy of the branch file placed on the path as
``head_loadings_branch``, so it runs against the installed circuit-tracer without replacing it.
Pass the branch commit so the record says which code produced the numbers.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from types import SimpleNamespace

import torch
from circuit_tracer.graph import Graph
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
from head_loadings_branch import (
    FrozenRun,
    NodeLayout,
    edge_effect,
    head_loadings,
    path_head_loadings,
)
from transformer_lens import HookedTransformer

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--graphs", nargs="+", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--branch-commit", required=True)
parser.add_argument("--edges", type=int, default=60, help="half strongest, half random, per graph")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.set_grad_enabled(False)


def model_for_scan(scan: str) -> str:
    for size in ("0.6b", "1.7b", "4b", "8b", "14b"):
        if f"qwen3-{size}-" in scan:
            return f"Qwen/Qwen3-{size.upper()}"
    raise ValueError(f"cannot infer a model from scan {scan!r}")


def synced() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


backbone = None
loaded_model = None
report = {
    "branch_commit": args.branch_commit,
    "seed": args.seed,
    "edges_per_graph": args.edges,
    "model_dtype": "float32",
    "transcoder_dtype": "float32",
    "torch": torch.__version__,
    "graphs": [],
}

for graph_path in args.graphs:
    rng = random.Random(args.seed)
    graph = Graph.from_pt(graph_path)
    scan = graph.scan
    model_name = model_for_scan(scan)
    print(f"\n=== {graph_path}  model {model_name}  scan {scan}")

    transcoders, _ = load_transcoder_from_hub(
        scan, device=torch.device("cuda"), dtype=torch.float32, lazy_encoder=True, lazy_decoder=True
    )
    if loaded_model != model_name:
        backbone = HookedTransformer.from_pretrained_no_processing(
            model_name, device="cuda", dtype=torch.float32
        )
        loaded_model = model_name
    assert backbone is not None
    model = SimpleNamespace(
        cfg=backbone.cfg, blocks=backbone.blocks, W_E=backbone.W_E, transcoders=transcoders
    )
    run = FrozenRun.from_model(backbone, graph.input_tokens.unsqueeze(0).cuda())
    layout = NodeLayout.from_graph(graph)
    adjacency = graph.adjacency_matrix.cuda()
    layers = graph.active_features[graph.selected_features.cpu()][:, 0].cuda()

    block = adjacency[: layout.n_features, : layout.n_features]
    mask = (block.abs() > 1e-4) & (layers.unsqueeze(1) > layers.unsqueeze(0))
    pairs = torch.nonzero(mask)
    order = block[mask].abs().argsort(descending=True)
    half = max(1, args.edges // 2)
    strongest = [tuple(pairs[i].tolist()) for i in order[:half].tolist()]
    sampled = [
        tuple(pairs[i].tolist()) for i in rng.sample(range(len(pairs)), min(half, len(pairs)))
    ]
    chosen = list(dict.fromkeys(strongest + sampled))
    print(f"{len(pairs)} forward feature-to-feature edges, checking {len(chosen)}")

    rows = []
    for target, source in chosen:
        expected = float(adjacency[target, source])
        effect = edge_effect(model, graph, run, target, source)
        path_layers = list(range(int(layers[source]) + 1, int(layers[target]) + 1))

        start = synced()
        per_layer = [head_loadings(model, graph, run, target, source, a) for a in path_layers]
        loop_seconds = synced() - start

        start = synced()
        swept = path_head_loadings(model, graph, run, target, source)
        sweep_seconds = synced() - start

        partition = max(relative(part.total.reshape(1), effect.reshape(1)) for part in per_layer)
        sweep_vs_loop = max(
            relative(
                torch.cat([swept.at(a).per_head, swept.at(a).bypass.reshape(1)]),
                torch.cat([part.per_head, part.bypass.reshape(1)]),
            )
            for a, part in zip(path_layers, per_layer, strict=True)
        )
        rows.append(
            {
                "target": target,
                "source": source,
                "source_layer": int(layers[source]),
                "target_layer": int(layers[target]),
                "strongest": (target, source) in strongest,
                "adjacency": expected,
                "edge_effect": float(effect),
                "ratio": float(effect) / expected,
                "max_relative_partition_error": partition,
                "max_relative_sweep_vs_loop": sweep_vs_loop,
                "loop_seconds": loop_seconds,
                "sweep_seconds": sweep_seconds,
            }
        )

    ratios = torch.tensor([r["ratio"] for r in rows])
    speedups = torch.tensor([r["loop_seconds"] / r["sweep_seconds"] for r in rows])
    summary = {
        "graph": graph_path,
        "model": model_name,
        "scan": scan,
        "n_pos": run.n_pos,
        "n_forward_edges": len(pairs),
        "n_edges": len(rows),
        "median_ratio": float(ratios.median()),
        "within_5pct": int((ratios - 1).abs().lt(0.05).sum()),
        "within_10pct": int((ratios - 1).abs().lt(0.10).sum()),
        "min_ratio": float(ratios.min()),
        "max_ratio": float(ratios.max()),
        "max_relative_partition_error": max(r["max_relative_partition_error"] for r in rows),
        "max_relative_sweep_vs_loop": max(r["max_relative_sweep_vs_loop"] for r in rows),
        "median_speedup": float(speedups.median()),
        "rows": rows,
    }
    report["graphs"].append(summary)
    print(
        f"ratio median {summary['median_ratio']:.4f}"
        f"  within 5% {summary['within_5pct']}/{len(rows)}"
        f"  within 10% {summary['within_10pct']}/{len(rows)}"
        f"  range {summary['min_ratio']:.3f}..{summary['max_ratio']:.3f}"
        f"\npartition error {summary['max_relative_partition_error']:.1e}"
        f"  sweep vs loop {summary['max_relative_sweep_vs_loop']:.1e}"
        f"  median speedup {summary['median_speedup']:.1f}x"
    )

with open(args.out, "w") as handle:
    json.dump(report, handle, indent=1)
print("wrote", args.out)
