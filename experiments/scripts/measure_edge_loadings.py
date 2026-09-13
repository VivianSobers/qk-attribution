"""Compare computed edge effects against circuit-tracer's own adjacency, and split them by head.

Two questions. First, does propagating a source feature's decoder direction through frozen
attention and reading it with the target's encoder reproduce the edge weight the attribution graph
already holds? A constant ratio would reveal a normalisation; a varying one would mean the two are
measuring different things. Second, how concentrated are the resulting head loadings?
"""

from __future__ import annotations

import json

import torch
from common import load, parser

from qk_attribution.features import activations_for
from qk_attribution.loadings import edge_effect, path_loadings

extra = parser(__doc__.splitlines()[0])
extra.add_argument("--edges", type=int, default=40)
args = extra.parse_args()
fixtures = load(args)
graph, transcoders, model, cache = (
    fixtures.graph,
    fixtures.transcoders,
    fixtures.model,
    fixtures.cache,
)
N_EDGES = args.edges
MODEL_DTYPE = fixtures.model_dtype

active = graph.active_features[graph.selected_features].cuda()
acts = activations_for(graph).cuda().float()
adjacency = graph.adjacency_matrix.cuda().float()
n_features = len(graph.selected_features)
print(f"edges={N_EDGES} features={n_features} adjacency={tuple(adjacency.shape)}")

# Pick the strongest feature-to-feature edges that both span a layer and cross a position. A
# same-position edge travels down the residual stream and needs no head at all, so it says nothing
# about which head carried anything.
block = adjacency[:n_features, :n_features]
layers, positions = active[:, 0], active[:, 1]
spans = layers[:, None] > layers[None, :]
crosses = positions[:, None] != positions[None, :]
candidates = (block.abs() * spans * crosses).flatten().topk(N_EDGES).indices
rows = []
for flat in candidates.tolist():
    target_index, source_index = divmod(flat, n_features)
    source_layer, source_position, source_feature = active[source_index].tolist()
    target_layer, target_position, target_feature = active[target_index].tolist()
    source = transcoders[source_layer].W_dec[source_feature].to(MODEL_DTYPE) * acts[
        source_index
    ].to(MODEL_DTYPE)
    reader = transcoders[target_layer].W_enc[target_feature].to(MODEL_DTYPE)

    effect = edge_effect(
        model,
        cache,
        source,
        source_position,
        source_layer,
        reader,
        target_position,
        target_layer,
    )
    edge = block[target_index, source_index]

    # Split at every attention layer the signal could have used. Each is its own partition of the
    # same total, so the layer where the bypass is smallest is the one that actually carried it.
    # path_loadings computes every layer's split in one sweep each way, and matches calling
    # edge_loadings once per layer to floating-point precision (tests/test_loadings.py).
    path = path_loadings(
        model,
        cache,
        source,
        source_position,
        source_layer,
        reader,
        target_position,
        target_layer,
    )
    by_layer = []
    for attention_layer in path.layers:
        loading = path.at(attention_layer)
        magnitudes = loading.per_head.abs()
        by_layer.append(
            {
                "layer": attention_layer,
                "partition_error": ((loading.total - effect).abs() / effect.abs()).item(),
                "head_magnitude": magnitudes.sum().item(),
                "top_head": int(magnitudes.argmax()),
                "top_head_share": (magnitudes.max() / magnitudes.sum()).item(),
                "bypass_share": (
                    loading.bypass.abs() / (magnitudes.sum() + loading.bypass.abs())
                ).item(),
            }
        )
    carrier = min(by_layer, key=lambda entry: entry["bypass_share"])
    rows.append(
        {
            "source_layer": source_layer,
            "target_layer": target_layer,
            "source_position": source_position,
            "target_position": target_position,
            "edge": edge.item(),
            "effect": effect.item(),
            "ratio": (edge.float() / effect.float()).item() if effect.abs() > 0 else float("nan"),
            "partition_error": max(entry["partition_error"] for entry in by_layer),
            "carrier_layer": carrier["layer"],
            "carrier_bypass_share": carrier["bypass_share"],
            "top_head_share": carrier["top_head_share"],
            "bypass_share": carrier["bypass_share"],
            "layers_searched": len(by_layer),
            "by_layer": by_layer,
        }
    )

ratios = torch.tensor([r["ratio"] for r in rows])
errors = torch.tensor([r["partition_error"] for r in rows])
top = torch.tensor([r["top_head_share"] for r in rows])
bypass = torch.tensor([r["bypass_share"] for r in rows])
print(
    f"edge / effect ratio: median={ratios.median():.4f} min={ratios.min():.4f} "
    f"max={ratios.max():.4f} std={ratios.std():.4f}"
)
print(f"partition error: median={errors.median():.2e} max={errors.max():.2e}")
print(
    f"top head share of head magnitude: median={top.median():.3f} min={top.min():.3f} "
    f"max={top.max():.3f}"
)
print(
    f"carrier-layer bypass share: median={bypass.median():.3f} min={bypass.min():.3f} "
    f"max={bypass.max():.3f}"
)
offsets = torch.tensor([float(r["carrier_layer"] - r["source_layer"]) for r in rows])
searched = torch.tensor([float(r["layers_searched"]) for r in rows])
print(
    f"carrier layer is {offsets.median():.0f} layers above the source (median), "
    f"out of {searched.median():.0f} searched"
)
shares = torch.tensor(
    [
        max(entry["head_magnitude"] for entry in r["by_layer"])
        / sum(entry["head_magnitude"] for entry in r["by_layer"])
        for r in rows
    ]
)
print(f"busiest layer holds {shares.median():.3f} of all head magnitude (median)")

with open(args.out, "w") as fh:
    json.dump({**fixtures.config(), "n_edges": N_EDGES, "rows": rows}, fh)
print("wrote", args.out)
print("peak GPU MiB:", torch.cuda.max_memory_allocated() // 2**20)
