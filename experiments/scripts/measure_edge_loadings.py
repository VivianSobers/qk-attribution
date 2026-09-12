"""Compare computed edge effects against circuit-tracer's own adjacency, and split them by head.

Two questions. First, does propagating a source feature's decoder direction through frozen
attention and reading it with the target's encoder reproduce the edge weight the attribution graph
already holds? A constant ratio would reveal a normalisation; a varying one would mean the two are
measuring different things. Second, how concentrated are the resulting head loadings?
"""

from __future__ import annotations

import json
import os

import torch
from circuit_tracer.graph import Graph
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
from transformer_lens import HookedTransformer

from qk_attribution.loadings import edge_effect, edge_loadings

MODEL = "Qwen/Qwen3-0.6B"
GRAPH = "spike_out/graph.pt"
SEED = 0
# Both encoders and decoders are needed, and at float32 the pair does not fit on a 24 GB card when
# many layers are touched. The default run reads them in bfloat16; a short float32 run over fewer
# edges checks that the residual disagreement with the graph is rounding and not a modelling gap.
N_EDGES = int(os.environ.get("QK_N_EDGES", "40"))
DTYPE = torch.float32 if os.environ.get("QK_DTYPE") == "float32" else torch.bfloat16
# The saved graph records the dtype its own run used. Matching it matters: attention patterns and
# layernorm scales computed in bfloat16 differ from float32 ones by more than the arithmetic here.
MODEL_DTYPE = torch.bfloat16 if os.environ.get("QK_MODEL_DTYPE") == "bfloat16" else torch.float32

torch.manual_seed(SEED)
torch.set_grad_enabled(False)

graph = Graph.from_pt(GRAPH)
assert isinstance(graph.scan, str), "a multi-scan graph names several transcoder sets"
# Loaded lazily so only the layers actually touched are materialised. Rows are cast to float32 for
# the arithmetic regardless of how they were stored.
transcoders, _ = load_transcoder_from_hub(
    graph.scan,
    device=torch.device("cuda"),
    dtype=DTYPE,
    lazy_encoder=True,
    lazy_decoder=True,
)
model = HookedTransformer.from_pretrained_no_processing(MODEL, device="cuda", dtype=MODEL_DTYPE)
tokens = graph.input_tokens.unsqueeze(0).cuda()
_, cache = model.run_with_cache(tokens)

active = graph.active_features[graph.selected_features].cuda()
acts = graph.activation_values.cuda().float()
adjacency = graph.adjacency_matrix.cuda().float()
n_features = len(graph.selected_features)
print(
    f"model={MODEL} seed={SEED} transcoder_dtype={DTYPE} model_dtype={MODEL_DTYPE} "
    f"edges={N_EDGES} graph_dtype={graph.cfg.dtype} features={n_features} "
    f"adjacency={tuple(adjacency.shape)}"
)

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
    by_layer = []
    for attention_layer in range(source_layer + 1, target_layer + 1):
        loading = edge_loadings(
            model,
            cache,
            source,
            source_position,
            source_layer,
            reader,
            target_position,
            target_layer,
            attention_layer,
        )
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

name = f"measure_edge_loadings_{str(DTYPE).split('.')[-1]}_{str(MODEL_DTYPE).split('.')[-1]}.json"
with open(name, "w") as fh:
    json.dump(
        {
            "model": MODEL,
            "graph": GRAPH,
            "scan": graph.scan,
            "seed": SEED,
            "transcoder_dtype": str(DTYPE),
            "model_dtype": str(MODEL_DTYPE),
            "n_edges": N_EDGES,
            "rows": rows,
        },
        fh,
    )
print("peak GPU MiB:", torch.cuda.max_memory_allocated() // 2**20)
