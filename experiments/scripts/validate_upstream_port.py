"""Check the ported head-loadings module against a real graph's own adjacency matrix.

Two claims are checked. The edge effect the module computes must match the number circuit-tracer
independently put in the adjacency matrix for that edge, and the per-head parts plus the bypass
must reproduce that effect at every attention layer along the path.

The module expects a ReplacementModel, which is a HookedTransformer carrying its transcoders. Here
the two are loaded separately and joined by a small adapter, so the check needs no attribution run.
"""

import argparse
import json
import random
from types import SimpleNamespace

import torch
from circuit_tracer.graph import Graph
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
from head_loadings_port import FrozenRun, NodeLayout, edge_effect, head_loadings
from transformer_lens import HookedTransformer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--graph", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--model", default=None)
parser.add_argument("--edges", type=int, default=40)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.set_grad_enabled(False)
rng = random.Random(args.seed)

graph = Graph.from_pt(args.graph)
scan = graph.scan
model_name = args.model or graph.cfg.model_name
print(f"graph {args.graph}\nmodel {model_name}\nscan {scan}\nseed {args.seed}")

transcoders, _ = load_transcoder_from_hub(
    scan, device=torch.device("cuda"), dtype=torch.float32, lazy_encoder=True, lazy_decoder=True
)
backbone = HookedTransformer.from_pretrained_no_processing(
    model_name, device="cuda", dtype=torch.float32
)
model = SimpleNamespace(
    cfg=backbone.cfg, blocks=backbone.blocks, W_E=backbone.W_E, transcoders=transcoders
)
run = FrozenRun.from_model(backbone, graph.input_tokens.unsqueeze(0).cuda())
layout = NodeLayout.from_graph(graph)
adjacency = graph.adjacency_matrix.cuda()
active = graph.active_features
selected = graph.selected_features


layers = active[selected.cpu()][:, 0].to(adjacency.device)
block = adjacency[: layout.n_features, : layout.n_features]
# Vectorised: a candidate is a forward feature-to-feature edge with a non-negligible weight.
forward = layers.unsqueeze(1) > layers.unsqueeze(0)
mask = (block.abs() > 1e-4) & forward
pairs = torch.nonzero(mask)
if pairs.numel() == 0:
    raise SystemExit("no forward feature-to-feature edges in this graph")
weights = block[mask].abs()
order = weights.argsort(descending=True)
half = max(1, args.edges // 2)
strongest = pairs[order[:half]].tolist()
sampled = [pairs[i].tolist() for i in rng.sample(range(len(pairs)), min(half, len(pairs)))]
chosen = [tuple(pair) for pair in strongest + sampled]


def layer_of(node: int) -> int:
    return int(layers[node])


print(f"{len(pairs)} forward feature-to-feature edges, checking {len(chosen)}")

rows = []
for target, source in chosen:
    expected = float(adjacency[target, source])
    got = float(edge_effect(model, graph, run, target, source))
    errors = []
    for attention_layer in range(layer_of(source) + 1, layer_of(target) + 1):
        parts = head_loadings(model, graph, run, target, source, attention_layer)
        errors.append(abs(float(parts.total) - got))
    rows.append(
        {
            "target": target,
            "source": source,
            "source_layer": layer_of(source),
            "target_layer": layer_of(target),
            "adjacency": expected,
            "edge_effect": got,
            "ratio": got / expected if expected else float("nan"),
            "max_partition_error": max(errors) if errors else 0.0,
        }
    )
    print(
        f"  {source:5d} -> {target:5d}  adj {expected:+.5f}  got {got:+.5f}  "
        f"ratio {rows[-1]['ratio']:.4f}  partition err {rows[-1]['max_partition_error']:.2e}"
    )

ratios = torch.tensor([r["ratio"] for r in rows])
partition = torch.tensor([r["max_partition_error"] for r in rows])
scale = torch.tensor([abs(r["edge_effect"]) for r in rows]).clamp_min(1e-12)
print(f"\nedge effect / adjacency: median {ratios.median():.4f}, std {ratios.std():.4f}")
print(f"  within 5% of 1.0: {int((ratios - 1).abs().lt(0.05).sum())}/{len(rows)}")
print(
    f"partition error: max {partition.max():.3e}, relative {float((partition / scale).max()):.3e}"
)

with open(args.out, "w") as fh:
    json.dump(
        {
            "graph": args.graph,
            "model": model_name,
            "scan": scan,
            "seed": args.seed,
            "n_edges": len(rows),
            "median_ratio": float(ratios.median()),
            "std_ratio": float(ratios.std()),
            "max_partition_error": float(partition.max()),
            "max_relative_partition_error": float((partition / scale).max()),
            "rows": rows,
        },
        fh,
        indent=1,
    )
print("wrote", args.out)
