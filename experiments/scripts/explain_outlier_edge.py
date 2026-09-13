"""Recompute one head-loadings validation edge against a float32 graph of the same prompt.

Experiment 014 found one edge of 40 whose linear effect came to 0.760 of the graph's adjacency
entry. The graph was attributed in bfloat16, and experiment 007 showed that storage precision
explains most of the spread in that ratio, but this edge is strong, so small denominators do not
account for it. Two explanations remain: bfloat16 accumulation in the graph's backward pass, or a
real difference between what the backward pass counts and what forward propagation through frozen
attention counts.

A float32 graph of the same prompt separates them. If its adjacency entry for the same feature pair
matches the forward effect, precision was the cause. Node indices are not stable across graphs,
since selection can differ, so the edge is matched by its (layer, position, feature) triples.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import torch
from circuit_tracer.graph import Graph
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
from head_loadings_branch import FrozenRun, edge_effect
from transformer_lens import HookedTransformer

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--reference-graph", required=True, help="the bfloat16 graph from experiment 014"
)
parser.add_argument("--float32-graph", required=True)
parser.add_argument("--target", type=int, required=True, help="node index in the reference graph")
parser.add_argument("--source", type=int, required=True, help="node index in the reference graph")
parser.add_argument("--model", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.set_grad_enabled(False)


def triple(graph, node: int) -> tuple[int, int, int]:
    return tuple(int(x) for x in graph.active_features[int(graph.selected_features[node])])


def node_for(graph, wanted: tuple[int, int, int]) -> int | None:
    rows = graph.active_features[graph.selected_features]
    hits = torch.nonzero((rows == torch.tensor(wanted)).all(dim=1)).flatten()
    return int(hits[0]) if len(hits) else None


reference = Graph.from_pt(args.reference_graph)
precise = Graph.from_pt(args.float32_graph)
target_triple = triple(reference, args.target)
source_triple = triple(reference, args.source)
print(f"target (layer, pos, feature) {target_triple}, source {source_triple}")

transcoders, _ = load_transcoder_from_hub(
    reference.scan,
    device=torch.device("cuda"),
    dtype=torch.float32,
    lazy_encoder=True,
    lazy_decoder=True,
)
backbone = HookedTransformer.from_pretrained_no_processing(
    args.model, device="cuda", dtype=torch.float32
)
model = SimpleNamespace(
    cfg=backbone.cfg, blocks=backbone.blocks, W_E=backbone.W_E, transcoders=transcoders
)

record = {
    "reference_graph": args.reference_graph,
    "float32_graph": args.float32_graph,
    "model": args.model,
    "scan": reference.scan,
    "seed": args.seed,
    "target_triple": target_triple,
    "source_triple": source_triple,
}
for label, graph, target, source in (
    ("reference", reference, args.target, args.source),
    ("float32", precise, node_for(precise, target_triple), node_for(precise, source_triple)),
):
    if target is None or source is None:
        print(f"{label}: the feature pair is not among this graph's selected features")
        record[label] = {"present": False}
        continue
    run = FrozenRun.from_model(backbone, graph.input_tokens.unsqueeze(0).cuda())
    adjacency = float(graph.adjacency_matrix[target, source])
    effect = float(edge_effect(model, graph, run, target, source))
    selected_target = int(graph.selected_features[target])
    selected_source = int(graph.selected_features[source])
    record[label] = {
        "present": True,
        "target_node": target,
        "source_node": source,
        "adjacency": adjacency,
        "edge_effect": effect,
        "ratio": effect / adjacency if adjacency else None,
        "target_activation": float(graph.activation_values[selected_target]),
        "source_activation": float(graph.activation_values[selected_source]),
        "graph_dtype": str(getattr(graph.cfg, "dtype", "unknown")),
    }
    print(
        f"{label}: adjacency {adjacency:+.6f}  effect {effect:+.6f}  ratio {effect / adjacency:.4f}"
    )

with open(args.out, "w") as handle:
    json.dump(record, handle, indent=1)
print("wrote", args.out)
