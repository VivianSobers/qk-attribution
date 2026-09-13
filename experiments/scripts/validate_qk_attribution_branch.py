"""Validate the circuit-tracer QK attribution port against a real model's attention scores.

The port was verified on a CPU stub with an independent reference forward pass. That stub
implements rotary embeddings and QK-norm the way the port assumes a model does, so it cannot catch
an assumption both share. This checks the port against Qwen3's own ``hook_attn_scores``, which is
what experiment 003 did for the original implementation, and checks that the source-pair
decomposition sums to those scores on sampled heads.

The module is imported from a copy of the branch file placed on the path as
``qk_attribution_branch``, so it runs against the installed circuit-tracer without replacing it.
"""

from __future__ import annotations

import argparse
import json
import random
from types import SimpleNamespace

import torch
from circuit_tracer.graph import Graph
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
from qk_attribution_branch import FrozenScores, attention_scores, qk_attribution
from transformer_lens import HookedTransformer

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--graph", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--branch-commit", required=True)
parser.add_argument("--heads-per-layer", type=int, default=2, help="sampled for the decomposition")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.set_grad_enabled(False)
rng = random.Random(args.seed)

graph = Graph.from_pt(args.graph)
transcoders, _ = load_transcoder_from_hub(
    graph.scan,
    device=torch.device("cuda"),
    dtype=torch.float32,
    lazy_encoder=True,
    lazy_decoder=True,
)
backbone = HookedTransformer.from_pretrained_no_processing(
    args.model, device="cuda", dtype=torch.float32
)
model = SimpleNamespace(
    cfg=backbone.cfg,
    blocks=backbone.blocks,
    transcoders=transcoders,
    run_with_cache=backbone.run_with_cache,
)
tokens = graph.input_tokens.unsqueeze(0).cuda()
run = FrozenScores.from_model(model, tokens)
_, cache = backbone.run_with_cache(tokens, names_filter=lambda n: n.endswith("hook_attn_scores"))
n_pos = run.n_pos
causal = torch.tril(torch.ones(n_pos, n_pos, dtype=torch.bool, device="cuda"))


def relative(ours: torch.Tensor, theirs: torch.Tensor) -> float:
    return float((ours - theirs).norm() / theirs.norm().clamp_min(1e-30))


reconstruction = []
for layer in range(model.cfg.n_layers):
    theirs_all = cache[f"blocks.{layer}.attn.hook_attn_scores"][0].float()
    for head in range(model.cfg.n_heads):
        ours = attention_scores(model, run, layer, head)
        reconstruction.append(
            {
                "layer": layer,
                "head": head,
                "relative_error": relative(ours[causal], theirs_all[head][causal]),
            }
        )
    worst = max(r["relative_error"] for r in reconstruction[-model.cfg.n_heads :])
    print(f"layer {layer:2d}: worst head error {worst:.2e}")

decomposition = []
query_position = n_pos - 1
for layer in range(1, model.cfg.n_layers):
    theirs_all = cache[f"blocks.{layer}.attn.hook_attn_scores"][0].float()
    for head in rng.sample(range(model.cfg.n_heads), args.heads_per_layer):
        result = qk_attribution(model, graph, run, layer, head, query_position)
        row = theirs_all[head, query_position, : query_position + 1]
        got = result.by_key_position(n_pos)[: query_position + 1]
        features = ~result.key_sources.is_remainder
        feature_query = ~result.query_sources.is_remainder
        feature_pairs = result.contributions[feature_query][:, features]
        decomposition.append(
            {
                "layer": layer,
                "head": head,
                "relative_error": relative(got, row),
                "n_query_sources": len(result.query_sources),
                "n_key_sources": len(result.key_sources),
                "feature_pair_share": float(
                    feature_pairs.abs().sum() / result.contributions.abs().sum().clamp_min(1e-30)
                ),
            }
        )

recon_errors = torch.tensor([r["relative_error"] for r in reconstruction])
decomp_errors = torch.tensor([r["relative_error"] for r in decomposition])
summary = {
    "graph": args.graph,
    "model": args.model,
    "scan": graph.scan,
    "branch_commit": args.branch_commit,
    "seed": args.seed,
    "dtype": "float32",
    "n_pos": n_pos,
    "reconstruction_median": float(recon_errors.median()),
    "reconstruction_max": float(recon_errors.max()),
    "n_heads_checked": len(reconstruction),
    "decomposition_median": float(decomp_errors.median()),
    "decomposition_max": float(decomp_errors.max()),
    "n_decompositions": len(decomposition),
    "reconstruction": reconstruction,
    "decomposition": decomposition,
}
print(
    f"\nscore reconstruction over {len(reconstruction)} heads:"
    f" median {summary['reconstruction_median']:.2e}"
    f", max {summary['reconstruction_max']:.2e}"
)
print(
    f"decomposition over {len(decomposition)} heads: median {summary['decomposition_median']:.2e}"
    f", max {summary['decomposition_max']:.2e}"
)
with open(args.out, "w") as handle:
    json.dump(summary, handle, indent=1)
print("wrote", args.out)
