"""Measure the rank and concentration of the feature-feature contribution block.

Experiment 004 ruled out low-rank structure in the weights and found only a factor of two in the
score matrix. The object Anthropic's remark is about is the feature-pair matrix, and it has a
structural bound the others do not: it factors through head space, so its rank is at most d_head
however many features there are. What matters is whether the effective rank sits well below that,
and whether the mass concentrates in few pairs, which is what would make pruning work.
"""

from __future__ import annotations

import json

import torch
from circuit_tracer.graph import Graph
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
from transformer_lens import HookedTransformer

from qk_attribution.attribution import qk_attribution, residual_remainder
from qk_attribution.circuits import effective_rank, kv_head_for
from qk_attribution.features import feature_sources, remainder_sources
from qk_attribution.scores import (
    layernorm_scale,
    qk_norm_scales,
    rotation_matrices,
)

MODEL = "Qwen/Qwen3-0.6B"
GRAPH = "spike_out/graph.pt"
SEED = 0
LAYERS = (7, 14, 20, 27)
ENERGIES = (0.9, 0.99, 0.999)
RANKS = (1, 2, 4, 8, 16, 32, 64)
TOP_FRACTIONS = (0.001, 0.01, 0.1)

torch.manual_seed(SEED)
torch.set_grad_enabled(False)

graph = Graph.from_pt(GRAPH)
assert isinstance(graph.scan, str), "a multi-scan graph names several transcoder sets"
transcoders, _ = load_transcoder_from_hub(
    graph.scan,
    device=torch.device("cuda"),
    dtype=torch.float32,
    lazy_encoder=True,
    lazy_decoder=False,
)
model = HookedTransformer.from_pretrained_no_processing(MODEL, device="cuda", dtype=torch.float32)
tokens = graph.input_tokens.unsqueeze(0).cuda()
seq = tokens.shape[1]
_, cache = model.run_with_cache(tokens)
query_position = seq - 1
rotations = rotation_matrices(model, seq)
d_head = model.cfg.d_head
print(
    f"model={MODEL} seq={seq} query_position={query_position} d_head={d_head} seed={SEED} "
    f"scan={graph.scan}"
)

rows = []
for layer in LAYERS:
    features = feature_sources(graph, transcoders, below_layer=layer)
    resid_pre = cache[f"blocks.{layer}.hook_resid_pre"][0]
    sources = features.concat(remainder_sources(residual_remainder(resid_pre, features)))
    at_query = sources.at_position(query_position)
    scales = qk_norm_scales(model, cache, layer)
    assert scales is not None
    norm_scale = layernorm_scale(cache, layer)

    for head in range(model.cfg.n_heads):
        result = qk_attribution(
            model,
            layer,
            head,
            query_position,
            query_sources=at_query,
            key_sources=sources,
            rotations=rotations,
            norm_scale=norm_scale,
            query_scale=scales[0][:, head],
            key_scale=scales[1][:, kv_head_for(model, head)],
        )
        block = result.contributions[~at_query.is_remainder][:, ~sources.is_remainder]
        if min(block.shape) < 2:
            continue
        values = torch.linalg.svdvals(block.float())
        truncation = {}
        u, s, vh = torch.linalg.svd(block.float(), full_matrices=False)
        for rank in RANKS:
            if rank > s.numel():
                continue
            approx = (u[:, :rank] * s[:rank]) @ vh[:rank]
            truncation[rank] = ((approx - block).norm() / block.norm()).item()

        magnitudes = block.abs().flatten()
        total = magnitudes.sum()
        ordered = magnitudes.sort(descending=True).values
        cumulative = ordered.cumsum(0) / total
        concentration = {
            fraction: cumulative[min(int(fraction * ordered.numel()), ordered.numel() - 1)].item()
            for fraction in TOP_FRACTIONS
        }

        rows.append(
            {
                "layer": layer,
                "head": head,
                "shape": list(block.shape),
                "max_possible_rank": min(min(block.shape), d_head),
                "effective_rank": {f"r{e}": effective_rank(values, energy=e) for e in ENERGIES},
                "truncation": truncation,
                "concentration": concentration,
            }
        )

for energy in ENERGIES:
    ranks = torch.tensor([float(r["effective_rank"][f"r{energy}"]) for r in rows])
    bound = torch.tensor([float(r["max_possible_rank"]) for r in rows])
    print(
        f"effective rank at {energy:<6}: median={ranks.median():.0f} min={ranks.min():.0f} "
        f"max={ranks.max():.0f} (bound median {bound.median():.0f})"
    )
for rank in RANKS:
    present = [r["truncation"][rank] for r in rows if rank in r["truncation"]]
    if present:
        values = torch.tensor(present)
        print(
            f"truncated to rank {rank:3d}: relative error median={values.median():.4f} "
            f"p90={values.quantile(0.9):.4f} max={values.max():.4f}"
        )
for fraction in TOP_FRACTIONS:
    values = torch.tensor([r["concentration"][fraction] for r in rows])
    print(
        f"top {fraction:.1%} of pairs hold: median={values.median():.3f} "
        f"min={values.min():.3f} max={values.max():.3f} of total magnitude"
    )
print("blocks measured:", len(rows), "| example shape", rows[0]["shape"] if rows else None)

with open("measure_feature_rank.json", "w") as fh:
    json.dump(
        {
            "model": MODEL,
            "graph": GRAPH,
            "scan": graph.scan,
            "seed": SEED,
            "seq": seq,
            "query_position": query_position,
            "d_head": d_head,
            "energies": list(ENERGIES),
            "ranks": list(RANKS),
            "rows": rows,
        },
        fh,
    )
print("peak GPU MiB:", torch.cuda.max_memory_allocated() // 2**20)
