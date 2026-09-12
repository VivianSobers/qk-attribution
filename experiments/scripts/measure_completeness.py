"""Split a real attention score into feature and non-feature parts, and check the parts add up.

Transcoder features cover the MLP writes into the residual stream. Everything else there, meaning
earlier attention outputs, the token embedding, transcoder errors and decoder biases, is carried as
one lumped direction per position. The bilinear expansion then has four blocks, and their total
must equal the score the model computed. That equality is the check; the split between blocks is
the result.
"""

from __future__ import annotations

import json

import torch
from common import load, parser, sampled_heads

from qk_attribution.attribution import qk_attribution, residual_remainder
from qk_attribution.circuits import kv_head_for
from qk_attribution.features import feature_sources, remainder_sources
from qk_attribution.scores import (
    attention_input,
    attention_scores,
    layernorm_scale,
    qk_norm_scales,
    rotation_matrices,
)

args = parser(__doc__.splitlines()[0]).parse_args()
fixtures = load(args)
graph, transcoders, model, cache = (
    fixtures.graph,
    fixtures.transcoders,
    fixtures.model,
    fixtures.cache,
)
seq = fixtures.seq
query_position = seq - 1
rotations = rotation_matrices(model, seq)
heads = sampled_heads(model)
print(f"query_position={query_position} heads={heads}")

rows = []
for layer, head in heads:
    residual = attention_input(model, cache, layer)
    scales = qk_norm_scales(model, cache, layer)
    assert scales is not None
    scale_q = scales[0][:, head]
    scale_k = scales[1][:, kv_head_for(model, head)]
    truth = attention_scores(
        model,
        layer,
        head,
        residual,
        query_scale=scale_q,
        key_scale=scale_k,
        rotations=rotations,
    )[query_position]

    features = feature_sources(graph, transcoders, below_layer=layer)
    resid_pre = cache[f"blocks.{layer}.hook_resid_pre"][0]
    sources = features.concat(remainder_sources(residual_remainder(resid_pre, features)))
    at_query = sources.at_position(query_position)
    result = qk_attribution(
        model,
        layer,
        head,
        query_position,
        query_sources=at_query,
        key_sources=sources,
        rotations=rotations,
        norm_scale=layernorm_scale(cache, layer),
        query_scale=scale_q,
        key_scale=scale_k,
    )
    total = result.by_key_position(seq)

    query_is_feature = ~at_query.is_remainder
    key_is_feature = ~sources.is_remainder

    def block(
        query_feature: bool,
        key_feature: bool,
        *,
        rows=query_is_feature,
        cols=key_is_feature,
        contributions=result.contributions,
        positions=sources.positions,
    ) -> torch.Tensor:
        """Sum one quadrant of the expansion onto key positions."""
        rows_mask = rows if query_feature else ~rows
        cols_mask = cols if key_feature else ~cols
        part = contributions[rows_mask][:, cols_mask]
        out = torch.zeros(seq, device=part.device, dtype=part.dtype)
        out.index_add_(0, positions[cols_mask], part.sum(dim=0))
        return out

    blocks = {
        "feature_feature": block(True, True),
        "feature_remainder": block(True, False),
        "remainder_feature": block(False, True),
        "remainder_remainder": block(False, False),
    }
    reassembled = sum(blocks.values())
    row = {
        "layer": layer,
        "head": head,
        "n_query_sources": len(at_query),
        "n_key_sources": len(sources),
        "n_query_features": int(query_is_feature.sum()),
        "n_key_features": int(key_is_feature.sum()),
        "exactness": ((total - truth).norm() / truth.norm()).item(),
        "block_split_error": ((reassembled - total).norm() / total.norm()).item(),
        "share": {name: (value.norm() / truth.norm()).item() for name, value in blocks.items()},
    }
    rows.append(row)
    shares = " ".join(
        f"{name.split('_')[0][0]}{name.split('_')[1][0]}={value:.3f}"
        for name, value in row["share"].items()
    )
    print(
        f"layer {layer:2d} head {head:2d}: q={row['n_query_features']:4d}f "
        f"k={row['n_key_features']:5d}f | exactness={row['exactness']:.2e} "
        f"| split_error={row['block_split_error']:.2e} | {shares}"
    )

with open(args.out, "w") as fh:
    json.dump({**fixtures.config(), "query_position": query_position, "rows": rows}, fh)
print("wrote", args.out)
print("peak GPU MiB:", torch.cuda.max_memory_allocated() // 2**20)
