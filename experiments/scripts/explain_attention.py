"""Explain why one head attends where it does, in words rather than numbers.

Takes a (layer, head, query position), decomposes the attention scores into feature-pair terms, and
prints the strongest pairs with each feature's label. This is what the whole method is for: the
attribution graph already says information moved from one position to another, and this says why
the head looked there.

The remainder terms are printed alongside, since they usually carry more of the score than the
feature terms do and hiding that would misrepresent the result.
"""

from __future__ import annotations

import json

import torch
from common import load, parser

from qk_attribution.attribution import qk_attribution, residual_remainder
from qk_attribution.circuits import kv_head_for
from qk_attribution.features import feature_sources, remainder_sources
from qk_attribution.labels import FeatureStore
from qk_attribution.scores import (
    attention_input,
    attention_scores,
    layernorm_scale,
    qk_norm_scales,
    rotation_matrices,
)

extra = parser(__doc__.splitlines()[0])
extra.add_argument("--layer", type=int, default=None, help="defaults to three quarters depth")
extra.add_argument("--head", type=int, default=None, help="defaults to the head with the "
                   "most concentrated attention from the query position")
extra.add_argument("--query-position", type=int, default=None, help="defaults to the last token")
extra.add_argument("--pairs", type=int, default=12)
extra.add_argument("--no-labels", action="store_true", help="skip the metadata downloads")
args = extra.parse_args()

fixtures = load(args)
graph, transcoders, model, cache = (
    fixtures.graph,
    fixtures.transcoders,
    fixtures.model,
    fixtures.cache,
)
seq = fixtures.seq
layer = args.layer if args.layer is not None else (3 * model.cfg.n_layers) // 4
query_position = args.query_position if args.query_position is not None else seq - 1
tokens = [model.tokenizer.decode([int(t)]) for t in graph.input_tokens]
print("prompt tokens:", " | ".join(f"{i}:{t!r}" for i, t in enumerate(tokens)))

patterns = cache[f"blocks.{layer}.attn.hook_pattern"][0]
if args.head is not None:
    head = args.head
else:
    # Most heads put nearly all their mass on the first token, which is an attention sink and
    # explains nothing. Pick the head that attends most to actual content: any position other
    # than the sink and the query's own.
    row = patterns[:, query_position, : query_position + 1].clone()
    row[:, 0] = 0.0
    if query_position < row.shape[1]:
        row[:, query_position] = 0.0
    content_mass = row.sum(dim=-1)
    head = int(content_mass.argmax())
    print(
        f"head {head} chosen: {content_mass[head]:.3f} of its attention lands on content, "
        f"against a median of {content_mass.median():.3f} across heads"
    )
print(f"layer={layer} head={head} query_position={query_position} ({tokens[query_position]!r})")

attention = patterns[head, query_position, : query_position + 1]
ranked_keys = attention.argsort(descending=True)[:5].tolist()
print("attends to: " + ", ".join(f"{k}:{tokens[k]!r} p={attention[k]:.3f}" for k in ranked_keys))

residual = attention_input(model, cache, layer)
scales = qk_norm_scales(model, cache, layer)
assert scales is not None
scale_q = scales[0][:, head]
scale_k = scales[1][:, kv_head_for(model, head)]
rotations = rotation_matrices(model, seq)
truth = attention_scores(
    model, layer, head, residual, query_scale=scale_q, key_scale=scale_k, rotations=rotations
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
error = ((result.by_key_position(seq) - truth).norm() / truth.norm()).item()
print(f"decomposition reproduces the scores to {error:.2e} relative")

query_feature = ~at_query.is_remainder
key_feature = ~sources.is_remainder
block = result.contributions[query_feature][:, key_feature]
query_rows = torch.nonzero(query_feature).flatten()
key_rows = torch.nonzero(key_feature).flatten()

store = None if args.no_labels else FeatureStore(fixtures.graph.scan)


def describe(source_index: int) -> str:
    layer_id = int(sources.layers[source_index])
    feature_id = int(sources.feature_ids[source_index])
    if store is None:
        return f"L{layer_id}#{feature_id}"
    try:
        return store.card(layer_id, feature_id).label()
    except Exception as exc:  # noqa: BLE001
        return f"L{layer_id}#{feature_id} (label unavailable: {type(exc).__name__})"


print(f"\ntop {args.pairs} feature pairs (of {block.numel()}):")
values, flat = block.abs().flatten().topk(min(args.pairs, block.numel()))
rows = []
for rank, index in enumerate(flat.tolist()):
    row, column = divmod(index, block.shape[1])
    q_source = int(query_rows[row])
    k_source = int(key_rows[column])
    contribution = float(block[row, column])
    key_position = int(sources.positions[k_source])
    entry = {
        "rank": rank,
        "contribution": contribution,
        "key_position": key_position,
        "key_token": tokens[key_position],
        "query": describe(q_source),
        "key": describe(k_source),
    }
    rows.append(entry)
    print(f"  {contribution:+8.4f}  key pos {key_position} {tokens[key_position]!r}")
    print(f"            query {entry['query']}")
    print(f"            key   {entry['key']}")

# Summing over the query side gives the cleaner reading: which feature at which key position
# drew the head there, regardless of which query-side feature it matched.
by_key = block.sum(dim=0)
print(f"\ntop key-side features (summed over all {block.shape[0]} query features):")
key_values, key_flat = by_key.abs().topk(min(args.pairs, by_key.numel()))
key_rows_out = []
for index in key_flat.tolist():
    k_source = int(key_rows[index])
    position = int(sources.positions[k_source])
    entry = {
        "contribution": float(by_key[index]),
        "key_position": position,
        "key_token": tokens[position],
        "key": describe(k_source),
    }
    key_rows_out.append(entry)
    print(f"  {entry['contribution']:+8.4f}  pos {position} {tokens[position]!r}  {entry['key']}")

feature_share = (
    block.sum(dim=0).abs().sum() / result.contributions.abs().sum()
).item()
print(f"\nfeature-feature terms are {feature_share:.1%} of the total contribution magnitude")

with open(args.out, "w") as fh:
    json.dump(
        {
            **fixtures.config(),
            "layer": layer,
            "head": head,
            "query_position": query_position,
            "tokens": tokens,
            "attends_to": [
                {"position": k, "token": tokens[k], "probability": float(attention[k])}
                for k in ranked_keys
            ],
            "reconstruction_error": error,
            "feature_share": feature_share,
            "pairs": rows,
            "top_key_features": key_rows_out,
        },
        fh,
        indent=1,
    )
print("wrote", args.out)
