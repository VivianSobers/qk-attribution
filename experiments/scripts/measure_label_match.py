"""Test whether QK explanations point at the token they claim to.

Experiment 008 reads well on one head of one prompt. That is an anecdote. This turns it into a
number: across every head that attends to content, take the strongest key-side feature, and ask
whether that feature actually fires on the token sitting at the position it points to.

A feature counts as matching when the token at its key position appears among its top activating
tokens, compared case-insensitively and ignoring leading spaces. The control is the same test
against a randomly chosen other position in the prompt, which says how often a match happens by
coincidence given how common determiners and punctuation are.
"""

from __future__ import annotations

import json
import random

import torch
from common import load, parser, sampled_layers

from qk_attribution.attribution import qk_attribution, residual_remainder
from qk_attribution.circuits import kv_head_for
from qk_attribution.features import feature_sources, remainder_sources
from qk_attribution.labels import FeatureStore
from qk_attribution.scores import (
    layernorm_scale,
    qk_norm_scales,
    rotation_matrices,
)

extra = parser(__doc__.splitlines()[0])
extra.add_argument("--content-threshold", type=float, default=0.2,
                   help="minimum attention mass off the sink and off the query itself")
extra.add_argument("--top-tokens", type=int, default=8)
args = extra.parse_args()

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
tokens = [model.tokenizer.decode([int(t)]) for t in graph.input_tokens]
store = FeatureStore(fixtures.graph.scan)
rng = random.Random(args.seed)
print("tokens:", " | ".join(f"{i}:{t!r}" for i, t in enumerate(tokens)))


def normalise(text: str) -> str:
    return text.strip().lower()


def fires_on(card, token: str) -> bool:
    """Does this feature's own top activating tokens include the given one?"""
    target = normalise(token)
    if not target:
        return False
    return any(normalise(t) == target for t, _ in card.top_tokens[: args.top_tokens])


rows = []
for layer in sampled_layers(model, count=6):
    features = feature_sources(graph, transcoders, below_layer=layer)
    resid_pre = cache[f"blocks.{layer}.hook_resid_pre"][0]
    sources = features.concat(remainder_sources(residual_remainder(resid_pre, features)))
    at_query = sources.at_position(query_position)
    scales = qk_norm_scales(model, cache, layer)
    assert scales is not None
    norm_scale = layernorm_scale(cache, layer)
    patterns = cache[f"blocks.{layer}.attn.hook_pattern"][0]

    for head in range(model.cfg.n_heads):
        row = patterns[head, query_position, : query_position + 1].clone()
        row[0] = 0.0
        row[query_position] = 0.0
        content = float(row.sum())
        if content < args.content_threshold:
            continue

        result = qk_attribution(
            model, layer, head, query_position,
            query_sources=at_query, key_sources=sources,
            rotations=rotations, norm_scale=norm_scale,
            query_scale=scales[0][:, head], key_scale=scales[1][:, kv_head_for(model, head)],
        )
        key_feature = ~sources.is_remainder
        block = result.contributions[~at_query.is_remainder][:, key_feature]
        if block.numel() == 0:
            continue
        by_key = block.sum(dim=0)
        best = int(by_key.abs().argmax())
        source_index = int(torch.nonzero(key_feature).flatten()[best])
        position = int(sources.positions[source_index])
        feature_layer = int(sources.layers[source_index])
        feature_id = int(sources.feature_ids[source_index])
        try:
            card = store.card(feature_layer, feature_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  label unavailable for L{feature_layer}#{feature_id}: {type(exc).__name__}")
            continue

        others = [p for p in range(query_position + 1) if p != position]
        control_position = rng.choice(others) if others else position
        rows.append({
            "layer": layer,
            "head": head,
            "content_mass": content,
            "key_position": position,
            "key_token": tokens[position],
            "feature": f"L{feature_layer}#{feature_id}",
            "feature_tokens": [t for t, _ in card.top_tokens[: args.top_tokens]],
            "is_dead": card.is_dead,
            "match": fires_on(card, tokens[position]),
            "control_position": control_position,
            "control_match": fires_on(card, tokens[control_position]),
            "attention_to_key": float(patterns[head, query_position, position]),
        })

live = [r for r in rows if not r["is_dead"]]
matches = sum(r["match"] for r in live)
controls = sum(r["control_match"] for r in live)
print(f"\nheads examined: {len(rows)} ({len(live)} with a live top feature)")
if live:
    print(f"top key feature fires on the token it points to: {matches}/{len(live)} "
          f"= {matches / len(live):.1%}")
    print(f"same test against a random other position:       {controls}/{len(live)} "
          f"= {controls / len(live):.1%}")
    hit = [r for r in live if r["match"]]
    print("\nexamples that matched:")
    for r in hit[:8]:
        print(f"  L{r['layer']}H{r['head']} -> pos {r['key_position']} {r['key_token']!r} "
              f"{r['feature']} fires on {r['feature_tokens'][:4]}")

with open(args.out, "w") as fh:
    json.dump(
        {
            **fixtures.config(),
            "tokens": tokens,
            "content_threshold": args.content_threshold,
            "n_heads_examined": len(rows),
            "n_live": len(live),
            "n_match": matches,
            "n_control_match": controls,
            "rows": rows,
        },
        fh,
        indent=1,
    )
print("wrote", args.out)
