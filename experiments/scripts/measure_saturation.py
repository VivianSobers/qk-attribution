"""How many attributed features does an explanation of one head actually need?

Experiment 012 showed the ranking carries information: removing the top k features moves attention
further than removing k features from the same positions. It stopped at k=250 and never reached
the point where the curve flattens, so it bounds nothing. An explanation that needs 250 features is
not an explanation.

This measures the whole curve, out to ablating every attributed feature, and normalises by that
endpoint. The endpoint is the largest movement this decomposition can produce for the head, so
``k90``, the smallest k reaching 90% of it, is the number of features an explanation of that head
would have to name. Against it runs a random-order arm over the same pool: if the ranking were
uninformative the two would need the same k.

Movement is the total variation distance between the head's attention row before and after.
"""

from __future__ import annotations

import json
import random

import torch
from common import load, parser, sampled_layers

from qk_attribution.attribution import qk_attribution, residual_remainder
from qk_attribution.circuits import kv_head_for
from qk_attribution.features import feature_sources, remainder_sources
from qk_attribution.scores import layernorm_scale, qk_norm_scales, rotation_matrices

extra = parser(__doc__.splitlines()[0])
extra.add_argument("--content-threshold", type=float, default=0.2)
extra.add_argument("--max-heads", type=int, default=12)
extra.add_argument("--points", type=int, default=14, help="k values per head, geometrically spaced")
extra.add_argument("--repeats", type=int, default=3, help="random draws averaged at each k")
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
token_ids = graph.input_tokens.unsqueeze(0).cuda()
rng = random.Random(args.seed)
print(fixtures.header())
print("tokens:", " | ".join(f"{i}:{t!r}" for i, t in enumerate(tokens)))


def geometric_ks(total: int, points: int) -> list[int]:
    """A k grid from 1 to ``total`` inclusive, spaced so the small end is resolved."""
    if total <= points:
        return list(range(1, total + 1))
    ratio = total ** (1.0 / (points - 1))
    grid = sorted({min(total, max(1, round(ratio**i))) for i in range(points)})
    if grid[-1] != total:
        grid.append(total)
    return grid


def ablate(chosen, sources, read_layer: int, head: int) -> torch.Tensor:
    """Delete a set of features where each is written, and return the head's attention row."""
    removals: dict[tuple[int, int], torch.Tensor] = {}
    for index in chosen:
        key = (int(sources.layers[index]), int(sources.positions[index]))
        vector = sources.directions[index]
        removals[key] = removals.get(key, torch.zeros_like(vector)) + vector

    def make(position: int, total: torch.Tensor):
        def hook(value, hook):  # noqa: A002, ARG001
            patched = value.clone()
            patched[:, position, :] -= total.to(patched.dtype)
            return patched

        return hook

    hooks = [
        (f"blocks.{layer}.hook_mlp_out", make(position, total))
        for (layer, position), total in removals.items()
    ]
    with model.hooks(fwd_hooks=hooks):
        _, patched = model.run_with_cache(
            token_ids, names_filter=lambda n: n.endswith("hook_pattern")
        )
    return patched[f"blocks.{read_layer}.attn.hook_pattern"][0, head, query_position]


def movement(before: torch.Tensor, after: torch.Tensor) -> float:
    """Total variation distance between two attention rows."""
    return float(0.5 * (after - before).abs().sum())


def first_reaching(curve: dict[int, float], fraction: float, full: float) -> int | None:
    """Smallest k whose movement reaches ``fraction`` of the full-ablation movement.

    The curve is not monotone: features can cancel, so a larger set sometimes moves attention less.
    Taking the first crossing rather than the last is the conservative reading of "enough".
    """
    if full <= 0.0:
        return None
    for k in sorted(curve):
        if curve[k] >= fraction * full:
            return k
    return None


rows = []
for layer in sampled_layers(model, count=3):
    features = feature_sources(graph, transcoders, below_layer=layer)
    resid_pre = cache[f"blocks.{layer}.hook_resid_pre"][0]
    sources = features.concat(remainder_sources(residual_remainder(resid_pre, features)))
    at_query = sources.at_position(query_position)
    scales = qk_norm_scales(model, cache, layer)
    assert scales is not None
    norm_scale = layernorm_scale(cache, layer)
    patterns = cache[f"blocks.{layer}.attn.hook_pattern"][0]
    key_feature = ~sources.is_remainder
    feature_rows = torch.nonzero(key_feature).flatten()

    for head in range(model.cfg.n_heads):
        if len(rows) >= args.max_heads:
            break
        row = patterns[head, query_position, : query_position + 1].clone()
        row[0] = 0.0
        row[query_position] = 0.0
        if float(row.sum()) < args.content_threshold:
            continue

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
        by_key = result.contributions.sum(dim=0)[key_feature]
        order = by_key.abs().argsort(descending=True)
        ranked = [int(feature_rows[i]) for i in order.tolist()]
        baseline = patterns[head, query_position]
        full = movement(baseline, ablate(ranked, sources, layer, head))

        ks = geometric_ks(len(ranked), args.points)
        top_curve: dict[int, float] = {}
        random_curve: dict[int, float] = {}
        for k in ks:
            top_curve[k] = movement(baseline, ablate(ranked[:k], sources, layer, head))
            draws = [
                movement(baseline, ablate(rng.sample(ranked, k), sources, layer, head))
                for _ in range(args.repeats)
            ]
            random_curve[k] = sum(draws) / len(draws)

        entry = {
            "layer": layer,
            "head": head,
            "n_features": len(ranked),
            "full_movement": full,
            "ks": ks,
            "top": top_curve,
            "random": random_curve,
            "k50": first_reaching(top_curve, 0.5, full),
            "k90": first_reaching(top_curve, 0.9, full),
            "random_k50": first_reaching(random_curve, 0.5, full),
            "random_k90": first_reaching(random_curve, 0.9, full),
        }
        rows.append(entry)
        print(
            f"L{layer:2d}H{head:2d} {len(ranked):5d} features  full TV {full:.4f}  "
            f"k50 {entry['k50']} (random {entry['random_k50']})  "
            f"k90 {entry['k90']} (random {entry['random_k90']})"
        )


def summarise(field: str) -> str:
    values = [r[field] for r in rows if r[field] is not None]
    if not values:
        return "n/a"
    shares = [r[field] / r["n_features"] for r in rows if r[field] is not None]
    return (
        f"median {int(torch.tensor(values, dtype=torch.float).median())} features "
        f"({torch.tensor(shares).median():.1%} of the pool), "
        f"{len(values)}/{len(rows)} heads reached it"
    )


print("\nfeatures needed to reach a share of the full-ablation movement:")
for field in ("k50", "random_k50", "k90", "random_k90"):
    print(f"  {field:12s} {summarise(field)}")

with open(args.out, "w") as fh:
    json.dump(
        {**fixtures.config(), "tokens": tokens, "repeats": args.repeats, "rows": rows}, fh, indent=1
    )
print("wrote", args.out)
