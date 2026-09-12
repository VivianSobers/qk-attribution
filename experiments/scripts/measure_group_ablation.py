"""How many of the attributed features must be removed before attention actually moves?

Experiment 011 found that deleting the single largest key-side feature moves the attention pattern
no more than deleting an arbitrary one. That is consistent with the cause being distributed rather
than with the attribution being wrong, but it does not distinguish the two. This does: ablate the
top k features together, for growing k, and watch the pattern.

If the attribution ranks features usefully, removing the top k should move attention further than
removing k features chosen at random, and the gap should appear well before k reaches the total.
If the two curves lie on top of each other, the ranking carries no information about the pattern.

Movement is the total variation distance between the head's attention row before and after, which
counts every position rather than only the one the top feature pointed at.
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
extra.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10, 25, 50, 100, 250])
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
print("tokens:", " | ".join(f"{i}:{t!r}" for i, t in enumerate(tokens)))


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
            model, layer, head, query_position,
            query_sources=at_query, key_sources=sources,
            rotations=rotations, norm_scale=norm_scale,
            query_scale=scales[0][:, head], key_scale=scales[1][:, kv_head_for(model, head)],
        )
        by_key = result.contributions.sum(dim=0)[key_feature]
        order = by_key.abs().argsort(descending=True)
        ranked = [int(feature_rows[i]) for i in order.tolist()]
        pool = list(ranked)
        baseline = patterns[head, query_position]

        by_position: dict[int, list[int]] = {}
        for index in ranked:
            by_position.setdefault(int(sources.positions[index]), []).append(index)

        entry = {"layer": layer, "head": head, "n_features": len(ranked), "curve": {}}
        for k in args.ks:
            if k > len(ranked):
                continue
            top = ablate(ranked[:k], sources, layer, head)
            control = ablate(rng.sample(pool, k), sources, layer, head)
            # A stricter control: the same number of features drawn from the same positions as the
            # top k, so the comparison is not simply between attended and unattended positions.
            matched: list[int] = []
            wanted: dict[int, int] = {}
            for index in ranked[:k]:
                position = int(sources.positions[index])
                wanted[position] = wanted.get(position, 0) + 1
            for position, count in wanted.items():
                candidates = [i for i in by_position[position] if i not in set(ranked[:k])]
                matched += rng.sample(candidates, min(count, len(candidates)))
            matched_row = ablate(matched, sources, layer, head) if matched else baseline
            entry["curve"][k] = {
                "top": movement(baseline, top),
                "control": movement(baseline, control),
                "matched": movement(baseline, matched_row),
                "n_matched": len(matched),
            }
        rows.append(entry)
        shown = " ".join(
            f"k={k}:{v['top']:.3f}/{v['matched']:.3f}" for k, v in entry["curve"].items()
        )
        print(f"L{layer:2d}H{head:2d} ({len(ranked)} features) {shown}")

print("\nmovement in total variation distance, median over heads:")
print(f"  {'k':>5s} {'top-k':>8s} {'same-position':>14s} {'random':>8s} "
      f"{'vs position':>12s} {'wins':>7s}")
for k in args.ks:
    present = [r["curve"][k] for r in rows if k in r["curve"]]
    if not present:
        continue
    tops = torch.tensor([c["top"] for c in present])
    controls = torch.tensor([c["control"] for c in present])
    matched = torch.tensor([c["matched"] for c in present])
    wins = int((tops > matched).sum())
    print(
        f"  {k:5d} {tops.median():8.4f} {matched.median():14.4f} {controls.median():8.4f} "
        f"{(tops.median() / matched.median().clamp_min(1e-9)):12.2f} "
        f"{wins:3d}/{len(present):<3d}"
    )

with open(args.out, "w") as fh:
    json.dump({**fixtures.config(), "tokens": tokens, "ks": args.ks, "rows": rows}, fh, indent=1)
print("wrote", args.out)
