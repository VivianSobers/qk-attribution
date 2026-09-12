"""Check the attributed contributions against what actually happens when you remove the feature.

Every result so far is observational. This intervenes. For each qualifying head, take the strongest
key-side feature, delete its direction from the residual stream, and compare the attention score the
model then computes against the change the decomposition predicted.

Two interventions, answering different questions.

The tight one removes the direction at the attention layer's own input, which is exactly the
quantity the decomposition is a decomposition of. Predicted and actual should agree to machine
precision; if they do not, the attribution numbers are wrong.

The realistic one removes the feature where it is actually written, at its own layer's MLP output,
and lets everything downstream respond. It includes paths the decomposition does not model, so it
answers whether the named feature matters rather than whether the arithmetic is right.

The control repeats the realistic intervention on a different feature at the same position with a
comparable activation, which says whether any feature would have done.
"""

from __future__ import annotations

import json

import torch
from common import load, parser, sampled_layers

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

extra = parser(__doc__.splitlines()[0])
extra.add_argument("--content-threshold", type=float, default=0.2)
extra.add_argument("--max-heads", type=int, default=24)
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
print("tokens:", " | ".join(f"{i}:{t!r}" for i, t in enumerate(tokens)))


def _pin(name: str):
    """Hold a hook at its clean value, the way an attribution graph does."""

    def hook(value, hook):  # noqa: A002, ARG001
        return cache[name]

    return (name, hook)


def _remove(name: str, position: int, direction):
    def hook(value, hook):  # noqa: A002, ARG001
        patched = value.clone()
        patched[:, position, :] -= direction.to(patched.dtype)
        return patched

    return (name, hook)


def scores_with_direction_removed(layer: int, head: int, position: int, direction, *, freeze: bool):
    """Run the model with a direction deleted from one position of one layer's input.

    With ``freeze`` the normalisation scales are held at their clean values, which is the regime
    the decomposition describes. Without it they are recomputed, which is what the model would
    really do.
    """
    hooks = [_remove(f"blocks.{layer}.hook_resid_pre", position, direction)]
    if freeze:
        hooks += [
            _pin(f"blocks.{layer}.ln1.hook_scale"),
            _pin(f"blocks.{layer}.attn.q_norm.hook_scale"),
            _pin(f"blocks.{layer}.attn.k_norm.hook_scale"),
        ]
    with model.hooks(fwd_hooks=hooks):
        _, patched = model.run_with_cache(
            token_ids,
            names_filter=lambda n: n.endswith("hook_attn_scores"),
        )
    return patched[f"blocks.{layer}.attn.hook_attn_scores"][0, head]


def scores_with_feature_ablated(
    write_layer: int, read_layer: int, head: int, position: int, direction
):
    """Run the model with a feature removed where it is written, letting everything respond."""
    with model.hooks(
        fwd_hooks=[_remove(f"blocks.{write_layer}.hook_mlp_out", position, direction)]
    ):
        _, patched_cache = model.run_with_cache(
            token_ids, names_filter=lambda n: n.endswith("hook_pattern")
        )
    return patched_cache[f"blocks.{read_layer}.attn.hook_pattern"][0, head]


rows = []
for layer in sampled_layers(model, count=4):
    features = feature_sources(graph, transcoders, below_layer=layer)
    resid_pre = cache[f"blocks.{layer}.hook_resid_pre"][0]
    sources = features.concat(remainder_sources(residual_remainder(resid_pre, features)))
    at_query = sources.at_position(query_position)
    at_query_index = torch.nonzero(sources.positions == query_position).flatten()
    scales = qk_norm_scales(model, cache, layer)
    assert scales is not None
    norm_scale = layernorm_scale(cache, layer)
    patterns = cache[f"blocks.{layer}.attn.hook_pattern"][0]
    residual = attention_input(model, cache, layer)
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

        scale_q = scales[0][:, head]
        scale_k = scales[1][:, kv_head_for(model, head)]
        result = qk_attribution(
            model,
            layer,
            head,
            query_position,
            query_sources=at_query,
            key_sources=sources,
            rotations=rotations,
            norm_scale=norm_scale,
            query_scale=scale_q,
            key_scale=scale_k,
        )
        query_block = result.contributions
        block = result.contributions[~at_query.is_remainder][:, key_feature]
        if block.numel() == 0:
            continue
        by_key = block.sum(dim=0)
        best = int(by_key.abs().argmax())
        source = int(feature_rows[best])
        position = int(sources.positions[source])
        direction = sources.directions[source]
        # The score is bilinear, so deleting this key-side source removes its interaction with
        # every query-side source, the remainder included. Summing over query features alone
        # leaves out the largest part of the query side and predicts the wrong change.
        predicted = -float(result.contributions[:, source].sum())
        if position == query_position:
            # The direction sits at the query position too, so the same deletion moves both
            # arguments: s -> <q - d, k - d> = s - <d, k> - <q, d> + <d, d>. The second and third
            # terms are the query-side row and the source's interaction with itself.
            query_index = int(torch.nonzero(at_query_index == source).flatten()[0])
            # Only key sources at this position contribute to the score at (q, q), so the
            # query-side term is that row restricted to them, not the whole row.
            here = sources.positions == query_position
            predicted += -float(query_block[query_index][here].sum()) + float(
                query_block[query_index, source]
            )

        baseline_scores = attention_scores(
            model,
            layer,
            head,
            residual,
            query_scale=scale_q,
            key_scale=scale_k,
            rotations=rotations,
        )[query_position]
        frozen = scores_with_direction_removed(layer, head, position, direction, freeze=True)
        free = scores_with_direction_removed(layer, head, position, direction, freeze=False)
        actual = float(frozen[query_position, position] - baseline_scores[position])
        actual_free = float(free[query_position, position] - baseline_scores[position])

        # A different feature at the same position, with the closest activation, as a control.
        same_place = (sources.positions == position) & key_feature
        same_place[source] = False
        candidates = torch.nonzero(same_place).flatten()
        control_drop = None
        if candidates.numel():
            gaps = (sources.activations[candidates] - sources.activations[source]).abs()
            control = int(candidates[int(gaps.argmin())])
            control_pattern = scores_with_feature_ablated(
                int(sources.layers[control]), layer, head, position, sources.directions[control]
            )
            control_drop = float(
                control_pattern[query_position, position] - patterns[head, query_position, position]
            )

        realistic = scores_with_feature_ablated(
            int(sources.layers[source]), layer, head, position, direction
        )
        pattern_drop = float(
            realistic[query_position, position] - patterns[head, query_position, position]
        )

        rows.append(
            {
                "layer": layer,
                "head": head,
                "key_position": position,
                "key_token": tokens[position],
                "feature": f"L{int(sources.layers[source])}#{int(sources.feature_ids[source])}",
                "predicted_score_change": predicted,
                "actual_score_change": actual,
                "actual_score_change_unfrozen": actual_free,
                "baseline_attention": float(patterns[head, query_position, position]),
                "attention_change": pattern_drop,
                "control_attention_change": control_drop,
            }
        )
        print(
            f"L{layer:2d}H{head:2d} pos {position} {tokens[position]!r:14s} "
            f"predicted {predicted:+8.4f} frozen {actual:+8.4f} free {actual_free:+8.4f} "
            f"attn {rows[-1]['baseline_attention']:.3f} -> change {pattern_drop:+.4f}"
            + (f" (control {control_drop:+.4f})" if control_drop is not None else "")
        )

if rows:
    predicted = torch.tensor([r["predicted_score_change"] for r in rows])
    actual = torch.tensor([r["actual_score_change"] for r in rows])
    unfrozen = torch.tensor([r["actual_score_change_unfrozen"] for r in rows])
    error = (predicted - actual).abs() / actual.abs().clamp_min(1e-6)
    free_error = (predicted - unfrozen).abs() / unfrozen.abs().clamp_min(1e-6)
    drops = torch.tensor([r["attention_change"] for r in rows])
    controls = torch.tensor(
        [r["control_attention_change"] for r in rows if r["control_attention_change"] is not None]
    )
    print(f"\nheads: {len(rows)}")
    # Relative error explodes wherever the true change is near zero, so the share of heads within
    # a tolerance says more about agreement than the worst case does.
    within = int((error < 1e-4).sum())
    print(
        f"predicted vs actual score change: median relative error {error.median():.3e}; "
        f"{within}/{len(rows)} heads agree to better than 1e-04"
    )
    print(
        f"largest absolute disagreement: "
        f"{(predicted - actual).abs().max():.3e} on changes of size "
        f"{actual.abs().median():.3f} (median)"
    )
    print(
        f"correlation with frozen scales: "
        f"{torch.corrcoef(torch.stack([predicted, actual]))[0, 1]:.6f}"
    )
    print(
        f"predicted vs actual with scales free: median relative error "
        f"{free_error.median():.3e}, correlation "
        f"{torch.corrcoef(torch.stack([predicted, unfrozen]))[0, 1]:.6f}"
    )
    print(
        f"attention change when the named feature is ablated: median {drops.median():+.4f}, "
        f"mean {drops.mean():+.4f}"
    )
    if controls.numel():
        print(
            f"attention change for a control feature at the same position: "
            f"median {controls.median():+.4f}, mean {controls.mean():+.4f}"
        )
        bigger = int((drops[: controls.numel()].abs() > controls.abs()).sum())
        print(
            f"named feature moved attention more than the control in "
            f"{bigger}/{controls.numel()} heads"
        )

with open(args.out, "w") as fh:
    json.dump({**fixtures.config(), "tokens": tokens, "rows": rows}, fh, indent=1)
print("wrote", args.out)
