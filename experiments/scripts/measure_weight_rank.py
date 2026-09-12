"""Measure how much low-rank structure the QK operator has, and how many features it must cover.

The per-head operator is M(d) = A R(d) B.T with A = W_Q diag(w_q) and B = W_K diag(w_k), both
d_model x d_head. R(d) is an orthogonal rotation, so it cannot lower the rank: the rank of M is
bounded by the ranks of A and B for every offset at once. That makes the spectra of A and B the
offset-independent thing worth measuring.

Weight-space rank is only a bound on what matters, so this also truncates A and B and measures the
error in the reconstructed scores, which is the quantity a decomposition would actually inherit.

Finally it reads the feature counts from the saved attribution graph, since the contraction cost is
quadratic in active features per position and no estimate should stand in for the real count.
"""

from __future__ import annotations

import json

import torch
from transformer_lens import HookedTransformer

from qk_attribution.circuits import effective_rank, kv_head_for
from qk_attribution.scores import (
    attention_input,
    attention_scores,
    key_projection,
    qk_norm_scales,
    query_projection,
    rotation_matrices,
)

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The Eiffel Tower is located in the city of Paris, the capital of France."
SEED = 0
GRAPH = "spike_out/graph.pt"
RANKS = (4, 8, 16, 32, 64, 96, 128)
ENERGIES = (0.9, 0.99, 0.999)

torch.manual_seed(SEED)
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained_no_processing(MODEL, device="cuda", dtype=torch.float32)
cfg = model.cfg
d_head = cfg.d_head
print(
    f"model={MODEL} d_model={cfg.d_model} d_head={d_head} n_layers={cfg.n_layers} "
    f"n_heads={cfg.n_heads} seed={SEED}"
)

spectra = []
for layer in range(cfg.n_layers):
    for head in range(cfg.n_heads):
        row = {"layer": layer, "head": head}
        for side, projection in (("q", query_projection), ("k", key_projection)):
            values = torch.linalg.svdvals(projection(model, layer, head).float())
            row[side] = {f"r{e}": effective_rank(values, energy=e) for e in ENERGIES}
            row[f"{side}_condition"] = (values[0] / values[-1]).item()
        spectra.append(row)

for side in ("q", "k"):
    for energy in ENERGIES:
        ranks = torch.tensor([float(r[side][f"r{energy}"]) for r in spectra])
        print(
            f"{side} effective rank at {energy:<6}: median={ranks.median():.0f} "
            f"min={ranks.min():.0f} max={ranks.max():.0f} (of {d_head})"
        )

# Data-dependent truncation: does a rank-r projection still reproduce the scores?
tokens = model.to_tokens(PROMPT)
seq = tokens.shape[1]
_, cache = model.run_with_cache(tokens)
sample = [(0, 0), (0, 7), (13, 3), (13, 12), (27, 1), (27, 15)]
rotations = rotation_matrices(model, seq)
causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device="cuda"))


def truncate(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vh[:rank]


truncation = []
for layer, head in sample:
    residual = attention_input(model, cache, layer)
    scales = qk_norm_scales(model, cache, layer)
    assert scales is not None
    scale_q = scales[0][:, head]
    scale_k = scales[1][:, kv_head_for(model, head)]
    truth = cache[f"blocks.{layer}.attn.hook_attn_scores"][0, head]
    exact = attention_scores(
        model, layer, head, residual, query_scale=scale_q, key_scale=scale_k, rotations=rotations
    )
    base = ((exact[causal] - truth[causal]).norm() / truth[causal].norm()).item()
    left = query_projection(model, layer, head)
    right = key_projection(model, layer, head)
    errors = {}
    for rank in RANKS:
        query = (residual @ truncate(left, rank)) / scale_q
        key = (residual @ truncate(right, rank)) / scale_k
        query = torch.einsum("pa,pab->pb", query, rotations[:seq])
        key = torch.einsum("ja,jab->jb", key, rotations[:seq])
        got = query @ key.transpose(-1, -2) / float(cfg.attn_scale)
        errors[rank] = ((got[causal] - truth[causal]).norm() / truth[causal].norm()).item()
    truncation.append({"layer": layer, "head": head, "exact": base, "truncated": errors})
    shown = " ".join(f"r{r}={errors[r]:.3f}" for r in RANKS)
    print(f"layer {layer:2d} head {head:2d}: exact={base:.2e} | {shown}")

# Real feature counts from the saved graph, which set the contraction cost.
counts = None
try:
    graph = torch.load(GRAPH, map_location="cpu", weights_only=False)
    selected = graph.selected_features
    n_pos = len(graph.input_tokens)
    active = graph.active_features[selected] if hasattr(graph, "active_features") else None
    print(
        f"graph: {len(selected)} selected features over {n_pos} positions, "
        f"{len(selected) / n_pos:.1f} per position on average"
    )
    if active is not None:
        positions = active[:, 2] if active.ndim == 2 and active.shape[1] >= 3 else None
        if positions is not None:
            per_pos = torch.bincount(positions.cpu(), minlength=n_pos)
            print("features per position:", per_pos.tolist())
            counts = per_pos.tolist()
except Exception as exc:  # noqa: BLE001
    print("graph load failed:", type(exc).__name__, str(exc)[:200])

with open("measure_rank.json", "w") as fh:
    json.dump(
        {
            "model": MODEL,
            "seed": SEED,
            "d_head": d_head,
            "energies": list(ENERGIES),
            "ranks": list(RANKS),
            "spectra": spectra,
            "truncation": truncation,
            "features_per_position": counts,
        },
        fh,
    )
print("peak GPU MiB:", torch.cuda.max_memory_allocated() // 2**20)
print("wrote measure_rank.json")
