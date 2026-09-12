"""Measure low-rank structure where the claim actually applies: the score matrix, on real data.

Truncating W_Q and W_K by their own singular values ranks directions by weight magnitude, which is
not what the data puts through them. This measures the per-head score matrix itself on a longer
prompt, and the subspace the projected queries and keys actually occupy.
"""

from __future__ import annotations

import json

import torch
from transformer_lens import HookedTransformer

from qk_attribution.circuits import effective_rank, kv_head_for
from qk_attribution.scores import (
    attention_input,
    key_projection,
    qk_norm_scales,
    query_projection,
    rotation_matrices,
)

MODEL = "Qwen/Qwen3-0.6B"
SEED = 0
ENERGIES = (0.9, 0.99, 0.999)
RANKS = (4, 8, 16, 32, 64, 128)

TEXT = (
    "The Eiffel Tower is located in the city of Paris, the capital of France. It was designed by "
    "the engineer Gustave Eiffel and completed in 1889 for the World's Fair. At the time it was "
    "the tallest structure ever built, and it held that record until the Chrysler Building opened "
    "in New York in 1930. Visitors climb the stairs or take a lift to the observation decks. The "
    "tower is repainted every seven years to protect the iron from rust. During the Second World "
    "War the lift cables were cut so that the occupying forces would have to use the stairs. "
    "Today it is one of the most visited paid monuments in the world, and it appears on postcards, "
    "in films, and in countless photographs taken from the Champ de Mars below. Engineers still "
    "study its lattice construction, which distributes wind load through a curved profile that "
    "Eiffel derived from calculations rather than from any earlier building of comparable height."
)

torch.manual_seed(SEED)
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained_no_processing(MODEL, device="cuda", dtype=torch.float32)
cfg = model.cfg
tokens = model.to_tokens(TEXT)
seq = tokens.shape[1]
print(f"model={MODEL} seq={seq} d_head={cfg.d_head} seed={SEED}")

_, cache = model.run_with_cache(tokens)
rotations = rotation_matrices(model, seq)
causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device="cuda"))

rows = []
for layer in range(cfg.n_layers):
    residual = attention_input(model, cache, layer)
    scales = qk_norm_scales(model, cache, layer)
    assert scales is not None
    for head in range(cfg.n_heads):
        scale_q = scales[0][:, head]
        scale_k = scales[1][:, kv_head_for(model, head)]
        query = torch.einsum(
            "pa,pab->pb", (residual @ query_projection(model, layer, head)) / scale_q, rotations
        )
        key = torch.einsum(
            "ja,jab->jb", (residual @ key_projection(model, layer, head)) / scale_k, rotations
        )
        scores = query @ key.transpose(-1, -2) / float(cfg.attn_scale)

        masked = torch.where(causal, scores, torch.zeros_like(scores))
        score_values = torch.linalg.svdvals(masked)
        q_values = torch.linalg.svdvals(query)
        k_values = torch.linalg.svdvals(key)

        truncated = {}
        u, s, vh = torch.linalg.svd(masked, full_matrices=False)
        for rank in RANKS:
            approx = (u[:, :rank] * s[:rank]) @ vh[:rank]
            truncated[rank] = (
                (approx[causal] - masked[causal]).norm() / masked[causal].norm()
            ).item()

        rows.append(
            {
                "layer": layer,
                "head": head,
                "score_rank": {f"r{e}": effective_rank(score_values, energy=e) for e in ENERGIES},
                "query_rank": {f"r{e}": effective_rank(q_values, energy=e) for e in ENERGIES},
                "key_rank": {f"r{e}": effective_rank(k_values, energy=e) for e in ENERGIES},
                "truncated": truncated,
            }
        )
    del residual

for name in ("score_rank", "query_rank", "key_rank"):
    for energy in ENERGIES:
        values = torch.tensor([float(r[name][f"r{energy}"]) for r in rows])
        print(
            f"{name:11s} at {energy:<6}: median={values.median():.0f} min={values.min():.0f} "
            f"max={values.max():.0f}"
        )

for rank in RANKS:
    values = torch.tensor([r["truncated"][rank] for r in rows])
    print(
        f"score matrix truncated to rank {rank:3d}: relative error median={values.median():.4f} "
        f"p90={values.quantile(0.9):.4f} max={values.max():.4f}"
    )

with open("measure_rank2.json", "w") as fh:
    json.dump(
        {
            "model": MODEL,
            "seed": SEED,
            "seq": seq,
            "text": TEXT,
            "energies": list(ENERGIES),
            "ranks": list(RANKS),
            "rows": rows,
        },
        fh,
    )
print("peak GPU MiB:", torch.cuda.max_memory_allocated() // 2**20)
