"""Measure how wrong the plain bilinear QK form is, and verify the corrected one.

Two questions, one run, on Qwen3-0.6B:

  1. How large is the error if we treat W_Q @ W_K.T / sqrt(d_head) as the score operator?
  2. Does the derived form reproduce the model's own scores exactly?

     s(p, j) = x_p^T [W_Q diag(w_q) R(p - j) diag(w_k) W_K^T] x_j / (scale * s_q[p] * s_k[j])

     where x is the post-layernorm residual, w_q/w_k are the QK-norm gains, R is the rotary
     rotation, and s_q/s_k are the per-position RMS scales, frozen the way circuit-tracer already
     freezes layernorm scales.

Float32 throughout so the comparison measures the derivation and not bf16 rounding.
"""

from __future__ import annotations

import json

import torch
from transformer_lens import HookedTransformer

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The Eiffel Tower is located in the city of Paris, the capital of France."
SEED = 0

torch.manual_seed(SEED)
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained_no_processing(MODEL, device="cuda", dtype=torch.float32)
cfg = model.cfg
n_heads, n_kv, d_head = cfg.n_heads, cfg.n_key_value_heads, cfg.d_head
scale = float(cfg.attn_scale)
group = n_heads // n_kv

tokens = model.to_tokens(PROMPT)
seq = tokens.shape[1]
print(f"model={MODEL} seq={seq} n_layers={cfg.n_layers} n_heads={n_heads} n_kv={n_kv} "
      f"d_head={d_head} attn_scale={scale:.6f} seed={SEED}")

names = ("ln1.hook_normalized", "attn.hook_q", "attn.hook_k", "attn.hook_rot_q",
         "attn.hook_rot_k", "attn.hook_attn_scores", "attn.q_norm.hook_scale",
         "attn.k_norm.hook_scale")
_, cache = model.run_with_cache(
    tokens, names_filter=lambda n: any(n.endswith(s) for s in names)
)

# Rotation matrices by position, taken from the model's own apply_rotary so nothing is reimplemented.
attn0 = model.blocks[0].attn
basis = torch.eye(d_head, device="cuda").reshape(d_head, 1, 1, d_head).expand(d_head, seq, 1, d_head)
rot_basis = attn0.apply_rotary(basis.contiguous(), 0, None)  # [d_head, pos, 1, d_head]
R = rot_basis.squeeze(2).permute(1, 0, 2).contiguous()  # R[p] : v @ R[p] == apply_rotary(v, p)

# RoPE is relative: R[p] @ R[j].T should depend only on p - j. Check that rather than assume it.
rel = torch.einsum("pab,jcb->pjac", R, R)
offsets = torch.arange(seq, device="cuda")
delta = offsets[:, None] - offsets[None, :]
rel_dev = 0.0
for d in range(-(seq - 1), seq):
    sel = delta == d
    if sel.sum() < 2:
        continue
    block = rel[sel]
    rel_dev = max(rel_dev, (block - block[0]).abs().max().item())
print(f"rope_relative_max_deviation={rel_dev:.3e}")

causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device="cuda"))

def stats(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    g, w = got[causal], want[causal]
    return {
        "max_abs": (g - w).abs().max().item(),
        "rel_fro": ((g - w).norm() / w.norm()).item(),
        "corr": torch.corrcoef(torch.stack([g, w]))[0, 1].item(),
        "true_std": w.std().item(),
    }

rows = []
bias_norm = 0.0
for layer in range(cfg.n_layers):
    attn = model.blocks[layer].attn
    bias_norm = max(bias_norm, attn.b_Q.abs().max().item(), attn.b_K.abs().max().item())
    # TransformerLens fires ln1.hook_normalized *before* the learned gain, so the tensor the
    # attention projections actually see is hook_normalized * ln1.w.
    ln1 = model.blocks[layer].ln1
    assert not hasattr(ln1, "b"), "RMSNorm expected; a LayerNorm bias would need a term here"
    x = cache[f"blocks.{layer}.ln1.hook_normalized"][0] * ln1.w
    q_raw = cache[f"blocks.{layer}.attn.hook_q"][0]                  # [seq, n_heads, d_head]
    k_raw = cache[f"blocks.{layer}.attn.hook_k"][0]                  # [seq, n_kv, d_head]
    # QK-norm scales come back flattened over (batch, pos, head); restore the head axis.
    s_q = cache[f"blocks.{layer}.attn.q_norm.hook_scale"].reshape(seq, n_heads, 1)
    s_k = cache[f"blocks.{layer}.attn.k_norm.hook_scale"].reshape(seq, n_kv, 1)
    rot_q = cache[f"blocks.{layer}.attn.hook_rot_q"][0]
    rot_k = cache[f"blocks.{layer}.attn.hook_rot_k"][0]
    true = cache[f"blocks.{layer}.attn.hook_attn_scores"][0]         # [n_heads, seq, seq]

    # hook_k carries only kv heads, and W_K is the GQA-expanded view of _W_K.
    assert torch.equal(attn.W_K[0], attn._W_K[0])
    if layer == 0:
        proj_err = ((x @ attn.W_Q[0] - q_raw[:, 0]).norm() / q_raw[:, 0].norm()).item()
        print(f"projection check (x @ W_Q vs hook_q): rel={proj_err:.3e}")
        assert all(torch.equal(attn.W_K[h], attn._W_K[h // group]) for h in range(n_heads))
    w_q = attn.q_norm.w.float()
    w_k = attn.k_norm.w.float()

    for head in range(n_heads):
        kvh = head // group
        want = true[head]

        # (1) plain weight product, the form that ignores QK-norm and rotary entirely
        w_qk = attn.W_Q[head] @ attn.W_K[head].transpose(-1, -2) / scale
        plain = x @ w_qk @ x.transpose(-1, -2)

        # (2) QK-norm applied, rotary still ignored
        qn = q_raw[:, head] * w_q / s_q[:, head]
        kn = k_raw[:, kvh] * w_k / s_k[:, kvh]
        no_rope = qn @ kn.transpose(-1, -2) / scale

        # (3) rotary from the model's own hooks: the reference decomposition
        from_hooks = rot_q[:, head] @ rot_k[:, kvh].transpose(-1, -2) / scale

        # (4) the derived form, built from the residual stream and weights only
        wq_eff = attn.W_Q[head] * w_q                # d_model x d_head, gain folded in
        wk_eff = attn.W_K[head] * w_k
        left = x @ wq_eff                            # [seq, d_head]
        right = x @ wk_eff
        rq = torch.einsum("pa,pab->pb", left, R[:seq])
        rk = torch.einsum("jc,jcb->jb", right, R[:seq])
        derived = (rq @ rk.transpose(-1, -2)) / (scale * s_q[:, head] * s_k[:, kvh].transpose(0, 1))

        rows.append({
            "layer": layer, "head": head,
            "plain": stats(plain, want),
            "no_rope": stats(no_rope, want),
            "from_hooks": stats(from_hooks, want),
            "derived": stats(derived, want),
        })
    del x, q_raw, k_raw, rot_q, rot_k, true

print(f"max_abs_attn_bias={bias_norm:.3e}  (zero means the bilinear form needs no bias term)")

def summarise(key: str) -> None:
    rel = torch.tensor([r[key]["rel_fro"] for r in rows])
    mx = torch.tensor([r[key]["max_abs"] for r in rows])
    cr = torch.tensor([r[key]["corr"] for r in rows])
    print(f"{key:12s} rel_fro median={rel.median():.4g} mean={rel.mean():.4g} "
          f"min={rel.min():.4g} max={rel.max():.4g} | max_abs max={mx.max():.4g} "
          f"| corr median={cr.median():.4f} min={cr.min():.4f}")

print(f"--- {len(rows)} (layer, head) pairs, errors against hook_attn_scores on the causal mask ---")
for key in ("plain", "no_rope", "from_hooks", "derived"):
    summarise(key)

true_std = torch.tensor([r["plain"]["true_std"] for r in rows])
print(f"true score std: median={true_std.median():.4g} max={true_std.max():.4g}")

worst = max(rows, key=lambda r: r["derived"]["rel_fro"])
print("worst derived:", worst["layer"], worst["head"], worst["derived"])

with open("measure_qk.json", "w") as fh:
    json.dump({"model": MODEL, "prompt": PROMPT, "seed": SEED, "seq": seq,
               "attn_scale": scale, "rope_relative_max_deviation": rel_dev,
               "max_abs_attn_bias": bias_norm, "rows": rows}, fh)
print("peak GPU MiB:", torch.cuda.max_memory_allocated() // 2**20)
print("wrote measure_qk.json")
