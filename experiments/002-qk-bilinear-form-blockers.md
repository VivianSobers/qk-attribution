# 002: which architectures admit a plain bilinear QK form

Date: 2026-09-12
Machine: worker-1, RTX 4090

## Question

QK attribution rests on attention scores being bilinear in the residual stream, so that inserting a
sparse decomposition at the query and key positions expands the score into feature-pair terms. Does
that identity actually hold for the models circuit-tracer supports?

## Finding: not for Qwen3, and probably not for most modern models

Qwen3-0.6B config, read from the loaded model:

| Feature | Value | Effect on the bilinear form |
|---|---|---|
| `positional_embedding_type` | `rotary` | breaks it as a single matrix |
| `rotary_dim` | 128 (equals `d_head`, so every dimension is rotated) | as above |
| `use_qk_norm` | `True`, with `q_norm` and `k_norm` modules present | breaks it |
| `attn_scores_soft_cap` | `-1.0`, meaning disabled | no effect here |
| `attn_scale` | 11.3137, equal to sqrt(128) | standard |
| `n_heads` / `n_key_value_heads` | 16 / 8 | GQA |

Rotary embeddings rotate the query and key by a position-dependent angle before the dot product.
Because the rotation is linear, bilinearity in the residual stream survives, but the operator
becomes position-offset dependent: `W_Q R(p_k - p_q) W_K.T` rather than one fixed matrix. RoPE's
relative property means it depends only on the offset, which is what keeps this tractable.

Query/key normalisation is the harder one. RMSNorm applied to the projected query and key is
nonlinear. It is a per-position, per-head scalar rescaling, so it can be absorbed as a frozen scalar
the way circuit-tracer already freezes layernorm scales, but it cannot be folded into a weight
product.

## Consequences for the implementation

The naive `W_Q @ W_K.T` operator is not the score operator for Qwen3. Returning it silently as
though it were would produce confidently wrong attributions, so `qk_matrix` now documents that it is
only the weight product, and `require_plain_qk` raises `UnsupportedArchitecture` listing every
offending feature. Measurement can still opt out with `check_architecture=False`.

The OV circuit is unaffected. Nothing is applied between the value and output projections, so
`W_V @ W_O` is exact for every architecture here. That makes head loadings the sound first milestone
and QK attribution the part needing architectural work.

## Second finding: HookedTransformer weight stacking causes OOM

`model.W_Q` on a HookedTransformer is a property that runs
`torch.stack([block.attn.W_Q for block in blocks])` on every access, materialising all 28 layers.
At float32 that is roughly 224 MiB per call. Helper functions that each read it turned one logical
operation into several hundred megabytes and exhausted a 24 GB card that was already holding the
transcoders.

Fixed by reading `model.blocks[layer].attn.W_Q` throughout, so only the requested layer is touched.
Anything iterating over layers and heads must avoid the stacked properties entirely.

## Third finding: cached key/value activations are not GQA-expanded

Weights are expanded, so `W_K` has one slice per query head. Activations are not: `hook_k` and
`hook_rot_k` come back with shape `(batch, seq, n_key_value_heads, d_head)`. Indexing those by query
head silently reads the wrong group. `kv_head_for` performs the mapping.

## Minor: benign teardown error

`TransformerLensReplacementModel.__del__` raises `TypeError: isinstance() arg 2 must be a type` via
`transformer_lens.hook_points.clear_context` during garbage collection. It fires after work
completes and does not affect results. Possible upstream report.

## Next

- Measure the size of the discrepancy between the plain form and true scores, for the record
- Check Gemma-2-2B, which has soft-capping but may lack QK-norm, and may therefore be the better
  target for a first QK implementation
- Derive the offset-dependent rotary operator and verify it reproduces true scores
