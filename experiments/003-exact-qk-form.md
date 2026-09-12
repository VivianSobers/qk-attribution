# 003: the exact bilinear QK form, verified against Qwen3

Date: 2026-09-12
Machine: worker-1, RTX 4090
Script: `experiments/scripts/measure_qk_form.py`
Raw output: `experiments/results/003-qk-form-qwen3-0.6b.json`
Config: Qwen3-0.6B, float32, seed 0, 18-token prompt, all 28 layers and 16 heads, 448 pairs.

## Question

Experiment 002 found that rotary embeddings and QK-norm both break `W_Q @ W_K.T` as the score
operator for Qwen3, which is the operator QK attribution needs. Two things were left open: how
wrong the plain form actually is, and whether a corrected form can be written down and shown to
reproduce the model's own scores.

## The form

Attention scores stay bilinear in the residual stream. For one head:

    s(p, j) = x_p @ [ W_Q diag(w_q) R(p) R(j).T diag(w_k) W_K.T ] @ x_j
              / ( attn_scale * sigma_q[p] * sigma_k[j] )

`x` is the residual entering attention. `w_q` and `w_k` are the QK-norm gains, which fold into the
projections because they scale each channel by a fixed amount. `R(p)` is the rotary rotation at
position `p`. `sigma_q[p]` and `sigma_k[j]` are the QK-norm RMS scales, which are per-position
scalars and can be frozen the way circuit-tracer already freezes layernorm scales.

Everything outside the bracket is a scalar, so the bracket is what a feature-pair decomposition
attaches to. A query-side and a key-side feature vector each project into head space once, and
their interaction is then a `d_head x d_head` rotation apart.

## Results

Relative Frobenius error against `hook_attn_scores`, restricted to positions the causal mask keeps.
For reference the true scores have a median standard deviation of 2.30.

| Form | median | p90 | max | median correlation |
|---|---|---|---|---|
| plain `W_Q @ W_K.T` | 1.175 | 43.7 | 218.1 | 0.109 |
| QK-norm applied, rotary ignored | 0.0986 | 0.569 | 1.401 | 0.963 |
| from `hook_rot_q` and `hook_rot_k` | 7.0e-08 | 9.0e-08 | 1.4e-07 | 1.0000 |
| the form above, from weights and residual | 2.5e-07 | 3.4e-07 | 5.1e-07 | 1.0000 |

The plain form has a median relative error slightly above 1, meaning it is on average further from
the true scores than predicting zero everywhere would be. Its correlation with the truth falls
below 0.5 for 395 of the 448 heads, and it is negative for some. Anything built on it would rank
feature pairs by something other than the model's behaviour.

The derived form reproduces the model to 2.5e-07 median and 5.1e-07 worst case, with a largest
absolute deviation of 6.0e-06 on scores whose spread is around 2. That is float32 rounding. The row
above it confirms the reference path itself is sound, so the two agreeing is a check on the
rearrangement rather than on a shared assumption.

Two supporting measurements. `R(p) @ R(j).T` depends on `p - j` to within 8.0e-07, so the operator
can be cached by offset rather than by position pair, which is what keeps the cost linear in
context length instead of quadratic. Qwen3's attention projection biases are exactly zero, so the
score is bilinear rather than affine and no constant term is needed.

## The bug this run exposed

The first attempt reproduced nothing, at 0.998 median relative error, which is worse than ignoring
rotary altogether. The cause was `ln1.hook_normalized`. TransformerLens fires that hook after
dividing by the RMS scale but before applying the learned gain, so the tensor the attention
projections actually see is `hook_normalized * ln1.w`. Checking `x @ W_Q` against `hook_q` localised
it in one step: the projection was off by a factor of 3.7 in relative norm before anything
position-dependent entered.

The same applies to `q_norm.hook_normalized` and `k_norm.hook_normalized`, which are also pre-gain.
`qk_attribution.scores.attention_input` now does this, and a unit test covers it, because the
failure is silent: the result stays a plausible-looking matrix of scores.

## Architecture survey

Configs read through `get_pretrained_model_config`, no weights downloaded.

| Model | rotary | QK-norm | score soft cap | GQA | attention window |
|---|---|---|---|---|---|
| Qwen3-0.6B | yes, dim 128 | yes | no | 16/8 | none |
| Gemma-2-2B | yes, dim 256 | no | 50.0 | 8/4 | 4096, alternating local and global |
| Gemma-3-1B-IT | yes, dim 256 | yes | no | 4/1 | 512, local with every sixth layer global |
| Llama-3.2-1B | yes, dim 64 | no | no | 32/8 | none |

Experiment 002 suggested Gemma-2-2B as a possible easier first target on the grounds that it might
lack QK-norm. It does lack it, but it soft-caps scores at 50.0, which is a `tanh` applied to the
score itself and therefore sits outside the bilinear form rather than rescaling it. Since Qwen3 now
works end to end, there is no reason to switch. Llama-3.2-1B is the cleanest architecture of the
four, with rotary and nothing else, and is worth keeping as a second model once the decomposition
exists.

## What this settles

QK attribution is implementable on the model circuit-tracer already supports best. The remaining
work is the decomposition itself rather than the algebra it rests on.

`scores.require_supported` refuses soft-capped models rather than returning numbers that look
reasonable, so Gemma-2 fails loudly until the cap is handled.

## Next

- Decompose the query and key sides into transcoder features and measure the cost of the pairwise
  contraction at realistic context lengths
- Measure the rank of the per-head operator on real weights, since Anthropic report these matrices
  are approximately low rank and that is the route to making the contraction affordable
- Head loadings, which need only the OV circuit and are unaffected by any of this
