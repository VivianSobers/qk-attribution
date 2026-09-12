# 005: the decomposition is exact, and features carry a minority of the score

Date: 2026-09-12
Machine: worker-1, RTX 4090
Script: `experiments/scripts/measure_completeness.py`
Raw output: `experiments/results/005-completeness-qwen3-0.6b.json`
Config: Qwen3-0.6B, float32, seed 0, the 9-token attribution graph from experiment 001, query
position 8, transcoder set `mwhanna/qwen3-0.6b-transcoders-lowl0`.

## Question

The score reconstruction from experiment 003 works on the whole residual stream. Substituting a
per-source breakdown should expand it into source-pair terms. Do those terms sum back to the score
the model computed, and how much of it do transcoder features actually carry?

## The decomposition is exact

Relative error of the summed source-pair contributions against the model's own scores:

| Layer | Head | query sources | key sources | relative error |
|---|---|---|---|---|
| 7 | 3 | 38 | 1111 | 4.2e-07 |
| 14 | 5 | 101 | 1930 | 3.2e-07 |
| 20 | 3 | 167 | 2740 | 3.1e-07 |
| 27 | 9 | 221 | 3137 | 5.6e-07 |

That is float32 rounding. `tests/test_attribution_gpu.py` holds these four cases at a 1e-05
tolerance so a regression in the layernorm gain, the QK-norm scales, the rotary rotation or the
grouped-query mapping fails a test rather than producing plausible wrong numbers.

Exactness requires carrying the part of the residual that features do not explain. Per-layer
transcoders reconstruct MLP writes and nothing else, so earlier attention outputs, the token
embedding, transcoder errors and decoder biases are lumped into one direction per position by
`features.remainder_sources`. Without it the expansion loses most of the score silently.

## Features carry a minority of the score

The expansion has four blocks, depending on whether each side is a feature or the remainder. Each
entry below is the norm of that block's contribution across key positions, as a fraction of the
true score's norm. The blocks partly cancel, so they do not sum to one.

| Layer | Head | feature-feature | feature-remainder | remainder-feature | remainder-remainder |
|---|---|---|---|---|---|
| 7 | 3 | 0.168 | 0.392 | 0.603 | 0.811 |
| 14 | 5 | 0.123 | 0.211 | 0.373 | 0.481 |
| 20 | 3 | 0.079 | 0.219 | 0.293 | 0.780 |
| 27 | 9 | 0.210 | 0.281 | 0.111 | 0.569 |

The feature-feature block, which is the interpretable one and the one the method exists to produce,
carries between 8% and 21% of the score norm. The remainder-remainder block is the largest at every
head sampled.

## Why, and what it means

Two things drive this, and only one is a limitation of the method.

The first is architectural. Qwen3's residual stream is dominated by a small number of very large
directions: at layer 7 the residual norm is 6821 while that layer's MLP output norm is 41. Those
directions come from attention outputs and the embedding, not from MLP features, and they survive
into the query and key vectors. RMSNorm divides them out in part, but they still dominate what the
head sees.

The second is the transcoder set. Reconstructing each layer's MLP output from the graph's features
gives 40% to 70% relative error at layers 7, 14 and 20, and the low-L0 set used here has only 28
selected features at layer 27. A higher-fidelity transcoder set would move weight from the
remainder into the feature blocks, and this measurement should be repeated on one.

The deeper point is that per-layer transcoder features do not explain the residual stream, they
explain the MLP writes into it. Attributing the remainder would mean tracing it back through
earlier attention, which is what circuit-tracer's adjacency already does for the OV pathway. Until
that is wired in, the honest statement is that this gives an exact decomposition of the attention
score in which the interpretable part is a minority share, and the size of that share is now
measured rather than assumed.

## A bug this exposed

`scores.to_head_space` divided by the layernorm RMS scale but never applied `ln1`'s learned gain.
Every unit test passed because the stub's gain was all ones. The completeness check on the real
model failed at a relative error of about 1.0, which is what made it visible. The stub now carries
a random gain, so the tests constrain it.

This is the second bug of the same kind in this project: experiment 003 found that
`ln1.hook_normalized` fires before the gain. Both failures produce well-shaped, plausible numbers
and neither is caught by anything except a comparison against the model's own output.

## Next

- Repeat this on a higher-L0 transcoder set and see how much weight moves into the feature blocks
- Measure the rank of the feature-feature block, which is the object Anthropic's low-rank remark
  refers to and which experiments 004 did not reach
- Head loadings, which use the OV circuit and are unaffected by any of this
