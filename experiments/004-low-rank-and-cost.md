# 004: how much low-rank structure is there, and what does the contraction cost

Date: 2026-09-12
Machine: worker-1, RTX 4090
Scripts: `experiments/scripts/measure_weight_rank.py`, `measure_score_rank.py`,
`graph_feature_counts.py`
Raw output: `experiments/results/004-weight-rank-qwen3-0.6b.json`,
`004-score-rank-qwen3-0.6b.json`
Config: Qwen3-0.6B, float32, seed 0. Weight ranks over all 448 (layer, head) pairs. Score ranks
over the same 448 pairs on a 192-token prompt. Feature counts from the graph saved in experiment
001.

## Question

Anthropic note that "many QK attribution matrices are approximately low-rank, which may permit a
shorter description", and that is the stated route to making feature-pair decomposition affordable
at realistic context lengths. Does that structure exist in Qwen3-0.6B, and how much does it buy?

## Weight-space truncation does not work

Effective rank of the projections `W_Q diag(w_q)` and `W_K diag(w_k)`, out of `d_head` = 128:

| Energy | query, median (min, max) | key, median (min, max) |
|---|---|---|
| 0.9 | 70 (32, 87) | 61 (2, 82) |
| 0.99 | 106 (76, 115) | 100 (34, 111) |
| 0.999 | 119 (103, 123) | 113 (80, 120) |

Truncating both projections by their own singular values and recomputing the scores, on six sampled
heads, gives relative errors between 0.11 and 0.68 at rank 96, and between 0.45 and 0.73 at rank
64. Rank 128 is exact, as it must be. So there is no usable low-rank structure in the weights
themselves.

This is the expected result in hindsight: singular values of `W_Q` rank directions by weight
magnitude, which says nothing about how much energy the residual stream actually puts through them.
The measurement is recorded because it is the obvious thing to try and it fails.

## Data-aware truncation buys about a factor of two

Effective rank of the per-head score matrix on a 192-token prompt, and of the projected queries and
keys, out of 128:

| Energy | score matrix | query | key |
|---|---|---|---|
| 0.9 | 7 (3, 36) | 26 (1, 51) | 12 (1, 31) |
| 0.99 | 60 (26, 132) | 74 (34, 96) | 55 (1, 81) |
| 0.999 | 144 (67, 178) | 106 (64, 118) | 93 (24, 113) |

Seven components carrying 90% of the squared energy looks like strong low-rank structure until it
is converted into error. Truncating the score matrix to its own optimal rank-r approximation:

| Rank | median error | p90 | max | heads under 5% error |
|---|---|---|---|---|
| 4 | 0.332 | 0.528 | 0.721 | 0 of 448 |
| 8 | 0.257 | 0.423 | 0.596 | 0 of 448 |
| 16 | 0.185 | 0.307 | 0.408 | 0 of 448 |
| 32 | 0.121 | 0.191 | 0.279 | 0 of 448 |
| 64 | 0.070 | 0.106 | 0.165 | 26 of 448 |
| 128 | 0.028 | 0.039 | 0.069 | 439 of 448 |

This is the best any rank-r approximation can do, since it is the SVD of the target itself. A real
implementation would do worse. Rank 64 halves the work for a 7% median error in the scores, which
is a factor of two rather than the order of magnitude that would make full decomposition tractable.

The spectrum does decay, so the low-rank description is not wrong. It is just that attribution needs
the tail: the leading components carry the gross attention structure, and the distinctions between
competing keys live below them.

## What the contraction actually costs

Active feature counts from the saved attribution graph, a 9-token prompt: 3165 features total, 352
per position on average, and 312 per position in layers below layer 21.

Cost for one head is the sum over causal position pairs of `k_q * d_head * k_k`. Taking 300
features per position and `d_head` = 128:

| Context | position pairs | per head | 16 heads |
|---|---|---|---|
| 9 | 45 | 0.5 GFLOP | 8 GFLOP |
| 128 | 8256 | 95 GFLOP | 1.5 TFLOP |
| 512 | 131328 | 1.5 PFLOP | 24 PFLOP |

The 9-token case is free. At 512 tokens a single head is about 19 seconds on one RTX 4090 at
realistic throughput, and a full layer is several minutes. Memory rules it out before compute does:
the feature-pair matrix for one position pair is 300 by 300, and there are 131328 pairs.

So full decomposition over all positions and heads is not the thing to build. The usable form takes
a query position and a head and decomposes that, which is also the question a user of an attribution
graph asks: this head attended from here to there, why.

## Caveat on what was measured

The score matrix is indexed by position. The QK attribution matrix Anthropic refer to is indexed by
feature pair, and it is a different object: it is the score matrix conjugated by the feature decoder
directions, restricted to features that are actually active. It could be far lower rank than the
score matrix, because the active features span a limited subspace and their activations are sparse.
That has not been measured here, because the feature decomposition does not exist yet. Nothing in
this note rules their observation out. It rules out the two cheaper proxies for it.

## Bug found

`circuits.effective_rank` raised on any CUDA input, because it built the comparison threshold with
`torch.tensor(energy)` on the CPU and passed it to `torch.searchsorted`. Every unit test ran on the
CPU, so nothing caught it. Now counted rather than searched, with a GPU-marked regression test.

## Next

- Extract active features and decoder directions from a circuit-tracer graph
- Measure the rank of the feature-space attribution matrix, the object the low-rank claim is about
- Build the decomposition for one query position and head, which is the form that fits in memory
