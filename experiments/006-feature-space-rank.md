# 006: the feature-pair matrix has modest structure, worth about a factor of two

Date: 2026-09-12
Machine: worker-1, RTX 4090
Script: `experiments/scripts/measure_feature_rank.py`
Raw output: `experiments/results/006-feature-rank-qwen3-0.6b.json`
Config: Qwen3-0.6B, float32, seed 0, the 9-token graph from experiment 001, query position 8,
transcoder set `mwhanna/qwen3-0.6b-transcoders-lowl0`. 64 blocks, being 16 heads at each of layers
7, 14, 20 and 27.

## Question

Anthropic note that "many QK attribution matrices are approximately low-rank, which may permit a
shorter description". Experiment 004 tested two proxies for that and both failed: no usable
structure in the weight projections, and about a factor of two in the score matrix. This measures
the object the remark is actually about, the feature-by-feature contribution block.

## A structural bound comes free

The block factors through head space: it is a product of an `(n_query_features, d_head)` matrix and
a `(d_head, n_key_features)` one. Its rank is therefore at most `d_head`, which is 128 here,
however many features are active. A 38 by 1111 block with rank at most 128 is already a compact
description, and it is the factorisation the implementation uses. So part of the low-rank
observation is a property of attention rather than of the features.

## Effective rank sits well below the bound

Effective rank of the feature-feature block, against a median bound of 101:

| Energy | median | min | max |
|---|---|---|---|
| 0.9 | 13 | 5 | 21 |
| 0.99 | 37 | 20 | 48 |
| 0.999 | 63 | 29 | 80 |

Converted to error, using the block's own optimal rank-r approximation, which is the best any
method can do:

| Rank | median error | p90 | max |
|---|---|---|---|
| 1 | 0.824 | 0.871 | 0.899 |
| 4 | 0.606 | 0.660 | 0.692 |
| 8 | 0.438 | 0.509 | 0.557 |
| 16 | 0.263 | 0.342 | 0.380 |
| 32 | 0.123 | 0.169 | 0.194 |
| 64 | 0.033 | 0.045 | 0.056 |

Rank 64 of a 101 bound gives 3.3% median error. That is better than the score matrix managed in
experiment 004, where rank 64 of 128 left 7%, but it is the same order: a factor of two, reached at
a few percent error.

## Pruning is lossy in the way that matters

Fraction of the block's total absolute magnitude held by the largest pairs:

| Top | median | min | max |
|---|---|---|---|
| 0.1% of pairs | 0.100 | 0.044 | 0.208 |
| 1% of pairs | 0.304 | 0.220 | 0.528 |
| 10% of pairs | 0.748 | 0.701 | 0.899 |

Keeping the top 1% of pairs retains under a third of the magnitude, and keeping the top 10% loses a
quarter. There is real concentration, since 10% of pairs carrying 75% of the mass is far from
uniform, but nothing like the heavy tail that would let most pairs be discarded safely. This
matches Anthropic's own position that pruning is likely needed and is not free.

## What the three measurements together say

| Object | best case | error there |
|---|---|---|
| weight projections (004) | rank 96 of 128 | 0.11 to 0.68 |
| score matrix (004) | rank 64 of 128 | 0.070 median |
| feature-pair block (006) | rank 64 of 101 | 0.033 median |

The structure is real and it improves as the object gets closer to what the method produces, but on
this model it is worth a factor of two rather than the order of magnitude that would make
decomposition over all position pairs tractable. The cost arithmetic in experiment 004 stands, and
so does the decision to scope the implementation to one query position and head.

One caveat carried from experiment 005: this uses a low-L0 transcoder set whose features account
for 8% to 21% of the score norm. A higher-fidelity set would change the block being measured, and
this should be repeated on one before the conclusion is treated as settled.

## Next

- Head loadings, which need only the OV circuit and are exact for every architecture
- Repeat experiments 005 and 006 on a higher-L0 transcoder set
