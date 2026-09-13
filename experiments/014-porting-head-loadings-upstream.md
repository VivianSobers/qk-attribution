# 014: the upstream port reproduces circuit-tracer's own edge weights

Date: 2026-09-13
Machine: worker-2, RTX 4090
Script: `experiments/scripts/validate_upstream_port.py`
Raw output: `experiments/results/014-port-validation-1.7b.json`
Branch: `head-loadings` on `VivianSobers/circuit-tracer`, commits `13f9a46`, `182d102`, `40ccaa7`
Config: seed 0, Qwen3-1.7B in float32, the graph from experiment 009, 40 edges (the 20 strongest
feature-to-feature edges and 20 drawn at random from the 1.4 million forward ones), spanning 1 to
18 attention layers.

## Question

`qk_attribution.loadings` computes head loadings against a HookedTransformer and the project's own
helpers. Offering the work upstream means rewriting it against circuit-tracer's `Graph` and
`ReplacementModel`, which changes how source vectors, reader vectors and node indices are obtained.
A rewrite that silently disagrees with the original would be worse than no contribution, so the
port is checked against the same reference experiment 007 used: circuit-tracer's own adjacency
matrix, computed by a completely different route.

## Result

| | value |
|---|---|
| edge effect / adjacency, median | 0.9968 |
| standard deviation | 0.0435 |
| within 5% of unity | 35 of 40 |
| within 10% of unity | 39 of 40 |
| worst edge | 0.760 |
| partition error, absolute | 6.1e-05 |
| partition error, relative to the effect | 8.3e-06 |

The partition error is the quantity that tests the module rather than the graph: for every edge and
every attention layer along its path, the per-head parts plus the bypass were summed and compared
against the edge effect the same module computes. Agreement to 8e-06 in float32 is the claim the
module makes, and it holds over paths as long as 18 attention layers.

The spread against the adjacency matrix is the graph's own precision. Experiment 007 established
that on Qwen3-0.6B by re-running the comparison in three dtype combinations: the residual tracks
the graph's bfloat16 storage and not the loadings. The median here, 0.9968 against 1.0064 there,
is the same result on a different model.

## The one outlier

One edge of 40 came in at 0.760. It is a strong edge, adjacency 0.0133, which rules out the small
denominators that explain most of the tail. This is not resolved. The candidates are the graph's
bfloat16 accumulation over a path with many contributing heads, and a genuine difference between
what circuit-tracer's backward pass counts and what forward propagation through frozen attention
counts. Distinguishing them needs the same edge recomputed with a float32 graph, which is the
obvious next step and is not done here.

## What the port refuses

The rewrite states its assumptions as guards rather than leaving them in a docstring. It raises,
naming the reason, on cross-layer transcoders, on transcoders with a skip connection, on layer
norms with a bias, on feature input hooks away from the MLP input, on error nodes as sources, and
on targets that are not features.

The skip-connection case is the one worth spelling out. With a skip connection an MLP's output
moves when its input does, even with activations frozen, so the claim that every linear path runs
through attention fails and the partition would be silently incomplete. The Qwen3 transcoder sets
used throughout this project carry no `W_skip` tensor, which was checked by reading the safetensors
headers rather than assumed.

## Tests

19 tests in `tests/test_head_loadings.py` on the branch, running in under ten seconds against a
stub with random weights and no GPU. The central one asserts the partition at every valid attention
layer. One is a regression for `activation_values` being aligned with `active_features` rather than
with `selected_features`, the bug that cost this project a day on the 4B graph.

## Next

- Recompute the outlier edge from a float32 graph to separate graph precision from a real
  disagreement
- Open the pull request
