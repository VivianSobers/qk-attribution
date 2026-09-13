# 016: the QK attribution port reproduces Qwen3's attention scores to 6e-07

Date: 2026-09-13
Machines: worker-1 (Qwen3-0.6B) and worker-2 (Qwen3-1.7B), RTX 4090 each
Script: `experiments/scripts/validate_qk_attribution_branch.py`
Raw output: `experiments/results/016-qk-attribution-branch-{0.6b,1.7b}.json`
Branch: `main` on `VivianSobers/circuit-tracer` at `46cbddd`, with
`circuit_tracer/attribution/qk_attribution.py` copied to the workers
Config: seed 0, model and transcoders in float32, the bfloat16 graph of "The capital of the state
containing Dallas is" (9 tokens) for each model, from experiment 015. All 448 heads for
reconstruction; 2 heads per layer drawn with the seed, layers 1 to 27, at the last query position,
for the decomposition.

## Question

The port of QK attribution to circuit-tracer was tested against a CPU stub with an independent
reference forward pass. The stub implements rotary embeddings and QK-norm the way the port assumes
a model does, so an assumption shared by both would pass unnoticed. Experiment 003 caught exactly
that kind of error in the original implementation, by comparing against the model's own
`hook_attn_scores`. This repeats that check for the port.

The decomposition check asks whether the source-pair contributions, summed by key position,
reproduce the same row of scores. Sources are the graph's active features on each side, plus a
remainder per position that carries everything the features do not.

## Result

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|
| heads reconstructed | 448 | 448 |
| reconstruction error, median | 1.98e-07 | 1.40e-07 |
| reconstruction error, worst | 5.86e-07 | 4.74e-07 |
| heads decomposed | 54 | 54 |
| decomposition error, median | 2.48e-07 | 4.16e-07 |
| decomposition error, worst | 6.37e-07 | 1.10e-06 |
| feature-pair share, median | 0.388 | 0.481 |
| feature-pair share, range | 0.053 to 0.601 | 0.069 to 0.699 |

Errors are the norm of the difference over the norm of the reference, over the causal part of each
score matrix for reconstruction and over one row for the decomposition. Both sit at float32
precision, and the reconstruction median is in line with the 2.5e-07 experiment 003 reported for the
original implementation.

The feature-pair share is the summed |contribution| of pairs in which both the query source and the
key source are features, divided by the summed |contribution| of all pairs. It measures something
different from experiment 005's share of the score norm, so the two numbers should not be compared.

## Limits

- One prompt per model, and the decomposition covers one query position on 54 of 448 heads.
- The branch is not part of the head-loadings pull request. It stays on the fork's `main` until that
  one is reviewed.
