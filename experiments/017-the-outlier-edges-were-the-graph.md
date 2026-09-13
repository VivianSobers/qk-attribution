# 017: the worst edges were misstated by the bfloat16 graph, not the loadings

Date: 2026-09-13
Machines: worker-1 (Qwen3-0.6B) and worker-2 (Qwen3-1.7B), RTX 4090 each
Script: `experiments/scripts/explain_outlier_edge.py`
Raw output: `experiments/results/017-outlier-{0.6b-p0,0.6b-p1,0.6b-p2,1.7b-p0}.json`
Branch: `head-loadings` at `dd8ecbe`
Config: seed 0, model and transcoders in float32, the bfloat16 and float32 graphs of experiment
015. The 0.6B edges are the three worst ratios in 015's bfloat16 run. The 1.7B edge is the one at
0.760 that experiment 014 left unexplained.

## Question

Experiments 014 and 015 found edges whose forward effect was far from the graph's adjacency entry.
015 showed that the spread disappears as a distribution when graphs are attributed in float32. That
leaves open whether the same edges are right in float32 or whether different edges happened to be
sampled. This matches each edge in the float32 graph of the same prompt, by the (layer, position,
feature) of both ends, since node indices move when selection changes.

## Result

All four pairs were present in the float32 graphs.

| edge (target <- source) | bf16 adjacency | bf16 ratio | float32 adjacency | float32 ratio | bf16 / float32 adjacency |
|---|---|---|---|---|---|
| 0.6B "Dallas": (25, 7, 51296) <- (17, 1, 66863) | -3.60e-04 | 0.4046 | -1.53e-04 | 1.0007 | 2.36 |
| 0.6B "Eiffel": (9, 11, 161871) <- (3, 4, 80855) | -2.33e-04 | 0.4252 | -9.9e-05 | 1.0001 | 2.36 |
| 0.6B "Jordan": (22, 6, 155563) <- (2, 5, 89078) | +1.007e-03 | 0.1803 | +1.81e-04 | 1.0001 | 5.56 |
| 1.7B "Dallas": (19, 3, 114698) <- (4, 1, 69573) | +1.3306e-02 | 0.7600 | +1.0084e-02 | 1.0000 | 1.32 |

In float32, each edge's adjacency entry agrees with the forward effect to within 7e-04. The
bfloat16 entries overstate the same edges by factors of 1.32 to 5.56, always in magnitude and never
in sign. The head loadings were right each time, and 014's 0.760 edge has an explanation.

The forward effect itself moves a little between the two graphs, for example +0.010112 against
+0.010084 on the 1.7B edge. That movement is entirely the source feature's activation, which the
effect scales by and which the graph stores at its own precision: on every edge, the ratio of the
two effects equals the ratio of the two stored source activations to five digits. The bfloat16
activations are visibly quantised (2.046875, 0.96484375, 0.44921875, 5.1875).

## What this does and does not say about circuit-tracer

Four edges were examined, chosen for being the worst. Across 015, every miss beyond 10% had
|adjacency| of 1.9e-02 or less, and every one of the 240 strongest edges was within 10%. On this
evidence, reading strong edges from a bfloat16 graph holds up, while small entries, near the 1e-4
threshold used here, can be off by several times. Where the ordering of small edges matters, a
float32 graph is the safer input. Where in the backward pass bfloat16 loses the precision was not
investigated.
