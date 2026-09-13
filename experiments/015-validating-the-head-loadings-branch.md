# 015: the head-loadings branch matches float32 graphs to within 1.6e-04 on two models

Date: 2026-09-13
Machines: worker-1 (Qwen3-0.6B) and worker-2 (Qwen3-1.7B), RTX 4090 each
Script: `experiments/scripts/validate_head_loadings_branch.py`
Raw output: `experiments/results/015-head-loadings-branch-{0.6b,1.7b}.json` and
`experiments/results/015-head-loadings-branch-{0.6b,1.7b}-float32-graphs.json`
Branch: `head-loadings` on `VivianSobers/circuit-tracer` at `dd8ecbe`, with
`circuit_tracer/attribution/head_loadings.py` copied to the workers and checked by md5
Config: seed 0, model and transcoders in float32, torch 2.14.0+cu130, transcoders
`mwhanna/qwen3-{0.6b,1.7b}-transcoders-lowl0`, 60 edges per graph (the 30 strongest forward
feature-to-feature edges with |weight| above 1e-4, plus 30 drawn at random from the rest of them).
Four prompts per model, each attributed twice: once in bfloat16, once in float32 with
`circuit-tracer attribute --dtype float32 --offload cpu --lazy-encoder`.

| prompt | tokens |
|---|---|
| The capital of the state containing Dallas is | 9 |
| The Eiffel Tower is located in the city of | 12 |
| Michael Jordan played the sport of | 7 |
| The currency used in Japan is called the | 9 |

## Question

Experiment 014 checked an early version of the port on one graph and 40 edges. Since then the
branch has gained tighter guards and `path_head_loadings`, which splits an edge at every attention
layer in one forward and one backward sweep. A pull request should be backed by numbers from the
code it contains, so this runs the branch file itself.

Each edge gets three checks. The first compares the edge effect with the graph's adjacency entry,
which circuit-tracer computed by a backward pass and which is therefore an independent reference.
The second is the partition: at every attention layer on the path, the per-head parts plus the
bypass must add up to the edge effect. The third compares the single sweep with the per-layer loop,
both for agreement and for wall-clock time on a real model.

## Result

| | 0.6B, bf16 graphs | 0.6B, float32 graphs | 1.7B, bf16 graphs | 1.7B, float32 graphs |
|---|---|---|---|---|
| edges | 240 | 240 | 240 | 240 |
| path length, attention layers | 1 to 24 | 1 to 27 | 1 to 22 | 1 to 23 |
| effect / adjacency, median | 0.999961 | 1.000000 | 0.999439 | 1.000000 |
| effect / adjacency, range | 0.180 to 1.156 | 0.999981 to 1.000157 | 0.561 to 1.571 | 0.999948 to 1.000026 |
| within 0.01% | 3 | 239 | 6 | 240 |
| within 1% | 156 | 240 | 172 | 240 |
| within 10% | 231 | 240 | 232 | 240 |
| strongest half within 10% | 120 of 120 | 120 of 120 | 120 of 120 | 120 of 120 |
| worst partition error, relative | 5.9e-05 | 1.2e-05 | 1.5e-05 | 1.4e-05 |
| worst sweep against loop, relative | 4.8e-06 | 5.2e-06 | 5.5e-06 | 6.2e-06 |

Against float32 graphs, all 480 edges agree with the adjacency matrix to within 1.6e-04 of the
edge. The partition and the sweep hold to about 1e-05 whichever graph dtype is used, which is
consistent with float32 rounding over paths up to 27 layers long. These two rows test this code. The agreement
rows test the graph just as much.

## Where bfloat16 graphs disagree

Against bfloat16 graphs the ratio spreads widely, and the size of the spread depends on the size of
the edge. Every edge that misses by 10% or more comes from the random half: 9 on 0.6B, with
|adjacency| no larger than 5.7e-03, and 8 on 1.7B, no larger than 1.9e-02. The median |adjacency|
in the strongest half is 34 and 144 respectively. The Spearman correlation between |adjacency| and
|ratio - 1| is -0.56 on 0.6B and -0.53 on 1.7B.

Two things could explain the spread. Either bfloat16 misstates small adjacency entries, or the
branch's forward propagation counts something the backward pass does not. The float32 graphs are
the test between them. Their spread falls from 0.18..1.57 to 0.99995..1.00016, and because the
branch code and the float32 model are identical across both columns, the bfloat16 graph is what
changed. Experiment 017 follows the four worst edges individually.

Regenerating a graph changes which features selection keeps, so the float32 columns sample
different edges from the bfloat16 columns. These two comparisons are between distributions of
edges. Only 017 matches individual edges.

## Timing

| | 0.6B | 1.7B |
|---|---|---|
| total, per-layer loop | 163.9 s | 754.4 s |
| total, single sweep | 22.9 s | 142.1 s |
| median speedup per edge | 4.0x | 2.9x |
| median speedup, paths of 1 to 3 layers | 2.0x | 1.0x |
| median speedup, paths of 11 or more layers | 17.2x | 15.8x |

These are the bfloat16-graph runs; the float32-graph runs give 167.6 s against 23.0 s and 750.9 s
against 141.8 s. On a real model the sweep takes about the same time per edge at every path length,
roughly 95 ms on 0.6B and 590 ms on 1.7B. The loop repeats that cost once per layer. The measured
gain therefore tracks path length and not the `L**2` step count the stub benchmark was built
around. On a single-layer path there is nothing to save, and the sweep came out slower (down to
0.92x) on 20 such edges on 0.6B and 62 on 1.7B.

## Limits

- Only the Qwen3 per-layer transcoders with low L0 were tested, on the TransformerLens backend.
  Cross-layer transcoders and skip connections are refused by the module and were not exercised.
- The prompts are short, 7 to 12 tokens.
- Timing comes from one run per edge with no warm-up. Worker-1 was also running the Qwen3-8B
  attribution job during its runs, so the machines were not controlled for timing.
- circuit-tracer's GPU test suite has not been run on the branch. Its CPU tests have.
