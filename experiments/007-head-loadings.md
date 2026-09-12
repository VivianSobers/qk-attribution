# 007: head loadings reproduce circuit-tracer's edges, and one head usually dominates

Date: 2026-09-12
Machine: worker-1, RTX 4090
Script: `experiments/scripts/measure_edge_loadings.py`
Raw output: `experiments/results/007-edge-loadings-{bfloat16_float32,float32_float32,
bfloat16_bfloat16}.json`
Config: Qwen3-0.6B, seed 0, the 9-token graph from experiment 001, transcoder set
`mwhanna/qwen3-0.6b-transcoders-lowl0`. The 40 strongest feature-to-feature edges that both span a
layer and cross a position; a same-position edge travels down the residual stream and needs no head
at all.

## Question

An attribution graph says a feature at one position influenced a feature at another, but not which
head moved the signal. Under the constraints a graph is built with, that question has an exact
answer. Does computing it reproduce the edge weights the graph already holds, and is the answer
concentrated enough to be worth reading?

## The computation reproduces circuit-tracer's adjacency

Propagating a source feature's decoder direction forward through frozen attention and reading it
with the target feature's encoder gives, as a ratio against the graph's own adjacency entry:

| Transcoder dtype | Model dtype | Edges | median ratio | min | max | std |
|---|---|---|---|---|---|---|
| bfloat16 | float32 | 40 | 1.0064 | 0.938 | 1.040 | 0.022 |
| float32 | float32 | 8 | 0.9907 | 0.975 | 1.040 | 0.028 |
| bfloat16 | bfloat16 | 40 | 1.0000 | 0.955 | 1.047 | 0.019 |

This matters because the adjacency comes from a separate implementation: backward passes rather
than forward propagation, in a separate process, with no shared code beyond the model weights. The
agreement says the frozen-attention model implemented here is the one attribution graphs are
actually built under, and that no normalisation constant sits between them.

The few percent spread is not rounding in the transcoder rows. Loading them in float32 rather than
bfloat16 does not reduce it, and if anything widens it. It is the precision of the forward pass the
graph itself was built at: the saved graph records `dtype=torch.bfloat16`, and matching that dtype
moves the median to exactly 1.0000 while leaving the spread where it was. A float32 recomputation
cannot reproduce bfloat16 rounding accumulated over up to 28 layers, so a few percent is the floor
for this comparison rather than a modelling gap.

## The split is exact

Splitting the propagated effect across the heads of any single attention layer, plus a bypass for
what that layer's attention did not touch, reproduces the whole to 2.5e-07 in float32. The
bfloat16 run gives 9.2e-03, which is that dtype's arithmetic rather than a different answer.

The split is per layer. Paths compose, so a signal can pass through heads at several layers on the
way and there is no single head to credit for the edge as a whole. Each layer gives its own exact
partition of the same total, which is a well-posed question; a joint attribution across layers is
not, and the implementation does not offer one.

## One head usually carries the edge

For each edge, splitting at every layer between source and target and taking the layer where the
bypass is smallest, meaning the layer that actually moved the signal across positions:

| Quantity | median | min | max |
|---|---|---|---|
| largest head's share of head magnitude | 0.598 | 0.162 | 0.878 |
| bypass share at that layer | 0.418 | 0.094 | 0.703 |
| busiest layer's share of all head magnitude | 0.221 | | |

The carrier layer sits 3 layers above the source at the median, out of 13 layers searched. So
cross-position edges are moved soon after the source feature is written, usually by one head
carrying about 60% of the routed magnitude at that layer. That is concentrated enough to be worth
reading off a graph, which is what the feature is for.

The bypass at the carrier layer is still 42% at the median. That is not contradictory: the bypass
covers everything the layer's attention left in place, including signal that reached the target
position through some other layer's heads. Head magnitude spreads across layers, with the busiest
holding 22%.

## Note on scope

Nothing here depends on rotary embeddings, QK-norm or soft-capping. The OV pathway is a plain
product of the value and output projections for every model in scope, which is why head loadings
were the sound first milestone while QK attribution needed the architectural work in experiments
002 and 003.

## Next

- Repeat on a higher-L0 transcoder set, which experiments 005 and 006 also want
- Comment on circuit-tracer issue 53 with the head loadings result, since it is the half that is
  finished and architecture independent
