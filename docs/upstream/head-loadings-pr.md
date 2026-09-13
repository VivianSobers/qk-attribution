# Head loadings: which attention head carried an attribution edge

Relates to #53.

## What this adds

`circuit_tracer.attribution.head_loadings` splits one edge of an attribution graph across the
attention heads of a single layer. Given a source node, a target feature and a layer, it returns
one number per head plus a bypass term, and those parts sum to the edge effect.

```python
from circuit_tracer.attribution.head_loadings import FrozenRun, head_loadings

run = FrozenRun.from_model(model, graph.input_tokens)
loadings = head_loadings(model, graph, run, target_node=412, source_node=57, attention_layer=14)
heads, values = loadings.ranked()
print(f"head {heads[0].item()} carried {values[0].item():.3f} of {loadings.total.item():.3f}")
```

To see which layer carried an edge, `path_head_loadings` returns the split at every attention layer
on the path at once. Calling `head_loadings` once per layer repeats the propagation for each layer.
The map is linear, so one forward sweep and one backward sweep give every layer's split as dot
products, and the result matches the per-layer calls to within 6.2e-06.

## Why the split is exact

A graph is built with attention patterns and normalisation scales frozen, and with transcoder
activations frozen. An MLP's output therefore does not move when its input does, so every linear
path from a residual-stream perturbation to a later feature's pre-activation runs through attention.
At any attention layer the incoming residual splits into one part per head plus a part that bypasses
attention, each part propagates forward on its own, and the target reads their sum.

The split is per attention layer. Paths compose, so a signal can pass through heads at several
layers on the way and no single head deserves credit for the whole journey. Which heads at one
layer carried an edge has an exact answer. One joint attribution across every layer does not, and
this module does not offer one.

## Validation

I ran the branch file at this commit on Qwen3-0.6B and Qwen3-1.7B with the `mwhanna` low-L0
transcoders, over four prompts per model. Each prompt was attributed twice, once in bfloat16 and
once with `--dtype float32`. From each graph I took 60 forward feature-to-feature edges: the 30
strongest, and 30 drawn at random from those with |weight| above 1e-4. Every edge was checked
against the graph's own adjacency entry, which circuit-tracer computes by a backward pass, so it is
an independent reference.

| | 0.6B, bf16 graphs | 0.6B, float32 graphs | 1.7B, bf16 graphs | 1.7B, float32 graphs |
|---|---|---|---|---|
| edges | 240 | 240 | 240 | 240 |
| edge effect / adjacency, median | 0.999961 | 1.000000 | 0.999439 | 1.000000 |
| edge effect / adjacency, range | 0.180 to 1.156 | 0.999981 to 1.000157 | 0.561 to 1.571 | 0.999948 to 1.000026 |
| within 1% | 156 | 240 | 172 | 240 |
| strongest 120 within 10% | 120 | 120 | 120 | 120 |
| worst partition error, relative | 5.9e-05 | 1.2e-05 | 1.5e-05 | 1.4e-05 |
| worst single sweep vs per-layer, relative | 4.8e-06 | 5.2e-06 | 5.5e-06 | 6.2e-06 |

The partition row tests this code and nothing else. For every edge and every attention layer on its
path, the per-head parts plus the bypass are summed and compared with the edge effect, on paths up
to 27 layers long.

Against float32 graphs, all 480 edges agree with the adjacency matrix to within 1.6e-04. Against
bfloat16 graphs the spread is wide, and every edge that misses by more than 10% is a small one
drawn at random (|adjacency| at most 1.9e-02, where the strongest half has a median of 34 on 0.6B
and 144 on 1.7B). To confirm that the graph and not the loadings accounts for these misses, I
matched the four worst edges in the float32 graphs by (layer, position, feature). All four came to
1.0000 to 1.0007 there. The bfloat16 adjacency entries had overstated them by 1.32 to 5.56 times.
Strong edges in the bfloat16 graphs all held up; the small entries are the ones that can be off by
several times.

On a real model the single sweep costs about the same per edge at every path length (95 ms on 0.6B,
590 ms on 1.7B on an RTX 4090), and the per-layer loop repeats that once per layer. Over the 240
edges it took 22.9 s against 163.9 s on 0.6B and 142.1 s against 754.4 s on 1.7B. The gain grows
with path length, to about 16 times on paths of 11 or more layers. On single-layer paths there is
nothing to save.

The scripts, the raw per-edge JSON, the seed and the full configuration are in
[qk-attribution](https://github.com/VivianSobers/qk-attribution), experiments 015 and 017.

## Scope

The module refuses, with a message naming the reason, anything that breaks the argument above:

- cross-layer transcoders, since a feature writes into every later layer and a source node then has
  no single point of entry
- transcoders with a skip connection, since an MLP's output then moves when its input does
- layer norms with a bias, which does not act on a perturbation
- feature input hooks the readout does not model, meaning anything other than the MLP input
- error nodes as sources, which need the error vectors from the attribution run rather than
  anything the graph carries
- targets that are not features

Sources may be features or token embeddings. The backend is TransformerLens. Only the Qwen3
per-layer transcoders above were tested on real weights.

## Tests

`tests/test_head_loadings.py` adds 25 tests that run in under ten seconds with no weights and no
GPU, against a stub with random weights. The central ones check the partition property at every
valid attention layer and that the single sweep reproduces the per-layer split. The rest cover the
refusals above, the node-index arithmetic, and a regression for indexing `activation_values`. I
have not run the GPU part of the existing suite.

## A note on `activation_values`

The `Graph` docstring describes `activation_values` as "Activation values for selected features",
and both attribution backends pass `activation_matrix.values()`, which is one entry per *active*
feature. `create_graph_files.py` indexes it correctly as
`graph.activation_values[graph.selected_features[node_idx]]`. The two coincide whenever selection
prunes nothing, which is what makes the difference easy to miss. This PR follows the code rather
than the docstring; happy to fix the docstring here or separately.
