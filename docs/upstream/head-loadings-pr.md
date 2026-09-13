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
products, and the result matches the per-layer calls to within 1e-05.

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

I ran the module file from this branch on real graphs and compared every edge's effect with the
graph's own adjacency entry, which circuit-tracer computes by a backward pass, so it is an
independent reference. From each graph I took 60 forward feature-to-feature edges: the 30 strongest,
and 30 drawn at random from those with |weight| above 1e-4.

| | Gemma-2-2B | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|---|
| transcoders | `mwhanna/gemma-scope-transcoders` | `mwhanna` low-L0 | `mwhanna` low-L0 |
| float32 graphs, edges | 60 (1 prompt) | 240 (4 prompts) | 240 (4 prompts) |
| edge effect / adjacency, median | 1.000000 | 1.000000 | 1.000000 |
| worst \|ratio - 1\| | 8.6e-05 | 1.6e-04 | 4.9e-05 |
| worst partition error, relative | 1.8e-05 | 9.8e-06 | 1.2e-05 |

The partition row compares the module with itself: for every edge and every attention layer on its
path, the per-head parts plus the bypass are summed and compared with the edge effect. The ratio row
is the check against circuit-tracer.

Gemma-2-2B caught two mistakes in the first version of this PR, which I had only tested on Qwen3.
Gemma-2 normalises attention's output (`ln1_post`) before the residual add, and the code left that
out. Its transcoders read `ln2.hook_normalized`, which TransformerLens takes before the norm's gain,
and the code applied the gain anyway. On the same 60 Gemma edges the first version had a median
ratio of 0.607, with 6 edges of the wrong sign. The last two commits fix both, and the tests now
check the edge effect against a separately written forward loop for each supported input hook, with
and without a post-attention norm. On the Qwen3 float32 graphs the fixed code gives the same 480 edge
effects as before to within 6.9e-06.

I also attributed the Qwen3 prompts in bfloat16. There the ratio spreads from 0.18 to 1.57, but
every one of the 240 strongest edges stays within 10%, and the misses are all small edges. I
matched four of the worst in the float32 graphs of the same prompts, and all four agree there, at
1.0000 to 1.0007. The bfloat16 adjacency entries had overstated them by 1.32 to 5.56 times. So the
spread comes from the bfloat16 graph, and small adjacency entries in bfloat16 graphs can be off by
several times.

On a real model the single sweep costs about the same per edge at every path length, and the
per-layer loop repeats that once per layer. Over 240 Qwen3 edges it took 22.9 s against 163.9 s on
0.6B and 142.1 s against 754.4 s on 1.7B, reaching about 16 times on paths of 11 or more layers. On
single-layer paths there is nothing to save.

The scripts, raw per-edge JSON, seeds and configuration are in
[qk-attribution](https://github.com/VivianSobers/qk-attribution), experiments 015, 017 and 018.

## Scope

The module refuses, with a message naming the reason, anything that breaks the argument above:

- cross-layer transcoders, since a feature writes into every later layer and a source node then has
  no single point of entry
- transcoders with a skip connection, since an MLP's output then moves when its input does
- norms with a bias, which does not act on a perturbation
- a model with a post-attention norm when the `FrozenRun` holds no scale for it
- feature input hooks other than `hook_mlp_in`, `ln2.hook_normalized` and `mlp.hook_in`, which are
  read as the residual before `ln2`, after its scale, and after its gain respectively
- error nodes as sources, which need the error vectors from the attribution run rather than
  anything the graph carries
- targets that are not features

Sources may be features or token embeddings. The backend is TransformerLens. On real weights, only
the three transcoder sets above were tested; `hook_mlp_in` is covered by the stub tests only.

## Tests and checks

`tests/test_head_loadings.py` adds 36 tests that need no weights and no GPU, run against a stub with
random weights. With `tests/test_graph.py` they pass in about 11 seconds, and `ruff check`,
`ruff format --check` and `pyright` are clean on the changed files. Besides the partition and the
sweep, they check the edge effect against an independent forward loop, and cover the refusals, the
node-index arithmetic, and a regression for indexing `activation_values`.

The three notebooks CONTRIBUTING.md lists (`circuit_tracing_tutorial`, `attribute_demo`,
`intervention_demo`) executed end to end with no error outputs on this branch before the two fix
commits, which change only `head_loadings.py` and its tests. I have not run the GPU part of the
existing test suite.

## A note on `activation_values`

The `Graph` docstring describes `activation_values` as "Activation values for selected features",
and both attribution backends pass `activation_matrix.values()`, which is one entry per *active*
feature. `create_graph_files.py` indexes it correctly as
`graph.activation_values[graph.selected_features[node_idx]]`. The two coincide whenever selection
prunes nothing, which is what makes the difference easy to miss. This PR follows the code rather
than the docstring; happy to fix the docstring here or separately.
