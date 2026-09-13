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
on the path at once. Calling `head_loadings` per layer re-propagates each head's part to the target,
which is on the order of `L**2 * n_heads` attention steps. The map is linear, so one forward sweep
and one backward sweep give every layer's split as dot products. It matches the per-layer calls to
floating-point precision, and on a 28-layer stub it ran 16 times faster.

## Why the split is exact

A graph is built with attention patterns and normalisation scales frozen, and with transcoder
activations frozen. An MLP's output therefore does not move when its input does, so every linear
path from a residual-stream perturbation to a later feature's pre-activation runs through attention.
At any attention layer the incoming residual splits into one part per head plus a part that bypasses
attention, each part propagates forward on its own, and the target reads their sum.

The split is per attention layer. Paths compose, so a signal can pass through heads at several
layers on the way and no single head deserves credit for the whole journey. Asking which heads at
one layer carried an edge has an exact answer; asking for one joint attribution across every layer
does not, and this module does not offer one.

## Validation

Checked on Qwen3-1.7B with `mwhanna/qwen3-1.7b-transcoders-lowl0`, over 40 edges spanning 1 to 18
attention layers: the 20 strongest feature-to-feature edges of the graph plus 20 drawn at random
from the 1.4 million forward ones.

| | value |
|---|---|
| edge effect / adjacency, median | 0.9968 |
| within 5% of unity | 35 of 40 |
| within 10% of unity | 39 of 40 |
| partition error, relative to the effect | 8.3e-06 |

The partition error is what tests this code rather than the graph: for every edge and every
attention layer along its path, the per-head parts plus the bypass are summed and compared against
the edge effect. The spread against the adjacency matrix is the graph's own bfloat16 precision;
re-running the same comparison with the graph in three dtype combinations moves the residual with
the graph's storage dtype and not with the loadings.

One edge of 40 came in at 0.760, and I cannot account for it. It is a strong edge, so the small
denominators that explain most of the tail do not apply here. The candidates I have left are
bfloat16 accumulation over a path with many contributing heads, and a real difference between what
the backward pass counts and what forward propagation through frozen attention counts. Separating
those needs the same edge recomputed from a float32 graph, which I have not done.

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

Sources may be features or token embeddings. The backend is TransformerLens.

## Tests

`tests/test_head_loadings.py` adds 25 tests that run in under ten seconds with no weights and no
GPU, against a stub with random weights. The central ones check the partition property at every
valid attention layer and that the single sweep reproduces the per-layer split. The rest cover the
refusals above, the node-index arithmetic, and a regression for indexing `activation_values`.

## A note on `activation_values`

The `Graph` docstring describes `activation_values` as "Activation values for selected features",
and both attribution backends pass `activation_matrix.values()`, which is one entry per *active*
feature. `create_graph_files.py` indexes it correctly as
`graph.activation_values[graph.selected_features[node_idx]]`. The two coincide whenever selection
prunes nothing, which is what makes the difference easy to miss. This PR follows the code rather
than the docstring; happy to fix the docstring here or separately.
