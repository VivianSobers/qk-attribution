# QK Attribution

Explaining *why* language models attend where they do, by decomposing attention scores into
interactions between query-side and key-side features.

## The problem

Attribution graphs, as implemented in
[circuit-tracer](https://github.com/decoderesearch/circuit-tracer), freeze attention patterns and
treat them as constants. A graph therefore shows where information flows through attention via the
OV pathway, but says nothing about why the model attended to those positions. The QK pathway is
invisible.

Anthropic described a method that closes this gap in
[Tracing Attention Computation Through Feature Interactions](https://transformer-circuits.pub/2025/attention-qk/index.html),
calling QK attributions "a significant qualitative improvement on the original attribution graphs".
[Issue #53](https://github.com/decoderesearch/circuit-tracer/issues/53) requested it upstream and it
remains unimplemented there.

## What this does

Two things, with different maturity.

**Head loadings** split an existing graph edge across the attention heads that carried it. This is
finished and independent of architecture. Propagating a source feature's decoder direction forward
through frozen attention and reading it with the target's encoder reproduces the adjacency entries
circuit-tracer computes by an entirely different route, with no normalisation constant between them
(median ratio 1.0064 over 40 edges; exactly 1.0000 when dtypes match). Splitting that across one
layer's heads is exact to 2.5e-07. Ported to circuit-tracer and checked there on 960 edges from
Qwen3-0.6B and 1.7B graphs, every edge from a float32 graph agrees with the adjacency matrix to
within 1.6e-04. The port is open upstream as
[decoderesearch/circuit-tracer#114](https://github.com/decoderesearch/circuit-tracer/pull/114).

**QK attribution** decomposes an attention score into query-side by key-side feature terms. The
decomposition is exact, and its numbers predict interventions exactly: delete a feature from the
residual stream and the score moves by the attributed amount, correlation 1.000000 across all 263
heads tested on two models and four prompts. The features it names fire on the token they point at
in 79.4% of 412 head cases against a 10.2% control. Ablating the top-ranked features moves the
attention distribution two to ten times further than removing the same number from the same
positions, in about 90% of cases.

Read the ranking rather than the single top pair: removing the top feature shifts how a head
distributes attention, but it changes attention to the one position it names more than a
same-position competitor only 61% of the time.

Two caveats sit on top of that. Feature terms account for 8% to 21% of the score with the low-L0
transcoders published for Qwen3-0.6B. With the higher-L0 sets they reach 63% on one sampled
Qwen3-4B head and 86% on one Qwen3-8B head, though the explanations on those models did not read
any better. And the exactness holds only while the normalisation scales are frozen, as attribution
graphs freeze them: let RMSNorm respond and the same prediction correlates at 0.17 to 0.54 instead
of 1.0.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="experiments/figures/011-interventions-dark.png">
  <img alt="Predicted against measured change in attention score when the top attributed feature is removed, for 263 heads on Qwen3-0.6B and Qwen3-1.7B. With normalisation frozen every point lies on the diagonal; with normalisation free to respond the points scatter widely around it." src="experiments/figures/011-interventions-light.png">
</picture>

Every head from the intervention runs, four prompts per model. The left panel is the setting
attribution graphs use; the right shows how much of that exactness comes from freezing the norms.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="experiments/figures/010-label-match-dark.png">
  <img alt="Share of heads whose strongest key-side feature fires on the token at the position it points to, against a random other position: 146 of 171 against 22 of 171 on Qwen3-0.6B, 111 of 153 against 11 of 153 on Qwen3-1.7B, and 70 of 88 against 9 of 88 on Qwen3-4B." src="experiments/figures/010-label-match-light.png">
</picture>

Whether the explanation's top key feature fires on the token it points to, against a random other
position in the same prompt. Counts are printed on each bar.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="experiments/figures/009-models-dark.png">
  <img alt="Two panels over four Qwen3 models. Left: feature-pair share of the attention score at four sampled depths, staying under a third for the low-L0 models and peaking at 0.63 on 4B and 0.86 on 8B at three fifths of the depth. Right: median error of the feature-pair block truncated to rank 1 through 64, falling similarly for all four models, with the high-L0 models slightly harder to compress." src="experiments/figures/009-models-light.png">
</picture>

The four-model comparison. Each point on the left is one head on one prompt, so single points should
not be leaned on; the right panel is a median over 64 to 128 blocks per model.

## The form that actually holds

`W_Q @ W_K.T` is not the score operator for a modern model. On Qwen3-0.6B its median relative error
against the model's own scores is 1.175, worse than predicting zero, because rotary embeddings and
QK-normalisation both intervene. What does hold is

    s(p, j) = x_p @ [W_Q diag(w_q) R(p) R(j).T diag(w_k) W_K.T] @ x_j
              / (attn_scale * sigma_q[p] * sigma_k[j])

The QK-norm gains fold into the projections, the rotary operator depends on the position offset
rather than being one matrix, and the RMS scales are per-position scalars frozen the way
circuit-tracer already freezes layernorm scales. This reproduces the model to 2.5e-07 median across
all 448 (layer, head) pairs. Gemma-2's attention-score soft-capping does not reduce this way and is
refused rather than approximated.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="experiments/figures/003-score-form-dark.png">
  <img alt="Cumulative share of Qwen3-0.6B's 448 heads against relative error in reproducing the model's attention scores. The corrected form sits near 2.5e-07 on every head, leaving rotary embeddings out gives a median of 0.099, and the plain W_Q W_K transpose product has a median of 1.2, reaching above 100 on some heads." src="experiments/figures/003-score-form-light.png">
</picture>

Each curve is the share of heads reproduced to within a given error, per candidate form.

## What has been measured

| Question | Answer | Where |
|---|---|---|
| Is the plain QK form usable? | No. Median error 1.175; correlation under 0.5 for 395 of 448 heads | `experiments/003` |
| Does the corrected form hold? | Yes, to 2.5e-07 median | `experiments/003` |
| Is the decomposition exhaustive? | Yes, to 3e-07 with the remainder carried | `experiments/005` |
| How much do features explain? | 8% to 21% of the score norm | `experiments/005` |
| Is there low-rank structure to exploit? | About a factor of two, not an order of magnitude | `experiments/004`, `006` |
| Do head loadings match upstream? | Yes, median ratio 1.0064 | `experiments/007` |
| Does the output read as anything? | Key side yes, query side not with low-L0 transcoders | `experiments/008` |
| Does it hold on other models? | Yes, on four; feature share reaches 86% at one 8B high-L0 head | `experiments/009` |
| Do explanations point at the right token? | 79.4% of 412 head cases, against a 10.2% control | `experiments/010` |
| Do the numbers predict interventions? | Exactly: 263/263 heads, correlation 1.000000 | `experiments/011` |
| Is one feature the cause of one edge? | Only 61% of the time; the ranking is the useful part | `experiments/011` |
| Does the ranking predict the pattern? | Yes: 2 to 10 times a matched control, in ~90% of cases | `experiments/012` |
| How long is an explanation? | 25 features for half the movement, 197 for 90% | `experiments/013` |
| Does the upstream port agree? | Yes: median ratio 0.9968, partition error 8.3e-06 | `experiments/014` |
| Does the branch code agree, on float32 graphs? | 480 of 480 edges within 1.6e-04 on 0.6B and 1.7B | `experiments/015` |
| Does the QK port match the model's scores? | Yes, to 6e-07 worst over 448 heads | `experiments/016` |
| Were the outlier edges the loadings' fault? | No; bfloat16 graphs overstated them 1.3 to 5.6 times | `experiments/017` |

Every number here came from a run whose script, config and raw output are in `experiments/`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="experiments/figures/013-saturation-dark.png">
  <img alt="Normalised attention movement against the share of attributed features ablated, for Qwen3-0.6B and Qwen3-1.7B. Features ranked by attribution move attention much sooner than the same number in random order." src="experiments/figures/013-saturation-light.png">
</picture>

Ablating the top-ranked features against ablating the same number at random, over 60 head-curves.
Half the movement the attributed features can produce comes from about 1.5% of them.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="experiments/figures/015-graph-precision-dark.png">
  <img alt="Distance of each edge's forward effect from its adjacency entry, against the entry's size, for Qwen3-0.6B and Qwen3-1.7B. Edges from bfloat16 graphs spread widely at small sizes; edges from float32 graphs all agree to within 2e-4." src="experiments/figures/015-graph-precision-light.png">
</picture>

The upstream port against 960 graph edges. The spread in bfloat16 graphs belongs to the graph and
not to the head loadings: the same prompts attributed in float32 agree on every edge, and the
worst bfloat16 edges agree once followed into the float32 graphs.

## Upstream

| Pull request | Contents | Evidence |
|---|---|---|
| [circuit-tracer#114](https://github.com/decoderesearch/circuit-tracer/pull/114) | Head loadings: per-layer and single-sweep splits of an edge across attention heads | `experiments/014`, `015`, `017` |

The QK attribution port sits on the fork's `main` at `46cbddd`, checked against real attention
scores in `experiments/016`, and waits on review of the first pull request.

## Modules

| Module | Responsibility |
|---|---|
| `circuits` | Per-head QK and OV weight products, architecture detection, effective rank |
| `scores` | Exact score reconstruction; rotary operators; frozen QK-norm scales |
| `features` | Reading source directions out of a circuit-tracer graph |
| `attribution` | The feature-pair contraction and its remainder accounting |
| `propagate` | Forward propagation through frozen attention |
| `loadings` | Splitting an edge across one layer's heads, or every layer on its path in one sweep |
| `labels` | Feature descriptions, fetched by byte range from the transcoder repository |
| `nodes` | Adjacency index arithmetic |

## Scope

Decomposition runs for one query position and one head at a time. Over all position pairs the
contraction reaches 1.5 PFLOP per head at 512 tokens and the intermediate does not fit in memory;
the arithmetic is in `experiments/004`. Attributing the part of the residual that transcoder
features do not cover, which is most of it, would mean tracing back through earlier attention and is
not implemented.

## Installation

    uv venv --python 3.10 .venv
    . .venv/bin/activate
    uv pip install -e ".[dev]"

Transcoder weights and gated model weights come from Hugging Face, so `hf auth login` is required.

## Checks

Upstream circuit-tracer requires all of these to pass before a PR:

    pytest
    ruff check
    ruff format --check
    pyright

Tests that need a GPU and downloaded weights are marked `gpu` and `slow` and are deselected by
default. Run them with `pytest -m "gpu and slow"`.

## Licence

MIT, matching upstream.
