# 008: what a QK attribution actually says, once features are labelled

Date: 2026-09-12
Machine: worker-1, RTX 4090
Script: `experiments/scripts/explain_attention.py`
Raw output: `experiments/results/008-explain-qwen3-0.6b-L{12,21,25}.json`
Config: Qwen3-0.6B, float32 model, bfloat16 transcoders, seed 0, the 9-token graph from experiment
001, query position 8. Feature labels from the transcoder repository's own metadata.

## Question

Experiments 003 to 007 establish that the decomposition is exact and that feature terms carry a
minority of the score. None of that says whether the feature pairs it produces mean anything. This
reads them.

The prompt is `The capital of the state containing Dallas is`, and the query position is the final
token.

## Labels come from the transcoder repository, one range request each

Each transcoder repository publishes per-feature metadata: the tokens a feature promotes, how often
it fires, and the examples it fires hardest on. The files are about 1.3 GB per layer, but an index
gives each feature's byte range, so a label costs one HTTP range request rather than a download.
`qk_attribution.labels` does this and caches the chunks.

## The key side is right, and it is readable

Summing contributions over the query side gives the reading that matters: which feature at which
key position drew the head there.

Layer 12, head 5, which places 90.8% of its attention on content rather than on the sink token:

| Contribution | Key position | Feature |
|---|---|---|
| +0.932 | 4 `' the'` | L10#35961, fires on `' his' ' the' ' P' ' a'` |
| -0.607 | 4 `' the'` | L11#91721, fires on `'1' '2' ' the' ' an'` |
| +0.291 | 3 `' of'` | L7#68729, promotes `' city'` |
| +0.284 | 2 `' capital'` | L7#68729, promotes `' city'` |
| +0.269 | 5 `' state'` | L10#141346, fires on `' state' ' State' ' states' ' States'` |

Layer 21, head 3:

| Contribution | Key position | Feature |
|---|---|---|
| +0.330 | 2 `' capital'` | L20#126331, fires on `' capital' ' Capital' ' capitals' 'itol'` |
| +0.269 | 4 `' the'` | L20#132183, fires on `' the' 'the' ' The' 'The'` |

Layer 25, head 8 puts its largest term on the same capital feature at the same position.

These are correct. The head attends to `' state'` because a states feature is active there, and to
`' capital'` because a capitals feature is active there. A city-promoting feature sits on `' of'`
and `' capital'`, which is the right shape for a prompt whose answer is a city. That is the claim
the method exists to support, and on this prompt it holds.

## The query side does not read as cleanly

The strongest individual pairs are dominated by layer 0 and layer 1 features whose labels are not
interpretable: firing on `'t' 'l' 's' 'm'` and promoting Cyrillic fragments. There are many more
early-layer features than late ones in these graphs, and their decoder directions are not small, so
they crowd the per-pair ranking even when each contributes little that can be read.

Aggregating over the query side is what makes the result legible. Anyone using this should read the
key-side summary first and treat individual pairs as a drill-down.

## Attention sinks dominate, and they hide the interesting heads

At layers 18 and above, the median head sends under 11% of its attention from the final token to
anything other than the first token or itself. The first version of this script picked heads by
lowest attention entropy, which selected exactly those sink heads: one had 97.8% of its mass on
`<|im_end|>` and explained nothing. Selecting by attention mass on content instead finds heads that
retrieve. This is a property of the model, and it is consistent with the large-magnitude residual
directions recorded in experiment 005.

## Caveats

Feature terms are 7.6% to 12.4% of total contribution magnitude here, matching experiments 005.
Reconstruction error is 4e-04 to 1e-03 because the transcoders are read in bfloat16; the float32
path in experiment 005 reaches 3e-07.

One prompt, one query position, three heads. This shows the method produces correct readings where
it produces readings at all. It does not establish how often that happens.

## Next

- The same reading on the higher-L0 4B transcoders, where the query side may be legible too
- Aggregate over many prompts to find how often the top key-side feature matches the token
