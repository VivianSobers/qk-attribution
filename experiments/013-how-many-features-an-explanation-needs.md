# 013: half the movement comes from 1.5% of the features, and the other half is expensive

Date: 2026-09-13
Machines: worker-1 and worker-2, RTX 4090 each
Scripts: `experiments/scripts/measure_saturation.py`, pooled by
`experiments/scripts/summarise_saturation.py`
Raw output: `experiments/results/013-saturation-{0.6b,0.6b-p1,0.6b-p2,0.6b-p3,1.7b}.json`
Config: seed 0, five graphs over four prompts, Qwen3-0.6B and Qwen3-1.7B in float32, query position
last, three random draws averaged at every k, features ablated where they are written with the whole
model free to respond. 60 head-curves.

## Question

Experiment 012 ablated the top k attributed features for k up to 250 and found the ranking
informative at every k it tried. It never reached the point where the curve flattens, so it says
nothing about how long an explanation has to be. An explanation that names 250 features is not an
explanation, and 012 could not rule that out.

The fix is to run k all the way to the full pool. Ablating every attributed feature is the largest
movement this decomposition can produce for a head, so it is the natural denominator: `k50` and
`k90` are the smallest k reaching half and nine tenths of it.

## What the denominator is, and what it is not

Ablating every attributed feature moves the attention row by a median total variation distance of
0.316, with a range of 0.090 to 0.782. The features are therefore a minority of where the head
looks, which is what experiment 005 found by a different route. `k50` is half of what the features
can do, not half of the attention pattern.

## Result

| | features at `k50` | share of pool | features at `k90` | share of pool |
|---|---|---|---|---|
| Qwen3-0.6B, 48 heads | 20 | 1.5% | 134 | 12.1% |
| Qwen3-1.7B, 12 heads | 77 | 1.4% | 558 | 10.5% |
| pooled, 60 heads | 25 | 1.5% | 197 | 12.1% |
| random order, pooled | 420 | 33% | 966 | 100% |

Medians throughout. The pool is every attributed feature below the head's layer, a median of 1260
on 0.6B and 5332 on 1.7B.

Half the movement the features can produce comes from about 1.5% of them, and the ranking finds
those. The same fraction on both models, with the absolute count growing roughly with the pool, so
this reads as a property of the ranking rather than of the model.

The second half is where it gets expensive. Reaching 90% takes an order of magnitude more features,
12% of the pool, and on 7 of 60 heads it takes the entire pool. So there is a short list worth
reading and a long tail that cannot be summarised, and the tail is not small.

## The first crossing is a noisy statistic, so here is a robust one

The curves are not monotone. Features cancel, so ablating a larger set sometimes moves attention
less than ablating a subset of it: the ranked curve steps downward at 286 of 780 k-increments.
A first crossing of a wobbling curve is sensitive to one point.

The area under the normalised curve against log k does not have that problem. It is 1.0 for a
ranking that puts all the movement in the first feature and 0.0 for one that puts it all in the
last.

| | area, ranked | area, random | per-head ratio |
|---|---|---|---|
| Qwen3-0.6B | 0.570 | 0.278 | 2.02x |
| Qwen3-1.7B | 0.525 | 0.193 | 2.57x |
| pooled | 0.544 | 0.247 | 2.09x |

The ranked ablation moves attention further than a random one of the same size at 725 of 780
(head, k) points. That replicates 012 on a wider grid and with a cleaner summary.

## Concentration is easiest where it matters least

Rank the heads by how concentrated their explanation is and by how much the features can move them
at all, and the two agree: Spearman +0.48 for the `k50` share and +0.54 for `k90`. The heads with
the shortest explanations are the heads the features barely move.

The extreme case is Qwen3-1.7B layer 7 head 8, where two features out of 5332 carry half the
movement and four carry 90%, and the whole pool moves the row by 0.254. At the other end, layer 7
head 15 of the same model has the largest movement in the sample at 0.780, and needs 736 features
for half of it.

So the short explanations are real, and they are concentrated in the heads where the decomposition
has least to explain. Anyone quoting a median here should quote the correlation next to it.

## What this bounds

For a head picked at random from this sample, reading the top 25 attributed features accounts for
half of what every attributed feature together can do to its attention pattern. That is a usable
budget for a graph interface. Reading the top 200 gets to 90%, which is not.

## Next

- Plot the curves. The tables above hide the shape, and the shape is the finding.
- The same measurement on the 4B and 8B high-L0 sets, where the feature share of the score is much
  higher and the denominator therefore means more.
