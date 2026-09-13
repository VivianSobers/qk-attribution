# 011: the attributed numbers are causally exact, and the frozen scales are load-bearing

Date: 2026-09-13
Machines: worker-1 and worker-2, RTX 4090 each
Script: `experiments/scripts/measure_intervention.py`
Raw output: `experiments/results/011-intervention-{0.6b,1.7b}.json`
Config: seed 0, four prompts, Qwen3-0.6B and Qwen3-1.7B in float32, query position last, 263
qualifying heads across eight runs.

## Question

Everything up to experiment 010 is observational. The decomposition reproduces the scores the model
computed, and the features it names usually fire on the tokens it points at. Neither shows that the
attributed numbers predict what happens if you remove a feature.

## The test

For each head, take the strongest key-side feature, delete its direction from the residual stream at
the attention layer's input, and compare the resulting change in the pre-softmax score against the
change the decomposition predicts.

The prediction is not simply the negative of that feature's column. The score is bilinear, so when
the feature sits at the query position as well, deleting it moves both arguments:

    s -> <q - q_j, k - k_j> = s - <q_j, k> - <q, k_j> + <q_j, k_j>

and the query-side term covers only key sources at that same position. Getting either of those
wrong produces predictions that are 4 to 6 times too large, which is how both errors were found.

## The prediction is exact

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/011-interventions-dark.png">
  <img alt="Predicted against measured score change for 263 heads, with normalisation frozen and with it free to respond." src="figures/011-interventions-light.png">
</picture>

Every head from both models and all four prompts, drawn by `experiments/scripts/plot_interventions.py`.

| Model | Prompt | Heads agreeing to better than 1e-04 |
|---|---|---|
| 0.6B | capital / Dallas | 36/36 |
| 0.6B | Eiffel Tower | 31/31 |
| 0.6B | Michael Jordan | 24/24 |
| 0.6B | Japanese currency | 29/29 |
| 1.7B | capital / Dallas | 40/40 |
| 1.7B | Eiffel Tower | 34/34 |
| 1.7B | Michael Jordan | 35/35 |
| 1.7B | Japanese currency | 34/34 |
| **All** | | **263/263** |

Correlation between predicted and actual is 1.000000 on every run, median relative error under
1e-06, largest absolute disagreement 3.7e-06 on score changes whose median size is 0.85. The
attribution is not an approximation of the intervention; under the stated assumptions it is the
intervention.

## The frozen scales are doing a lot of work

The result above holds the normalisation scales at their clean values, which is what the
decomposition assumes and what circuit-tracer already does for layernorm. Repeating the identical
intervention while letting the scales recompute:

Correlation between the prediction and the unfrozen outcome, by run: -0.08, +0.03, +0.18, +0.86 on
0.6B, and +0.54, +0.79, +0.86, +0.93 on 1.7B. Median relative error ranges from 0.39 to 1.38.

So the same deletion, measured on the same model, produces a score change the attribution predicts
perfectly or barely at all, depending only on whether RMSNorm is allowed to respond. How much is
lost is prompt-dependent and not predictable from the frozen numbers. Removing a
feature changes the norm of the residual at that position, which rescales everything else there.

This is a caveat on the frozen-scale methodology rather than on this implementation. Any attribution
that freezes normalisation inherits it. It deserves saying plainly: the numbers are exact
descriptions of a linearised model, and the linearisation is not a small correction here.

## The named feature does not control the attention pattern

The decomposition names one feature as the largest reason a head attends where it does. Ablating
that feature where it is actually written, and letting the whole model respond, moves the attention
probability by a median of -0.0010 on 0.6B and -0.0013 on 1.7B. Ablating a different feature at the
same position with a comparable activation moves it by +0.0001 and +0.0007.

Pooled over all eight runs the named feature moved attention to that position more than the control
in 160 of 263 heads, or 61%. Better than chance, but not by much.

This is a clean negative and it matters for how the method should be read. Being the largest single
term in an exact decomposition does not make a feature the cause of the attention pattern. The terms
are many and they largely cancel, the softmax compresses what survives, and attention to a given
position is supported by a broad set of contributions rather than one. A user reading the top pair
off a graph is reading the largest term, not the load-bearing one.

It also sits consistently with experiment 006: the top 1% of pairs hold under a third of the
magnitude, so no individual pair should be expected to dominate.

## This is narrower than it first appears

The test above asks about attention to the one position the named feature points at. Experiment 012
asks a broader question, whether removing the top-ranked features moves the head's whole attention
distribution more than removing the same number from the same positions, and there the answer is
yes in 72 of 78 cases even at k = 1.

Both are true. The top feature reliably shifts how a head distributes its attention, and does not
reliably shift attention to the specific position it names more than a same-position competitor
would. Ranking first by contribution is informative; reading the top pair as the cause of one
particular attention edge is not supported.

## Next

- Find the k at which movement saturates, which would bound how many features an explanation needs
