# 011: the attributed numbers are causally exact, and the frozen scales are load-bearing

Date: 2026-09-13
Machines: worker-1 and worker-2, RTX 4090 each
Script: `experiments/scripts/measure_intervention.py`
Raw output: `experiments/results/011-intervention-{0.6b,1.7b}.json`
Config: seed 0, the prompt `The capital of the state containing Dallas is`, query position 8,
Qwen3-0.6B and Qwen3-1.7B in float32, 36 and 40 qualifying heads.

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

| Model | Heads agreeing to better than 1e-04 | Median relative error | Correlation |
|---|---|---|---|
| 0.6B | 36/36 | 9.6e-07 | 1.000000 |
| 1.7B | 40/40 | 4.6e-07 | 1.000000 |

Largest absolute disagreement is 3.7e-06 on score changes whose median size is 0.85. The
attribution is not an approximation of the intervention; under the stated assumptions it is the
intervention.

## The frozen scales are doing a lot of work

The result above holds the normalisation scales at their clean values, which is what the
decomposition assumes and what circuit-tracer already does for layernorm. Repeating the identical
intervention while letting the scales recompute:

| Model | Median relative error | Correlation |
|---|---|---|
| 0.6B | 1.24 | 0.175 |
| 1.7B | 0.39 | 0.536 |

So the same deletion, measured on the same model, produces a score change the attribution predicts
perfectly or barely at all, depending only on whether RMSNorm is allowed to respond. Removing a
feature changes the norm of the residual at that position, which rescales everything else there.

This is a caveat on the frozen-scale methodology rather than on this implementation. Any attribution
that freezes normalisation inherits it. It deserves saying plainly: the numbers are exact
descriptions of a linearised model, and the linearisation is not a small correction here.

## The named feature does not control the attention pattern

The decomposition names one feature as the largest reason a head attends where it does. Ablating
that feature where it is actually written, and letting the whole model respond, moves the attention
probability by a median of -0.0010 on 0.6B and -0.0013 on 1.7B. Ablating a different feature at the
same position with a comparable activation moves it by +0.0001 and +0.0007.

The named feature moved attention more than the control in 21 of 36 heads on 0.6B and 18 of 40 on
1.7B. That is a coin flip.

This is a clean negative and it matters for how the method should be read. Being the largest single
term in an exact decomposition does not make a feature the cause of the attention pattern. The terms
are many and they largely cancel, the softmax compresses what survives, and attention to a given
position is supported by a broad set of contributions rather than one. A user reading the top pair
off a graph is reading the largest term, not the load-bearing one.

It also sits consistently with experiment 006: the top 1% of pairs hold under a third of the
magnitude, so no individual pair should be expected to dominate.

## What would strengthen this

Ablating the top k features together, rather than one, and finding the k at which the attention
pattern does move. That would turn the negative into a measured statement about how distributed the
cause is. It needs no new machinery, only a loop.

## Next

- Group ablation, as above
- Repeat on the other prompts from experiment 010
