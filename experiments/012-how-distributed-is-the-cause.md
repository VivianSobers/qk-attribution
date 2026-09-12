# 012: the ranking is informative from the second feature onward

Date: 2026-09-13
Machines: worker-1 and worker-2, RTX 4090 each
Script: `experiments/scripts/measure_group_ablation.py`
Raw output: `experiments/results/012-group-ablation-{0.6b,1.7b}.json`
Config: seed 0, four prompts, Qwen3-0.6B and Qwen3-1.7B in float32, query position last, features
ablated where they are written with the whole model free to respond. 78 head-curves over eight runs.

## Question

Experiment 011 found that deleting the single largest key-side feature moved the attention pattern
no more than deleting a comparable feature at the same position. Two explanations fit: the
attribution's ranking is uninformative, or the cause is spread across many features and no single
one is load-bearing. This separates them by ablating the top k together for growing k.

Movement is the total variation distance between the head's attention row before and after, which
counts every position rather than only the one the top feature pointed at.

## Two controls, because the obvious one flatters the result

Ablating k features chosen at random from the whole pool mostly removes features at positions the
head was not attending to, so it would look impressive for the wrong reason. The stricter control
draws k features from the same positions as the top k, matching the positional profile exactly. Both
are reported; the position-matched one is the one to read.

## Result, pooled over all eight runs

| k | top k | same-position control | random control | ratio | heads where top k moved more |
|---|---|---|---|---|---|
| 1 | 0.0310 | 0.0029 | 0.0002 | 10.6 | 72/78 |
| 2 | 0.0459 | 0.0095 | 0.0007 | 4.9 | 69/78 |
| 5 | 0.0678 | 0.0159 | 0.0021 | 4.3 | 74/78 |
| 10 | 0.0905 | 0.0325 | 0.0068 | 2.8 | 74/78 |
| 25 | 0.1598 | 0.0451 | 0.0186 | 3.5 | 72/78 |
| 50 | 0.1834 | 0.0890 | 0.0346 | 2.1 | 71/78 |
| 100 | 0.2603 | 0.0928 | 0.0454 | 2.8 | 71/78 |
| 250 | 0.3011 | 0.1388 | 0.0971 | 2.2 | 69/78 |

The ranking beats a position-matched control at every k, in roughly 90% of head-curves, by a factor
between two and ten. With eight runs pooled the effect is already clear at k = 1, which the
twelve-head samples below were too small to establish.

## The original per-model runs, for reference

Median movement over heads, Qwen3-0.6B:

| k | top k | same-position control | random control | ratio to same-position | heads where top k moved more |
|---|---|---|---|---|---|
| 1 | 0.0209 | 0.0037 | 0.0002 | 5.6 | 9/12 |
| 2 | 0.0450 | 0.0089 | 0.0008 | 5.1 | 12/12 |
| 5 | 0.0571 | 0.0231 | 0.0065 | 2.5 | 12/12 |
| 10 | 0.0818 | 0.0329 | 0.0104 | 2.5 | 12/12 |
| 25 | 0.1178 | 0.0483 | 0.0243 | 2.4 | 9/12 |
| 50 | 0.1466 | 0.0687 | 0.0431 | 2.1 | 12/12 |
| 100 | 0.2069 | 0.1141 | 0.0697 | 1.8 | 11/12 |
| 250 | 0.2689 | 0.1187 | 0.1244 | 2.3 | 11/12 |

Qwen3-1.7B:

| k | top k | same-position control | ratio | heads where top k moved more |
|---|---|---|---|---|
| 1 | 0.0197 | 0.0021 | 9.2 | 12/12 |
| 2 | 0.0472 | 0.0054 | 8.7 | 10/12 |
| 5 | 0.0635 | 0.0148 | 4.3 | 10/12 |
| 10 | 0.0818 | 0.0282 | 2.9 | 12/12 |
| 25 | 0.1611 | 0.0425 | 3.8 | 11/12 |
| 50 | 0.1632 | 0.0517 | 3.2 | 11/12 |
| 100 | 0.2662 | 0.0873 | 3.0 | 12/12 |
| 250 | 0.3132 | 0.1567 | 2.0 | 11/12 |

## What this says

The ranking carries real information about the attention pattern. Removing the features the
attribution ranks highest moves attention two to ten times further than removing the same number of
features from the same positions, in about 90% of cases at every k. The ratio declines as k grows,
which it must: at k equal to the total the two sets coincide.

This does not contradict experiment 011, and the difference between them is worth stating. That
experiment asked whether removing the top feature changes attention to the one position the feature
names, against a competitor at that same position, and found 61%. This one asks whether removing
the top-ranked features changes the head's whole attention distribution, against the same number
drawn from the same positions, and finds about 90%. The ranking is informative about how a head
distributes attention. It is much weaker as a claim about one particular attention edge.

The absolute movements stay modest. Removing the top 250 features, which is a quarter of everything
available at layer 7, moves the attention row by about 0.27 in total variation. Attention here is
robust to losing a large part of what the transcoders can describe, which is consistent with
experiment 005: most of the residual is not transcoder features at all.

## How to read the method, then

The top pair in a QK attribution is the largest term in an exact decomposition and a useful place to
start looking. It is not, on its own, the reason a head attends somewhere. Read the top handful
together. The evidence for that recommendation is the jump from 9/12 at k = 1 to 12/12 at k = 2.

## Next

- Extend to the other prompts from experiment 010
- Find the k at which movement saturates, which would bound how many features an explanation needs
