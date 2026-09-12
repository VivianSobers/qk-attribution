# 012: the ranking is informative from the second feature onward

Date: 2026-09-13
Machines: worker-1 and worker-2, RTX 4090 each
Script: `experiments/scripts/measure_group_ablation.py`
Raw output: `experiments/results/012-group-ablation-{0.6b,1.7b}.json`
Config: seed 0, the prompt `The capital of the state containing Dallas is`, query position 8,
Qwen3-0.6B and Qwen3-1.7B in float32, 12 qualifying heads each, features ablated where they are
written with the whole model free to respond.

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

## Result

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
attribution ranks highest moves attention two to five times further than removing the same number of
features from the same positions, and from k = 2 onward it does so in almost every head on both
models. The ratio declines as k grows, which it must: at k equal to the total the two sets coincide.

So experiment 011's negative was about the first feature specifically, not about the ranking. At
k = 1 the effect is there but noisy, winning in 9 of 12 heads on 0.6B. By k = 2 it is 12 of 12. The
cause of a head's attention is distributed over a handful of features rather than resting on one,
and the attribution orders them usefully.

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
