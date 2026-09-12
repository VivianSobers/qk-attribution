# 010: the explanations point at the right token 79% of the time

Date: 2026-09-13
Machines: worker-1 and worker-2, RTX 4090 each
Script: `experiments/scripts/measure_label_match.py`
Raw output: `experiments/results/010-labelmatch-*.json`
Config: seed 0, four prompts, three models, one attribution graph per prompt and model. Heads
included when at least 20% of their attention from the final token lands on something other than
the first token or the query itself. Qwen3-4B in bfloat16, the others in float32. 412 head cases in
total.

## Question

Experiment 008 reads one head of one prompt and finds the explanation sensible. That is an
anecdote. This asks how often it happens.

## The test

For each qualifying head, take the strongest key-side feature, which is the decomposition's answer
to why the head looked where it did. Ask whether that feature actually fires on the token sitting at
the position it points to, comparing its own top eight activating tokens case-insensitively and
ignoring leading spaces.

The control is the same question asked of a randomly chosen other position in the same prompt. It
matters because determiners and punctuation are common: a feature that fires on `' the'` will match
any position holding `' the'`, and without a baseline that would look like success.

## Result

| Model | Prompt | Matches | Control |
|---|---|---|---|
| 0.6B | capital / Dallas | 41/45 = 91.1% | 15.6% |
| 0.6B | Eiffel Tower | 31/45 = 68.9% | 17.8% |
| 0.6B | Michael Jordan | 32/36 = 88.9% | 2.8% |
| 0.6B | Japanese currency | 42/45 = 93.3% | 13.3% |
| 1.7B | capital / Dallas | 39/40 = 97.5% | 12.5% |
| 1.7B | Eiffel Tower | 35/43 = 81.4% | 11.6% |
| 1.7B | Michael Jordan | 17/30 = 56.7% | 0.0% |
| 1.7B | Japanese currency | 20/40 = 50.0% | 2.5% |
| 4B | capital / Dallas | 70/88 = 79.5% | 10.2% |
| **All** | | **327/412 = 79.4%** | **10.2%** |

The gap is roughly eightfold and holds on every prompt and model tested. The per-prompt rate is not
stable, ranging from 50% to 97.5%, and the variation does not follow model size: 1.7B is the best on
one prompt and the worst on two others.

Everything before this experiment established that the decomposition is arithmetically right.
This is the first evidence that its output is about anything.

## What this does and does not show

It shows a necessary condition. If the feature the decomposition names as the reason for attending
to a position does not even fire on the token at that position, the explanation is not about that
position. Passing at 79% against a 10% baseline says the decomposition usually points somewhere
defensible.

It does not show the explanation is causally right. A feature can fire on a token and still not be
what drew the head there, and this test cannot tell those apart. That would need intervention:
suppress the named feature and see whether the attention pattern moves. Nothing here does that.

The control is also not as strict as it could be. A random other position sometimes holds the same
token as the true one, which makes a control match possible without coincidence. That biases the
baseline upward, so the real gap is at least the one reported.

## A practical note

Some label fetches failed with HTTP errors during the 1.7B run and those heads were skipped rather
than counted either way. Three of roughly 160. The store has no retry.

## Next

- Intervene: ablate the named key-side feature and measure how far the attention pattern moves
- Add retry with backoff to the label store, since skipped heads are silently dropped from the
  denominator
