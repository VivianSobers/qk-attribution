# 018: on Gemma-2-2B the port missed two steps, and now agrees

Date: 2026-09-13
Machines: worker-2 (Gemma-2-2B, Qwen3-1.7B regression check) and worker-1 (Qwen3-0.6B regression
check), RTX 4090 each
Scripts: `experiments/scripts/validate_head_loadings_branch.py`,
`experiments/scripts/diagnose_gemma_readout.py`
Raw output: `experiments/results/018-head-loadings-branch-gemma-2-2b.json` (branch at `dd8ecbe`),
`experiments/results/018-head-loadings-fixed-gemma-2-2b.json` (branch at `6815d77`),
`experiments/results/018-head-loadings-fixed-qwen3-{0.6b,1.7b}-float32-graphs.json`,
`experiments/results/018-gemma-readout-diagnosis.log`
Config: seed 0, torch 2.14.0+cu130, Gemma-2-2B and `mwhanna/gemma-scope-transcoders` in float32,
one graph attributed with `--dtype float32` for "The capital of the state containing Dallas is"
(9 tokens), 60 edges (the 30 strongest forward feature-to-feature edges with |weight| above 1e-4
and 30 drawn at random from all 13.4 million of them), paths of 1 to 20 attention layers.

## Question

Experiments 014 to 017 tested head loadings on Qwen3 only. The demos in circuit-tracer run on
Gemma-2-2B, which differs from Qwen3 in two ways that touch this code. It normalises attention's
output (`ln1_post`) before adding it to the residual stream, and its transcoders read
`ln2.hook_normalized` where the Qwen3 transcoders read `mlp.hook_in`. The pull request accepted
both, so this checks whether it handles them.

## Result

| Gemma-2-2B, float32 graph, 60 edges | branch at `dd8ecbe` | branch at `6815d77` |
|---|---|---|
| effect / adjacency, median | 0.607 | 1.000000 |
| effect / adjacency, range | -7.00 to 2.71 | 0.999914 to 1.000039 |
| within 1e-4 | 0 | 60 |
| within 10% | 9 | 60 |
| strongest 30 within 10% | 9 | 30 |
| edges with the wrong sign | 6 | 0 |
| worst partition error, relative | 1.9e-05 | 1.8e-05 |
| worst single sweep against loop, relative | 3.8e-06 | 9.1e-06 |

The pull request as first opened was wrong on Gemma-2-2B, and on the strong edges as well as the
small ones. This is a float32 graph, so precision cannot account for it. With both fixes every edge
agrees with the adjacency matrix to within 8.6e-05.

The partition held to 1.9e-05 throughout. It compares the module with itself, and a step left out
everywhere leaves it intact, which is why the existing tests passed.

## What was missing

Reading the TransformerLens source gave two candidates. Gemma-2 blocks apply `ln1_post` to attention's
output, and the module never did. And `ln2.hook_normalized` holds the residual divided by the norm's
frozen scale *before* the gain `w` is applied, while the module multiplied by `w` for every hook it
accepted.

A diagnosis run crossed the two on 40 edges (the 20 strongest and 20 random, seed 0), reading the
same propagated perturbation four ways:

| readout | median ratio | within 1e-3 | within 1% | worst \|ratio - 1\| |
|---|---|---|---|---|
| neither fix | 0.617 | 0 | 2 | 3.1 |
| gain removed only | 0.795 | 1 | 5 | 1.7 |
| `ln1_post` applied only | 1.015 | 0 | 1 | 4.8 |
| both | 1.000000 | 40 | 40 | 1.6e-05 |

Neither fix is enough alone. The fixes are two commits on the branch. `b4c3a6e` applies `ln1_post`
at its frozen scale to each head's write, which keeps the split exact because the scale is a
per-position constant, and refuses a model that has the norm when the `FrozenRun` holds no scale for
it. `6815d77` reads each accepted hook as what it holds: the residual before `ln2` for
`hook_mlp_in`, divided by the scale for `ln2.hook_normalized`, and with the gain as well for
`mlp.hook_in`. The tests now check the edge effect against a plain forward loop written separately
in the test file, for all three hooks with and without `ln1_post`, which is the check that would
have caught both.

## A rerun that had to be thrown away

The first check of the `ln1_post` fix alone gave numbers identical to the unfixed branch, down to
the last digit. The cause was the harness, not the fix. The validation script imports the module by
name, Python puts the script's own directory ahead of `PYTHONPATH`, and that directory held the
unfixed copy. The result was deleted. Later runs placed the script beside the module it tests,
recorded the module's md5, and the diagnosis loaded each version from an explicit file path.

## Qwen3 is unchanged

The fixed module was rerun on the float32 graphs from experiment 015, four prompts per model, with
the same 240 edges per model. On Qwen3-0.6B edge effects moved by at most 2.4e-06 relative, the
worst |ratio - 1| is 1.57e-04 as before, and the partition error is at most 9.8e-06. On Qwen3-1.7B
they moved by at most 6.9e-06, the worst |ratio - 1| is 4.9e-05, and the partition error is at most
1.2e-05. The Qwen3 transcoders read `mlp.hook_in` and
the model has no `ln1_post`, so the arithmetic there is the same as before.

## Demo notebooks

The three notebooks upstream's CONTRIBUTING.md asks contributors to check, `circuit_tracing_tutorial`,
`attribute_demo` and `intervention_demo`, were executed end to end with `jupyter nbconvert
--execute` in a fresh environment with the branch installed (nnsight 0.8.0rc1, transformers 4.57.3).
All three finished with no error outputs. They ran on the branch at `dd8ecbe`. The two later commits
change only `head_loadings.py` and its tests, which the notebooks do not import.

## Limits

- One Gemma-2-2B prompt. Gemma-3 and the Gemma cross-layer transcoders were not tried; the latter
  are refused by the module.
- The `hook_mlp_in` readout is checked only against the stub, since no transcoder set tested here
  reads that hook.
