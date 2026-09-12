"""Check frozen-attention propagation against the real model it claims to describe.

The unit tests prove the split is a partition of whatever ``propagate`` computes. This proves
``propagate`` computes the right thing: perturb the residual stream at one layer with attention
patterns, layernorm scales and MLP outputs all pinned to their clean values, and the change at a
later layer must be what the propagation predicts. Those are exactly the constraints an attribution
graph is built under.

Needs a CUDA device and downloads about 1.5 GB of weights, so it is marked ``gpu`` and ``slow``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")

from transformer_lens import HookedTransformer  # noqa: E402

from qk_attribution.propagate import propagate, split_by_head  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of the state containing Dallas is"
SEED = 0
TOLERANCE = 1e-4

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def model_and_cache():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(SEED)
    with torch.no_grad():
        model = HookedTransformer.from_pretrained_no_processing(
            MODEL, device="cuda", dtype=torch.float32
        )
        tokens = model.to_tokens(PROMPT)
        _, cache = model.run_with_cache(tokens)
    return model, tokens, cache


def freezing_hooks(model, cache, delta, from_layer):
    """Hooks that pin everything an attribution graph pins, then inject the perturbation."""

    def pin(name):
        def hook(value, hook):  # noqa: A002, ARG001
            return cache[name]

        return (name, hook)

    def inject(value, hook):  # noqa: A002, ARG001
        return value + delta

    hooks = [(f"blocks.{from_layer}.hook_resid_pre", inject)]
    for layer in range(model.cfg.n_layers):
        hooks.append(pin(f"blocks.{layer}.attn.hook_pattern"))
        hooks.append(pin(f"blocks.{layer}.ln1.hook_scale"))
        hooks.append(pin(f"blocks.{layer}.hook_mlp_out"))
    return hooks


@pytest.mark.parametrize(("from_layer", "to_layer"), [(0, 3), (5, 12), (10, 28), (20, 27)])
def test_propagation_matches_a_frozen_forward_pass(model_and_cache, from_layer, to_layer):
    model, tokens, cache = model_and_cache
    seq = tokens.shape[1]
    generator = torch.Generator(device="cuda").manual_seed(SEED + from_layer)
    delta = torch.randn(seq, model.cfg.d_model, generator=generator, device="cuda") * 0.1

    with torch.no_grad():
        read = f"blocks.{to_layer}.hook_resid_pre" if to_layer < model.cfg.n_layers else None
        names = [read] if read else [f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]
        with model.hooks(fwd_hooks=freezing_hooks(model, cache, delta.unsqueeze(0), from_layer)):
            _, perturbed = model.run_with_cache(tokens, names_filter=lambda name: name in names)
        observed = perturbed[names[0]][0] - cache[names[0]][0]
        predicted = propagate(model, cache, delta, from_layer, to_layer)

    error = ((predicted - observed).norm() / observed.norm()).item()
    assert error < TOLERANCE, f"{from_layer} to {to_layer}: relative error {error:.3e}"


def test_head_split_partitions_a_real_propagation(model_and_cache):
    model, tokens, cache = model_and_cache
    seq = tokens.shape[1]
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    delta = torch.randn(seq, model.cfg.d_model, generator=generator, device="cuda") * 0.1

    with torch.no_grad():
        parts = split_by_head(model, cache, delta, 5, 9, 20)
        whole = propagate(model, cache, delta, 5, 20)

    assert parts.shape == (model.cfg.n_heads + 1, seq, model.cfg.d_model)
    error = ((parts.sum(dim=0) - whole).norm() / whole.norm()).item()
    assert error < TOLERANCE, f"head split lost {error:.3e} of the total"
