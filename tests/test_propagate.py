"""Tests for forward propagation through frozen attention.

The property that carries the whole method is that the head split is a partition: the parts must
sum to the unsplit result. A split that lost or double counted a path would still produce
per-head numbers that looked reasonable.
"""

from __future__ import annotations

import pytest
import torch

from qk_attribution.propagate import (
    attention_step,
    frozen_patterns,
    propagate,
    split_by_head,
)
from tests.stubs import D_MODEL, N_HEADS, N_LAYERS, make_model

SEQ = 5


def make_cache(n_layers: int = N_LAYERS, seq: int = SEQ, seed: int = 17) -> dict:
    """Frozen patterns and layernorm scales, the way a cached run supplies them."""
    gen = torch.Generator().manual_seed(seed)
    cache = {}
    for layer in range(n_layers):
        scores = torch.randn(N_HEADS, seq, seq, generator=gen)
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        cache[f"blocks.{layer}.attn.hook_pattern"] = torch.softmax(
            scores.masked_fill(~mask, float("-inf")), dim=-1
        )
        cache[f"blocks.{layer}.ln1.hook_scale"] = torch.rand(seq, 1, generator=gen) + 0.5
        cache[f"blocks.{layer}.ln2.hook_scale"] = torch.rand(seq, 1, generator=gen) + 0.5
    return cache


def delta(seq: int = SEQ, seed: int = 19) -> torch.Tensor:
    return torch.randn(seq, D_MODEL, generator=torch.Generator().manual_seed(seed))


def test_attention_step_matches_an_explicit_per_head_loop():
    model = make_model()
    cache = make_cache()
    patterns = frozen_patterns(cache, 0)
    scale = cache["blocks.0.ln1.hook_scale"]
    perturbation = delta()
    attn = model.blocks[0].attn
    normalised = perturbation * model.blocks[0].ln1.w / scale
    expected = torch.zeros(SEQ, D_MODEL)
    for head in range(N_HEADS):
        values = normalised @ attn.W_V[head]
        expected = expected + (patterns[head] @ values) @ attn.W_O[head]
    torch.testing.assert_close(attention_step(model, 0, perturbation, patterns, scale), expected)


def test_per_head_slices_sum_to_the_whole_step():
    model = make_model()
    cache = make_cache()
    patterns, scale = frozen_patterns(cache, 1), cache["blocks.1.ln1.hook_scale"]
    perturbation = delta()
    combined = attention_step(model, 1, perturbation, patterns, scale)
    split = attention_step(model, 1, perturbation, patterns, scale, per_head=True)
    assert split.shape == (N_HEADS, SEQ, D_MODEL)
    torch.testing.assert_close(split.sum(dim=0), combined)


def test_attention_step_is_linear_in_the_perturbation():
    """Linearity is what lets a residual be decomposed and each part propagated separately."""
    model = make_model()
    cache = make_cache()
    patterns, scale = frozen_patterns(cache, 0), cache["blocks.0.ln1.hook_scale"]
    first, second = delta(seed=1), delta(seed=2)
    combined = attention_step(model, 0, first + 3.0 * second, patterns, scale)
    separate = attention_step(model, 0, first, patterns, scale) + 3.0 * attention_step(
        model, 0, second, patterns, scale
    )
    torch.testing.assert_close(combined, separate)


def test_propagating_zero_layers_is_the_identity():
    model = make_model()
    perturbation = delta()
    torch.testing.assert_close(propagate(model, make_cache(), perturbation, 1, 1), perturbation)


def test_propagation_composes_across_layers():
    model = make_model()
    cache = make_cache()
    perturbation = delta()
    once = propagate(model, cache, perturbation, 0, 1)
    torch.testing.assert_close(
        propagate(model, cache, perturbation, 0, 2), propagate(model, cache, once, 1, 2)
    )


def test_head_split_is_an_exact_partition():
    model = make_model()
    cache = make_cache()
    perturbation = delta()
    parts = split_by_head(model, cache, perturbation, 0, 0, N_LAYERS)
    assert parts.shape == (N_HEADS + 1, SEQ, D_MODEL)
    torch.testing.assert_close(parts.sum(dim=0), propagate(model, cache, perturbation, 0, N_LAYERS))


def test_head_split_at_a_later_layer_is_also_a_partition():
    model = make_model()
    cache = make_cache()
    perturbation = delta()
    parts = split_by_head(model, cache, perturbation, 0, 1, N_LAYERS)
    torch.testing.assert_close(parts.sum(dim=0), propagate(model, cache, perturbation, 0, N_LAYERS))


def test_bypass_is_the_last_slice_and_carries_the_untouched_part():
    model = make_model()
    cache = make_cache()
    perturbation = delta()
    parts = split_by_head(model, cache, perturbation, 0, 0, 1)
    torch.testing.assert_close(parts[N_HEADS], perturbation)


def test_backwards_propagation_is_rejected():
    with pytest.raises(ValueError, match="from_layer <= to_layer"):
        propagate(make_model(), make_cache(), delta(), 2, 1)


def test_attention_layer_outside_the_span_is_rejected():
    with pytest.raises(ValueError, match="attention_layer must lie"):
        split_by_head(make_model(), make_cache(), delta(), 0, 2, 2)


def test_malformed_delta_is_rejected():
    model = make_model()
    cache = make_cache()
    with pytest.raises(ValueError, match="must be 2D"):
        attention_step(
            model,
            0,
            delta().unsqueeze(0),
            frozen_patterns(cache, 0),
            cache["blocks.0.ln1.hook_scale"],
        )


def test_pattern_head_count_is_checked():
    model = make_model()
    cache = make_cache()
    with pytest.raises(ValueError, match="patterns cover"):
        attention_step(
            model,
            0,
            delta(),
            frozen_patterns(cache, 0)[:2],
            cache["blocks.0.ln1.hook_scale"],
        )


def test_batched_pattern_loses_its_batch_axis():
    cache = make_cache()
    cache["blocks.0.attn.hook_pattern"] = cache["blocks.0.attn.hook_pattern"].unsqueeze(0)
    assert frozen_patterns(cache, 0).shape == (N_HEADS, SEQ, SEQ)


def test_propagation_follows_the_perturbations_dtype():
    """Graphs are often built in bfloat16, so the propagation must not force float32."""
    model, cache = make_model(), make_cache()
    perturbation = delta().to(torch.bfloat16)
    result = propagate(model, cache, perturbation, 0, N_LAYERS)
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(
        result.float(), propagate(model, cache, delta(), 0, N_LAYERS), rtol=0.05, atol=0.05
    )
