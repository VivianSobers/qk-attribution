"""Tests for splitting an edge's effect across the heads of one attention layer.

Every test here is a form of the same property: the split must be a partition. A head loading that
lost a path, or counted the target block's own attention twice, would still produce a plausible
ranking of heads, and nothing downstream would notice.
"""

from __future__ import annotations

import pytest
import torch

from qk_attribution.loadings import edge_effect, edge_loadings
from qk_attribution.propagate import to_transcoder_input
from tests.stubs import D_MODEL, N_HEADS, N_LAYERS, make_model
from tests.test_propagate import make_cache

SEQ = 5


def direction(seed: int = 23) -> torch.Tensor:
    return torch.randn(D_MODEL, generator=torch.Generator().manual_seed(seed))


def test_effect_matches_propagating_by_hand():
    model, cache = make_model(n_layers=4), make_cache(n_layers=4)
    source, reader = direction(1), direction(2)
    delta = torch.zeros(SEQ, D_MODEL)
    delta[1] = source
    from qk_attribution.propagate import propagate

    arriving = propagate(model, cache, delta, 1, 3)
    expected = to_transcoder_input(model, cache, arriving, 3)[4] @ reader
    got = edge_effect(model, cache, source, 1, 0, reader, 4, 3)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("attention_layer", [1, 2, 3])
def test_loadings_sum_to_the_edge_effect(attention_layer: int):
    model, cache = make_model(n_layers=4), make_cache(n_layers=4)
    source, reader = direction(3), direction(4)
    result = edge_loadings(model, cache, source, 1, 0, reader, 4, 3, attention_layer)
    expected = edge_effect(model, cache, source, 1, 0, reader, 4, 3)
    torch.testing.assert_close(result.total, expected, rtol=1e-4, atol=1e-6)


def test_loadings_have_one_entry_per_head_plus_a_bypass():
    model, cache = make_model(), make_cache()
    result = edge_loadings(model, cache, direction(), 0, 0, direction(5), 3, 1, 1)
    assert result.per_head.shape == (N_HEADS,)
    assert result.bypass.ndim == 0
    assert result.attention_layer == 1


def test_a_zero_source_loads_nothing():
    model, cache = make_model(), make_cache()
    result = edge_loadings(model, cache, torch.zeros(D_MODEL), 0, 0, direction(), 2, 1, 1)
    torch.testing.assert_close(result.per_head, torch.zeros(N_HEADS))
    torch.testing.assert_close(result.bypass, torch.zeros(()))


def test_loadings_are_linear_in_the_source():
    model, cache = make_model(), make_cache()
    reader = direction(6)
    scaled = edge_loadings(model, cache, direction(7) * 2.5, 0, 0, reader, 3, 1, 1)
    single = edge_loadings(model, cache, direction(7), 0, 0, reader, 3, 1, 1)
    torch.testing.assert_close(scaled.per_head, single.per_head * 2.5, rtol=1e-4, atol=1e-6)


def test_ranked_orders_heads_by_magnitude():
    model, cache = make_model(), make_cache()
    result = edge_loadings(model, cache, direction(8), 0, 0, direction(9), 3, 1, 1)
    order, values = result.ranked()
    assert order.shape == (N_HEADS,)
    torch.testing.assert_close(values, result.per_head[order])
    assert torch.all(values.abs()[:-1] >= values.abs()[1:])


def test_attention_layer_before_the_source_is_rejected():
    model, cache = make_model(n_layers=4), make_cache(n_layers=4)
    with pytest.raises(ValueError, match="attention_layer must lie"):
        edge_loadings(model, cache, direction(), 0, 1, direction(), 2, 3, 1)


def test_attention_layer_after_the_target_is_rejected():
    model, cache = make_model(n_layers=4), make_cache(n_layers=4)
    with pytest.raises(ValueError, match="attention_layer must lie"):
        edge_loadings(model, cache, direction(), 0, 0, direction(), 2, 2, 3)


def test_a_source_outside_the_prompt_is_rejected():
    model, cache = make_model(), make_cache()
    with pytest.raises(IndexError, match="position"):
        edge_effect(model, cache, direction(), SEQ, 0, direction(), 0, 1)


def test_a_two_dimensional_source_is_rejected():
    model, cache = make_model(), make_cache()
    with pytest.raises(ValueError, match="must be 1D"):
        edge_effect(model, cache, torch.zeros(2, D_MODEL), 0, 0, direction(), 0, 1)


def test_splitting_at_each_layer_gives_the_same_total():
    """Each layer is its own partition of the same quantity, not a joint decomposition."""
    model, cache = make_model(n_layers=N_LAYERS), make_cache()
    source, reader = direction(10), direction(11)
    totals = [
        edge_loadings(model, cache, source, 0, 0, reader, 3, 1, layer).total for layer in (1,)
    ]
    expected = edge_effect(model, cache, source, 0, 0, reader, 3, 1)
    for total in totals:
        torch.testing.assert_close(total, expected, rtol=1e-4, atol=1e-6)
