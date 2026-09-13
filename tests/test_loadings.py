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


# path_loadings computes every attention layer's split in one forward and one backward sweep. It is
# only worth having if it reproduces edge_loadings exactly, so most of these compare the two.

from qk_attribution.loadings import EdgeLoadings, path_loadings  # noqa: E402

LONG = 5


def test_path_loadings_match_splitting_one_layer_at_a_time():
    model, cache = make_model(n_layers=LONG), make_cache(n_layers=LONG)
    source, reader = direction(12), direction(13)
    result = path_loadings(model, cache, source, 1, 0, reader, 4, LONG - 1)
    for layer in range(1, LONG):
        expected = edge_loadings(model, cache, source, 1, 0, reader, 4, LONG - 1, layer)
        got = result.at(layer)
        torch.testing.assert_close(got.per_head, expected.per_head, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(got.bypass, expected.bypass, rtol=1e-4, atol=1e-6)


def test_path_loadings_total_is_the_edge_effect():
    model, cache = make_model(n_layers=LONG), make_cache(n_layers=LONG)
    source, reader = direction(14), direction(15)
    result = path_loadings(model, cache, source, 2, 0, reader, 3, LONG - 1)
    expected = edge_effect(model, cache, source, 2, 0, reader, 3, LONG - 1)
    torch.testing.assert_close(result.total, expected, rtol=1e-4, atol=1e-6)


def test_every_layer_on_the_path_partitions_the_same_total():
    model, cache = make_model(n_layers=LONG), make_cache(n_layers=LONG)
    result = path_loadings(model, cache, direction(16), 0, 0, direction(17), 4, LONG - 1)
    sums = result.per_head.sum(dim=1) + result.bypass
    torch.testing.assert_close(sums, result.total.expand_as(sums), rtol=1e-4, atol=1e-6)


def test_path_loadings_cover_every_layer_between_source_and_target():
    model, cache = make_model(n_layers=LONG), make_cache(n_layers=LONG)
    result = path_loadings(model, cache, direction(), 0, 1, direction(), 4, LONG - 1)
    assert result.layers == [2, 3, 4]
    assert result.per_head.shape == (3, N_HEADS)
    assert result.bypass.shape == (3,)


def test_adjacent_blocks_have_a_single_split_point():
    model, cache = make_model(n_layers=LONG), make_cache(n_layers=LONG)
    result = path_loadings(model, cache, direction(18), 0, 2, direction(19), 3, 3)
    expected = edge_loadings(model, cache, direction(18), 0, 2, direction(19), 3, 3, 3)
    assert result.layers == [3]
    torch.testing.assert_close(result.at(3).per_head, expected.per_head, rtol=1e-4, atol=1e-6)


def test_a_token_embedding_splits_from_block_zero():
    """A token is written before block 0, so block 0's own attention is on its path."""
    model, cache = make_model(n_layers=LONG), make_cache(n_layers=LONG)
    source, reader = direction(20), direction(21)
    result = path_loadings(model, cache, source, 1, -1, reader, 4, 2)
    assert result.layers == [0, 1, 2]
    expected = edge_loadings(model, cache, source, 1, -1, reader, 4, 2, 0)
    torch.testing.assert_close(result.at(0).per_head, expected.per_head, rtol=1e-4, atol=1e-6)


def test_at_returns_an_edge_loadings_for_that_layer():
    model, cache = make_model(n_layers=LONG), make_cache(n_layers=LONG)
    got = path_loadings(model, cache, direction(), 0, 0, direction(22), 3, 2).at(2)
    assert isinstance(got, EdgeLoadings)
    assert got.attention_layer == 2


@pytest.mark.parametrize("layer", [0, 3, -1])
def test_at_rejects_a_layer_off_the_path(layer: int):
    model, cache = make_model(n_layers=LONG), make_cache(n_layers=LONG)
    result = path_loadings(model, cache, direction(), 0, 0, direction(), 3, 2)
    with pytest.raises(ValueError, match="not on this path"):
        result.at(layer)


def test_path_loadings_are_linear_in_the_source():
    model, cache = make_model(n_layers=LONG), make_cache(n_layers=LONG)
    reader = direction(24)
    single = path_loadings(model, cache, direction(25), 0, 0, reader, 3, LONG - 1)
    scaled = path_loadings(model, cache, direction(25) * -3.0, 0, 0, reader, 3, LONG - 1)
    torch.testing.assert_close(scaled.per_head, single.per_head * -3.0, rtol=1e-4, atol=1e-6)


def test_path_loadings_work_with_gradients_disabled_and_leave_them_disabled():
    """The measurement scripts turn autograd off globally, and this must not turn it back on."""
    model, cache = make_model(n_layers=LONG), make_cache(n_layers=LONG)
    with torch.no_grad():
        result = path_loadings(model, cache, direction(26), 0, 0, direction(27), 3, LONG - 1)
        assert not torch.is_grad_enabled()
    assert not result.per_head.requires_grad
    assert not result.total.requires_grad


def test_a_target_not_after_the_source_is_rejected():
    model, cache = make_model(n_layers=LONG), make_cache(n_layers=LONG)
    with pytest.raises(ValueError, match="target_layer must come after"):
        path_loadings(model, cache, direction(), 0, 2, direction(), 3, 2)


def test_path_loadings_agree_with_the_per_layer_loop_on_a_deep_path():
    """Short paths hid nothing, but a long one is where a missed sensitivity would compound.

    The stub's weights are unnormalised, so values grow several-fold per layer and reach ~1e20 at
    this depth; the comparison is relative, and in float64 so rounding cannot mask a real gap.
    """
    depth = 28
    model, cache = make_model(n_layers=depth), make_cache(n_layers=depth)
    source, reader = direction(28).double(), direction(29).double()
    result = path_loadings(model, cache, source, 1, 0, reader, 4, depth - 1)
    for layer in range(1, depth):
        expected = edge_loadings(model, cache, source, 1, 0, reader, 4, depth - 1, layer)
        reference = torch.cat([expected.per_head, expected.bypass.reshape(1)])
        got = torch.cat([result.at(layer).per_head, result.at(layer).bypass.reshape(1)])
        relative = (got - reference).abs().max() / reference.abs().max()
        assert relative < 1e-12, f"layer {layer}: relative difference {relative:.2e}"
