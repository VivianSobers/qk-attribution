"""Tests for the source-pair decomposition of attention scores.

The property that matters is that the parts sum to the whole. Bilinearity guarantees it in theory,
so these tests check that the implementation actually inherits it: a decomposition whose terms do
not add up would still produce plausible per-pair numbers and would be hard to notice later.
"""

from __future__ import annotations

import pytest
import torch

from qk_attribution.attribution import QKAttribution, qk_attribution, residual_remainder
from qk_attribution.circuits import UnsupportedArchitecture
from qk_attribution.features import SourceSet
from qk_attribution.scores import attention_scores, rotation_matrices
from tests.stubs import D_MODEL, make_model

SEQ = 6
QUERY_POSITION = 2


def rotary_model(**kwargs):
    return make_model(positional_embedding_type="rotary", **kwargs)


def make_sources(
    positions: list[int], seed: int = 7, scale: float = 1.0, zero_first: bool = False
) -> SourceSet:
    """Build a source set with one direction per entry in ``positions``."""
    gen = torch.Generator().manual_seed(seed)
    directions = torch.randn(len(positions), D_MODEL, generator=gen) * scale
    if zero_first and len(positions):
        directions[0] = 0.0
    return SourceSet(
        directions=directions,
        positions=torch.tensor(positions, dtype=torch.int64),
        layers=torch.zeros(len(positions), dtype=torch.int64),
        feature_ids=torch.arange(len(positions)),
        activations=torch.ones(len(positions)),
    )


def residual_from(*source_sets: SourceSet) -> torch.Tensor:
    """Sum the sources back into a residual stream, which is what they decompose."""
    residual = torch.zeros(SEQ, D_MODEL)
    for sources in source_sets:
        residual.index_add_(0, sources.positions, sources.directions)
    return residual


def test_contributions_sum_to_the_score_of_the_residual_they_build():
    """Bilinearity is the whole method: the parts must add up to the whole.

    The source set is the residual here. Query-side sources are the subset at the query position,
    key-side sources are all of them, which is the arrangement a real decomposition uses.
    """
    model = make_model()
    sources = make_sources([0, 1, QUERY_POSITION, QUERY_POSITION, 4], seed=1)
    result = qk_attribution(
        model,
        0,
        1,
        QUERY_POSITION,
        query_sources=sources.at_position(QUERY_POSITION),
        key_sources=sources,
        rotations=None,
        norm_scale=torch.ones(SEQ, 1),
        query_scale=None,
        key_scale=None,
    )
    gain = model.blocks[0].ln1.w
    direct = attention_scores(model, 0, 1, residual_from(sources) * gain)[QUERY_POSITION]
    torch.testing.assert_close(result.by_key_position(SEQ), direct, rtol=1e-4, atol=1e-5)


def test_contributions_sum_to_the_score_with_rotary_and_qk_norm():
    """The same, with every frozen scale and the rotation in play."""
    model = rotary_model(use_qk_norm=True, n_key_value_heads=2)
    sources = make_sources([0, 1, QUERY_POSITION, 3, QUERY_POSITION, 5], seed=3)
    scale_q = torch.rand(SEQ, 1) + 0.5
    scale_k = torch.rand(SEQ, 1) + 0.5
    norm = torch.rand(SEQ, 1) + 0.5
    rotations = rotation_matrices(model, SEQ)
    result = qk_attribution(
        model,
        1,
        3,
        QUERY_POSITION,
        query_sources=sources.at_position(QUERY_POSITION),
        key_sources=sources,
        rotations=rotations,
        norm_scale=norm,
        query_scale=scale_q,
        key_scale=scale_k,
    )
    direct = attention_scores(
        model,
        1,
        3,
        residual_from(sources) * model.blocks[1].ln1.w / norm,
        query_scale=scale_q,
        key_scale=scale_k,
        rotations=rotations,
    )[QUERY_POSITION]
    torch.testing.assert_close(result.by_key_position(SEQ), direct, rtol=1e-4, atol=1e-5)


def test_contribution_matrix_has_one_entry_per_source_pair():
    model = make_model()
    query_sources = make_sources([QUERY_POSITION] * 3)
    key_sources = make_sources([0, 1, 4, 5], seed=9)
    result = qk_attribution(
        model,
        0,
        0,
        QUERY_POSITION,
        query_sources=query_sources,
        key_sources=key_sources,
        rotations=None,
        norm_scale=torch.ones(SEQ, 1),
        query_scale=None,
        key_scale=None,
    )
    assert result.contributions.shape == (3, 4)
    torch.testing.assert_close(result.by_key_source, result.contributions.sum(dim=0))
    torch.testing.assert_close(result.by_query_source, result.contributions.sum(dim=1))


def test_a_zero_direction_contributes_nothing():
    model = make_model()
    query_sources = make_sources([QUERY_POSITION] * 2, zero_first=True)
    key_sources = make_sources([0, 3], seed=5)
    result = qk_attribution(
        model,
        0,
        0,
        QUERY_POSITION,
        query_sources=query_sources,
        key_sources=key_sources,
        rotations=None,
        norm_scale=torch.ones(SEQ, 1),
        query_scale=None,
        key_scale=None,
    )
    torch.testing.assert_close(result.contributions[0], torch.zeros(2))


def test_empty_sources_give_a_zero_score():
    model = make_model()
    empty = make_sources([])
    result = qk_attribution(
        model,
        0,
        0,
        QUERY_POSITION,
        query_sources=empty,
        key_sources=make_sources([0, 1]),
        rotations=None,
        norm_scale=torch.ones(SEQ, 1),
        query_scale=None,
        key_scale=None,
    )
    assert result.contributions.shape == (0, 2)
    torch.testing.assert_close(result.by_key_position(SEQ), torch.zeros(SEQ))


def test_query_sources_at_other_positions_are_rejected():
    model = make_model()
    with pytest.raises(ValueError, match="query_position"):
        qk_attribution(
            model,
            0,
            0,
            QUERY_POSITION,
            query_sources=make_sources([QUERY_POSITION, 4]),
            key_sources=make_sources([0]),
            rotations=None,
            norm_scale=torch.ones(SEQ, 1),
            query_scale=None,
            key_scale=None,
        )


def test_top_pairs_returns_the_largest_magnitudes_with_their_signs():
    model = make_model()
    result = qk_attribution(
        model,
        0,
        0,
        QUERY_POSITION,
        query_sources=make_sources([QUERY_POSITION] * 3, seed=11),
        key_sources=make_sources([0, 1, 2, 3], seed=12),
        rotations=None,
        norm_scale=torch.ones(SEQ, 1),
        query_scale=None,
        key_scale=None,
    )
    values, indices = result.top_pairs(3)
    assert values.shape == (3,)
    assert values.abs()[0] >= values.abs()[1] >= values.abs()[2]
    row, column = divmod(int(indices[0]), result.contributions.shape[1])
    torch.testing.assert_close(result.contributions[row, column], values[0])


def test_top_pairs_caps_at_the_available_count():
    model = make_model()
    result = qk_attribution(
        model,
        0,
        0,
        QUERY_POSITION,
        query_sources=make_sources([QUERY_POSITION]),
        key_sources=make_sources([0, 1]),
        rotations=None,
        norm_scale=torch.ones(SEQ, 1),
        query_scale=None,
        key_scale=None,
    )
    values, _ = result.top_pairs(99)
    assert values.shape == (2,)


def test_remainder_is_zero_when_the_sources_rebuild_the_residual():
    sources = make_sources([0, 1, 1, 3], seed=13)
    residual = residual_from(sources)
    torch.testing.assert_close(
        residual_remainder(residual, sources), torch.zeros(SEQ, D_MODEL), atol=1e-6, rtol=1e-5
    )


def test_remainder_keeps_what_the_sources_do_not_cover():
    sources = make_sources([0, 2], seed=14)
    extra = torch.zeros(SEQ, D_MODEL)
    extra[4] = 1.0
    residual = residual_from(sources) + extra
    torch.testing.assert_close(residual_remainder(residual, sources), extra, atol=1e-6, rtol=1e-5)


def test_remainder_rejects_a_batched_residual():
    with pytest.raises(ValueError, match="must be 2D"):
        residual_remainder(torch.zeros(1, SEQ, D_MODEL), make_sources([0]))


def test_a_non_zero_attention_bias_is_refused():
    """The bias pairs with every source and with itself, and the decomposition omits those terms."""
    model = make_model(with_bias=True)
    sources = make_sources([0, QUERY_POSITION])
    with pytest.raises(UnsupportedArchitecture, match="b_Q"):
        qk_attribution(
            model,
            0,
            0,
            QUERY_POSITION,
            query_sources=sources.at_position(QUERY_POSITION),
            key_sources=sources,
            rotations=None,
            norm_scale=torch.ones(SEQ, 1),
            query_scale=None,
            key_scale=None,
        )


def test_key_position_totals_are_accumulated_in_float32():
    """Many terms of both signs land on one position; a bfloat16 accumulator loses the result."""
    model = make_model()
    sources = make_sources([0, 1, QUERY_POSITION, 2], seed=31)
    result = qk_attribution(
        model,
        0,
        0,
        QUERY_POSITION,
        query_sources=sources.at_position(QUERY_POSITION),
        key_sources=sources,
        rotations=None,
        norm_scale=torch.ones(SEQ, 1),
        query_scale=None,
        key_scale=None,
    )
    exact = result.by_key_position(SEQ)
    low = QKAttribution(
        contributions=result.contributions.to(torch.bfloat16),
        query_sources=result.query_sources,
        key_sources=result.key_sources,
        query_position=result.query_position,
    ).by_key_position(SEQ)
    assert low.dtype == torch.bfloat16
    torch.testing.assert_close(low.float(), exact, rtol=0.05, atol=0.05)
