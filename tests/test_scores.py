"""Tests for exact attention-score reconstruction.

The stub in :mod:`tests.stubs` computes reference scores the long way -- project, normalise,
rotate, dot -- so agreement here means the bilinear rearrangement is right, not merely
self-consistent. The corresponding check against real Qwen3 weights is in
:mod:`tests.test_scores_gpu`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from qk_attribution.circuits import UnsupportedArchitecture
from qk_attribution.scores import (
    attention_input,
    attention_scores,
    key_projection,
    layernorm_scale,
    qk_norm_scales,
    query_projection,
    relative_rotation,
    require_supported,
    rotation_matrices,
    to_head_space,
)
from tests.stubs import D_MODEL, N_HEADS, make_model, reference_scores

SEQ = 7


def rotary_model(**kwargs) -> SimpleNamespace:
    """A stub using rotary position embeddings."""
    return make_model(positional_embedding_type="rotary", **kwargs)


def residual(seq: int = SEQ, d_model: int = D_MODEL, seed: int = 11) -> torch.Tensor:
    return torch.randn(seq, d_model, generator=torch.Generator().manual_seed(seed))


def scales(model, layer: int, head: int, resid: torch.Tensor):
    """Compute the QK-norm scales the model itself would produce for this residual."""
    attn = model.blocks[layer].attn
    eps = model.cfg.eps
    q = resid @ attn.W_Q[head] + attn.b_Q[head]
    k = resid @ attn.W_K[head] + attn.b_K[head]
    return (
        (q.pow(2).mean(-1, keepdim=True) + eps).sqrt(),
        (k.pow(2).mean(-1, keepdim=True) + eps).sqrt(),
    )


def test_plain_model_reconstructs_reference_scores():
    model = make_model()
    resid = residual()
    for head in range(N_HEADS):
        torch.testing.assert_close(
            attention_scores(model, 1, head, resid), reference_scores(model, 1, head, resid)
        )


def test_rotary_model_reconstructs_reference_scores():
    model = rotary_model()
    resid = residual()
    torch.testing.assert_close(
        attention_scores(model, 0, 2, resid), reference_scores(model, 0, 2, resid)
    )


def test_qk_norm_model_reconstructs_reference_scores():
    model = make_model(use_qk_norm=True)
    resid = residual()
    s_q, s_k = scales(model, 0, 3, resid)
    got = attention_scores(model, 0, 3, resid, query_scale=s_q, key_scale=s_k)
    torch.testing.assert_close(got, reference_scores(model, 0, 3, resid))


def test_every_feature_at_once_reconstructs_reference_scores():
    """Rotary, QK-norm, grouped-query attention and projection biases together."""
    model = rotary_model(use_qk_norm=True, n_key_value_heads=2, with_bias=True)
    resid = residual()
    rotations = rotation_matrices(model, SEQ)
    for head in range(N_HEADS):
        s_q, s_k = scales(model, 1, head, resid)
        got = attention_scores(
            model, 1, head, resid, query_scale=s_q, key_scale=s_k, rotations=rotations
        )
        torch.testing.assert_close(got, reference_scores(model, 1, head, resid))


def test_rotations_reproduce_the_models_own_rotary():
    model = rotary_model()
    rotations = rotation_matrices(model, SEQ)
    size = rotations.shape[-1]
    vectors = torch.randn(1, SEQ, 1, size, generator=torch.Generator().manual_seed(5))
    expected = model.blocks[0].attn.apply_rotary(vectors)[0, :, 0]
    got = torch.einsum("pa,pab->pb", vectors[0, :, 0], rotations)
    torch.testing.assert_close(got, expected)


def test_rotations_are_orthogonal():
    rotations = rotation_matrices(rotary_model(), SEQ)
    identity = torch.eye(rotations.shape[-1]).expand_as(rotations)
    torch.testing.assert_close(
        rotations @ rotations.transpose(-1, -2), identity, atol=1e-6, rtol=1e-5
    )


def test_relative_rotation_depends_only_on_the_offset():
    """This is what lets the operator be cached by offset instead of by position pair."""
    rotations = rotation_matrices(rotary_model(), SEQ)
    for offset in range(1, 4):
        first = relative_rotation(rotations, offset, 0)
        for start in range(1, SEQ - offset):
            torch.testing.assert_close(
                relative_rotation(rotations, start + offset, start), first, atol=1e-6, rtol=1e-5
            )


def test_projections_fold_in_the_qk_norm_gain():
    model = make_model(use_qk_norm=True)
    attn = model.blocks[0].attn
    torch.testing.assert_close(query_projection(model, 0, 1), attn.W_Q[1] * attn.q_norm.w)
    torch.testing.assert_close(key_projection(model, 0, 1), attn.W_K[1] * attn.k_norm.w)


def test_projections_are_bare_weights_without_qk_norm():
    model = make_model()
    torch.testing.assert_close(query_projection(model, 0, 1), model.blocks[0].attn.W_Q[1])


def test_missing_qk_norm_scales_are_refused():
    model = make_model(use_qk_norm=True)
    with pytest.raises(ValueError, match="query_scale and key_scale are required"):
        attention_scores(model, 0, 0, residual())


def test_soft_capped_models_are_refused():
    model = make_model(attn_scores_soft_cap=50.0)
    with pytest.raises(UnsupportedArchitecture, match="soft-capped"):
        require_supported(model)
    with pytest.raises(UnsupportedArchitecture, match="soft-capped"):
        attention_scores(model, 0, 0, residual())


def test_rotations_need_a_rotary_model():
    with pytest.raises(UnsupportedArchitecture, match="rotary"):
        rotation_matrices(make_model(), SEQ)


@pytest.mark.parametrize("n_pos", [0, -1])
def test_non_positive_position_count_is_rejected(n_pos: int):
    with pytest.raises(ValueError, match="n_pos must be positive"):
        rotation_matrices(rotary_model(), n_pos)


def test_too_few_rotations_are_rejected():
    model = rotary_model()
    with pytest.raises(ValueError, match="need at least"):
        attention_scores(model, 0, 0, residual(), rotations=rotation_matrices(model, SEQ - 1))


def test_mismatched_scale_length_is_rejected():
    model = make_model(use_qk_norm=True)
    resid = residual()
    s_q, s_k = scales(model, 0, 0, resid)
    with pytest.raises(ValueError, match="query_scale covers"):
        attention_scores(model, 0, 0, resid, query_scale=s_q[:-1], key_scale=s_k)


def test_batched_residual_is_rejected():
    with pytest.raises(ValueError, match="must be 2D"):
        attention_scores(make_model(), 0, 0, residual().unsqueeze(0))


@pytest.mark.parametrize("head", [-1, N_HEADS])
def test_head_out_of_range_raises(head: int):
    with pytest.raises(IndexError, match="head"):
        query_projection(make_model(), 0, head)


def test_attention_input_applies_the_layernorm_gain():
    """The hook fires before the gain, which is the easiest thing in this module to get wrong."""
    model = make_model()
    model.blocks[0].ln1.w = torch.arange(1.0, D_MODEL + 1)
    normalized = residual()
    cache = {"blocks.0.ln1.hook_normalized": normalized}
    torch.testing.assert_close(
        attention_input(model, cache, 0), normalized * torch.arange(1.0, D_MODEL + 1)
    )


def test_attention_input_drops_the_batch_axis():
    model = make_model()
    normalized = residual()
    cache = {"blocks.0.ln1.hook_normalized": normalized.unsqueeze(0)}
    torch.testing.assert_close(attention_input(model, cache, 0), normalized * model.blocks[0].ln1.w)


def test_qk_norm_scales_restore_the_head_axis():
    model = make_model(use_qk_norm=True, n_key_value_heads=2)
    flat_q = torch.arange(float(SEQ * N_HEADS)).reshape(-1, 1)
    flat_k = torch.arange(float(SEQ * 2)).reshape(-1, 1)
    cache = {
        "blocks.0.attn.q_norm.hook_scale": flat_q,
        "blocks.0.attn.k_norm.hook_scale": flat_k,
    }
    result = qk_norm_scales(model, cache, 0)
    assert result is not None
    got_q, got_k = result
    assert got_q.shape == (SEQ, N_HEADS, 1)
    assert got_k.shape == (SEQ, 2, 1)
    # Flattening is over (batch, pos, head), so position 1 head 0 follows position 0 head 3.
    assert got_q[1, 0].item() == pytest.approx(float(N_HEADS))


def test_qk_norm_scales_are_none_without_qk_norm():
    assert qk_norm_scales(make_model(), {}, 0) is None


def test_layernorm_scale_drops_the_batch_axis():
    cache = {"blocks.0.ln1.hook_scale": torch.ones(1, SEQ, 1) * 3.0}
    torch.testing.assert_close(layernorm_scale(cache, 0), torch.ones(SEQ, 1) * 3.0)


def test_to_head_space_matches_projecting_the_residual_directly():
    """A set of directions, one per position, must project the way the residual does."""
    model = rotary_model(use_qk_norm=True)
    resid = residual()
    scale_q, _ = scales(model, 0, 2, resid)
    rotations = rotation_matrices(model, SEQ)
    got = to_head_space(
        model,
        0,
        2,
        resid,
        torch.arange(SEQ),
        side="query",
        rotations=rotations,
        norm_scale=torch.ones(SEQ, 1),
        qk_scale=scale_q,
    )
    gain = model.blocks[0].ln1.w
    expected = torch.einsum(
        "pa,pab->pb", ((resid * gain) @ query_projection(model, 0, 2)) / scale_q, rotations
    )
    torch.testing.assert_close(got, expected)


def test_to_head_space_divides_by_the_layernorm_scale():
    model = make_model()
    resid = residual()
    norm = torch.arange(1.0, SEQ + 1).unsqueeze(-1)
    got = to_head_space(
        model,
        0,
        1,
        resid,
        torch.arange(SEQ),
        side="key",
        rotations=None,
        norm_scale=norm,
        qk_scale=None,
    )
    gain = model.blocks[0].ln1.w
    torch.testing.assert_close(got, (resid * gain / norm) @ key_projection(model, 0, 1))


def test_to_head_space_uses_the_key_projection_for_the_key_side():
    model = make_model()
    resid = residual()
    got = to_head_space(
        model,
        0,
        1,
        resid,
        torch.arange(SEQ),
        side="key",
        rotations=None,
        norm_scale=torch.ones(SEQ, 1),
        qk_scale=None,
    )
    gain = model.blocks[0].ln1.w
    torch.testing.assert_close(got, (resid * gain) @ key_projection(model, 0, 1))


def test_to_head_space_reads_the_rotation_of_each_rows_own_position():
    """Two rows at different positions must be rotated differently."""
    model = rotary_model()
    rotations = rotation_matrices(model, SEQ)
    direction = torch.ones(1, D_MODEL)
    first = to_head_space(
        model,
        0,
        0,
        direction,
        torch.tensor([0]),
        side="query",
        rotations=rotations,
        norm_scale=torch.ones(SEQ, 1),
        qk_scale=None,
    )
    later = to_head_space(
        model,
        0,
        0,
        direction,
        torch.tensor([3]),
        side="query",
        rotations=rotations,
        norm_scale=torch.ones(SEQ, 1),
        qk_scale=None,
    )
    assert not torch.allclose(first, later)


def test_to_head_space_rejects_an_unknown_side():
    with pytest.raises(ValueError, match="side must be"):
        to_head_space(
            make_model(),
            0,
            0,
            residual(),
            torch.arange(SEQ),
            side="value",
            rotations=None,
            norm_scale=torch.ones(SEQ, 1),
            qk_scale=None,
        )


def test_to_head_space_rejects_mismatched_position_count():
    with pytest.raises(ValueError, match="positions for"):
        to_head_space(
            make_model(),
            0,
            0,
            residual(),
            torch.arange(SEQ - 1),
            side="query",
            rotations=None,
            norm_scale=torch.ones(SEQ, 1),
            qk_scale=None,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scores_follow_the_residuals_dtype(dtype: torch.dtype):
    """Graphs are built in bfloat16, so a low-precision residual must not meet float32 weights."""
    model = rotary_model(use_qk_norm=True)
    resid = residual().to(dtype)
    scale_q, scale_k = scales(model, 0, 1, residual())
    got = attention_scores(
        model,
        0,
        1,
        resid,
        query_scale=scale_q.to(dtype),
        key_scale=scale_k.to(dtype),
        rotations=rotation_matrices(model, SEQ).to(dtype),
    )
    assert got.dtype == dtype


def test_head_space_follows_the_directions_dtype():
    model = make_model()
    got = to_head_space(
        model,
        0,
        0,
        residual().to(torch.bfloat16),
        torch.arange(SEQ),
        side="query",
        rotations=None,
        norm_scale=torch.ones(SEQ, 1),
        qk_scale=None,
    )
    assert got.dtype == torch.bfloat16
