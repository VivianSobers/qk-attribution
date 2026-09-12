"""Tests for per-head QK and OV circuit construction.

These use a stub standing in for a HookedTransformer, so they need no GPU and no weights. Checks
against a real model's forward pass live in the GPU-marked tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from qk_attribution.circuits import (
    UnsupportedArchitecture,
    architecture_notes,
    attention_scale,
    attention_scores_from_residual,
    d_head,
    effective_rank,
    head_output_from_residual,
    kv_head_for,
    n_heads,
    n_kv_heads,
    n_layers,
    ov_matrix,
    qk_low_rank_spectrum,
    qk_matrix,
    require_plain_qk,
)
from tests.stubs import D_HEAD, D_MODEL, N_HEADS, N_LAYERS, make_model


def test_dimensions_read_from_weights():
    model = make_model()
    assert n_layers(model) == N_LAYERS
    assert n_heads(model) == N_HEADS
    assert d_head(model) == D_HEAD
    assert attention_scale(model) == pytest.approx(D_HEAD**0.5)


def test_qk_matrix_equals_manual_product():
    model = make_model()
    attn = model.blocks[1].attn
    expected = attn.W_Q[2] @ attn.W_K[2].T
    torch.testing.assert_close(qk_matrix(model, 1, 2), expected)
    assert qk_matrix(model, 1, 2).shape == (D_MODEL, D_MODEL)


def test_qk_matrix_scaling_divides_by_attention_scale():
    model = make_model(attn_scale=4.0)
    torch.testing.assert_close(qk_matrix(model, 0, 0, scaled=True), qk_matrix(model, 0, 0) / 4.0)


def test_ov_matrix_equals_manual_product():
    model = make_model()
    attn = model.blocks[0].attn
    expected = attn.W_V[3] @ attn.W_O[3]
    torch.testing.assert_close(ov_matrix(model, 0, 3), expected)
    assert ov_matrix(model, 0, 3).shape == (D_MODEL, D_MODEL)


def test_scores_from_residual_match_explicit_qk_path():
    """The bilinear form must agree with projecting q and k separately."""
    model = make_model()
    resid = torch.randn(5, D_MODEL, generator=torch.Generator().manual_seed(1))
    attn = model.blocks[0].attn
    q = resid @ attn.W_Q[1]
    k = resid @ attn.W_K[1]
    expected = (q @ k.T) / attention_scale(model)
    got = attention_scores_from_residual(model, 0, 1, resid)
    torch.testing.assert_close(got, expected)


def test_head_output_matches_explicit_value_path():
    model = make_model()
    gen = torch.Generator().manual_seed(2)
    resid = torch.randn(4, D_MODEL, generator=gen)
    pattern = torch.softmax(torch.randn(4, 4, generator=gen), dim=-1)
    attn = model.blocks[1].attn
    v = resid @ attn.W_V[0]
    expected = (pattern @ v) @ attn.W_O[0]
    torch.testing.assert_close(head_output_from_residual(model, 1, 0, resid, pattern), expected)


def test_batched_residual_is_supported():
    model = make_model()
    resid = torch.randn(2, 5, D_MODEL, generator=torch.Generator().manual_seed(3))
    scores = attention_scores_from_residual(model, 0, 0, resid)
    assert scores.shape == (2, 5, 5)
    torch.testing.assert_close(scores[0], attention_scores_from_residual(model, 0, 0, resid[0]))


def test_plain_architecture_is_accepted():
    model = make_model()
    notes = architecture_notes(model)
    assert notes == {
        "rotary": False,
        "positional_embedding_type": "standard",
        "rotary_dim": None,
        "qk_norm": False,
        "score_soft_cap": None,
        "gqa": False,
    }
    require_plain_qk(model)  # must not raise


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"positional_embedding_type": "rotary"}, "rotary position embeddings"),
        ({"use_qk_norm": True}, "query/key normalisation"),
        ({"attn_scores_soft_cap": 50.0}, "soft-capped"),
    ],
)
def test_unsupported_features_are_detected(kwargs: dict, fragment: str):
    model = make_model(**kwargs)
    with pytest.raises(UnsupportedArchitecture, match=fragment):
        require_plain_qk(model)
    with pytest.raises(UnsupportedArchitecture):
        attention_scores_from_residual(model, 0, 0, torch.randn(3, D_MODEL))


def test_all_unsupported_features_are_reported_together():
    model = make_model(
        positional_embedding_type="rotary", use_qk_norm=True, attn_scores_soft_cap=30.0
    )
    with pytest.raises(UnsupportedArchitecture) as excinfo:
        require_plain_qk(model)
    message = str(excinfo.value)
    assert "rotary" in message and "normalisation" in message and "soft-capped" in message


def test_architecture_check_can_be_bypassed_for_measurement():
    """Measuring the size of the discrepancy requires computing it anyway."""
    model = make_model(positional_embedding_type="rotary")
    scores = attention_scores_from_residual(
        model, 0, 0, torch.randn(3, D_MODEL), check_architecture=False
    )
    assert scores.shape == (3, 3)


def test_non_positive_soft_cap_means_disabled():
    for value in (-1.0, 0.0):
        assert architecture_notes(make_model(attn_scores_soft_cap=value))["score_soft_cap"] is None


def test_gqa_head_mapping_groups_query_heads():
    model = make_model(n_key_value_heads=2)
    assert n_kv_heads(model) == 2
    assert architecture_notes(model)["gqa"] is True
    # Four query heads over two kv groups: heads 0,1 -> group 0 and heads 2,3 -> group 1.
    assert [kv_head_for(model, h) for h in range(N_HEADS)] == [0, 0, 1, 1]


def test_without_gqa_head_mapping_is_identity():
    model = make_model()
    assert [kv_head_for(model, h) for h in range(N_HEADS)] == list(range(N_HEADS))


def test_indivisible_head_grouping_is_rejected():
    model = make_model(n_key_value_heads=3)
    with pytest.raises(ValueError, match="not a multiple"):
        kv_head_for(model, 0)


@pytest.mark.parametrize("layer", [-1, N_LAYERS, 99])
def test_layer_out_of_range_raises(layer: int):
    with pytest.raises(IndexError, match="layer"):
        qk_matrix(make_model(), layer, 0)


@pytest.mark.parametrize("head", [-1, N_HEADS, 99])
def test_head_out_of_range_raises(head: int):
    model = make_model()
    with pytest.raises(IndexError, match="head"):
        qk_matrix(model, 0, head)
    with pytest.raises(IndexError, match="head"):
        ov_matrix(model, 0, head)


def test_malformed_residual_shapes_are_rejected():
    model = make_model()
    with pytest.raises(ValueError, match="2D or 3D"):
        attention_scores_from_residual(model, 0, 0, torch.randn(D_MODEL))
    with pytest.raises(ValueError, match="must be 2D"):
        head_output_from_residual(model, 0, 0, torch.randn(2, 3, D_MODEL), torch.eye(3))
    with pytest.raises(ValueError, match="inconsistent"):
        head_output_from_residual(model, 0, 0, torch.randn(4, D_MODEL), torch.eye(3))


def test_model_without_blocks_is_rejected():
    with pytest.raises(AttributeError, match="no .blocks"):
        qk_matrix(SimpleNamespace(), 0, 0)


def test_spectrum_is_descending_and_full_length():
    spectrum = qk_low_rank_spectrum(make_model(), 0, 0)
    assert spectrum.shape == (D_MODEL,)
    assert torch.all(spectrum[:-1] >= spectrum[1:])


def test_effective_rank_of_identity_needs_almost_every_component():
    # A flat spectrum has no low-rank structure to exploit.
    assert effective_rank(torch.ones(10), energy=0.99) == 10


def test_effective_rank_of_rank_one_spectrum_is_one():
    assert effective_rank(torch.tensor([10.0, 1e-8, 1e-9]), energy=0.99) == 1


def test_effective_rank_grows_with_required_energy():
    spectrum = torch.tensor([4.0, 3.0, 2.0, 1.0])
    ranks = [effective_rank(spectrum, energy=e) for e in (0.5, 0.9, 1.0)]
    assert ranks == sorted(ranks)
    assert ranks[-1] == 4


@pytest.mark.parametrize("energy", [0.0, -0.1, 1.5])
def test_invalid_energy_is_rejected(energy: float):
    with pytest.raises(ValueError, match="energy"):
        effective_rank(torch.ones(3), energy=energy)


def test_empty_spectrum_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        effective_rank(torch.tensor([]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_effective_rank_accepts_any_float_dtype(dtype: torch.dtype):
    assert effective_rank(torch.tensor([10.0, 1e-3, 1e-4], dtype=dtype), energy=0.99) == 1


@pytest.mark.gpu
def test_effective_rank_accepts_cuda_tensors():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    spectrum = torch.tensor([4.0, 3.0, 2.0, 1.0], device="cuda")
    assert effective_rank(spectrum, energy=0.9) == effective_rank(spectrum.cpu(), energy=0.9)


def test_attention_scale_is_one_when_the_model_disables_scaling():
    """A missed flag here scales every score by sqrt(d_head) and nothing else fails."""
    model = make_model()
    model.cfg.use_attn_scale = False
    assert attention_scale(model) == 1.0


def test_attention_scale_is_unchanged_when_scaling_is_enabled():
    model = make_model(attn_scale=4.0)
    model.cfg.use_attn_scale = True
    assert attention_scale(model) == pytest.approx(4.0)


@pytest.mark.parametrize("size", [1, 5, 64])
def test_effective_rank_never_exceeds_the_component_count(size: int):
    spectrum = torch.linspace(1.0, 0.01, size)
    assert effective_rank(spectrum, energy=1.0) <= size


def test_effective_rank_of_a_flat_spectrum_at_full_energy():
    assert effective_rank(torch.ones(64), energy=1.0) == 64


@pytest.mark.parametrize("scheme", ["alibi", "shortformer", "relative_positional_bias"])
def test_unknown_position_schemes_are_refused(scheme: str):
    """ALiBi and friends add a term outside the bilinear form and used to pass the guard."""
    model = make_model(positional_embedding_type=scheme)
    with pytest.raises(UnsupportedArchitecture, match="positional embedding type"):
        require_plain_qk(model)


def test_architecture_notes_report_the_position_scheme():
    assert architecture_notes(make_model())["positional_embedding_type"] == "standard"
