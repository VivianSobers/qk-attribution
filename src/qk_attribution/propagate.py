"""Propagating a residual-stream perturbation forward through frozen attention.

Attribution graphs are built with attention patterns and normalisation scales held fixed. Under
those constraints the map from a perturbation of the residual stream at one layer to its effect at
a later layer is linear, and it runs entirely through attention: with transcoder activations frozen
an MLP's output does not move when its input does, so it passes nothing along.

That linearity is what makes head loadings possible. At any attention layer the incoming residual
splits into a part each head moves and a part that bypasses attention entirely, and those parts
propagate forward independently. Summing them recovers the whole, which is the property the tests
check.

Everything here works on deltas rather than absolute residuals, so the frozen scales enter as
divisors and never as offsets.
"""

from __future__ import annotations

import torch
from torch import Tensor

from qk_attribution.circuits import UnsupportedArchitecture, attention_block, n_heads, n_layers


def frozen_patterns(cache: object, layer: int) -> Tensor:
    """Return the attention probabilities at ``layer``, shaped ``(n_heads, query, key)``."""
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]  # type: ignore[index]
    return pattern[0] if pattern.ndim == 4 else pattern


def _normalised(model: object, layer: int, delta: Tensor, scale: Tensor) -> Tensor:
    """Apply ``ln1`` to a perturbation, with its scale frozen."""
    ln1 = model.blocks[layer].ln1  # type: ignore[attr-defined]
    if getattr(ln1, "b", None) is not None:
        raise UnsupportedArchitecture(
            "ln1 has a bias, which does not act on a perturbation; RMSNorm is assumed here"
        )
    return delta * ln1.w.to(delta.dtype) / scale.to(delta.dtype)


def attention_step(
    model: object,
    layer: int,
    delta: Tensor,
    patterns: Tensor,
    scale: Tensor,
    *,
    per_head: bool = False,
) -> Tensor:
    """Return what attention at ``layer`` writes back given a perturbation of its input.

    Args:
        model: A HookedTransformer-backed model.
        layer: Block index.
        delta: Perturbation of the residual entering the block, shaped ``(n_pos, d_model)``.
        patterns: Frozen probabilities from :func:`frozen_patterns`.
        scale: Frozen ``ln1`` scale, shaped ``(n_pos, 1)``.
        per_head: Return one slice per head instead of their sum.

    Returns:
        ``(n_pos, d_model)``, or ``(n_heads, n_pos, d_model)`` when ``per_head``.
    """
    if delta.ndim != 2:
        raise ValueError(f"delta must be 2D (n_pos, d_model), got {tuple(delta.shape)}")
    heads = n_heads(model)
    if patterns.shape[0] != heads:
        raise ValueError(f"patterns cover {patterns.shape[0]} heads, expected {heads}")
    attn = attention_block(model, layer)
    normalised = _normalised(model, layer, delta, scale)
    # W_V and W_O are materialised per query head, so no grouped-query mapping is needed here.
    weights_v = attn.W_V.to(normalised.dtype)  # type: ignore[attr-defined]
    values = torch.einsum("pd,hde->hpe", normalised, weights_v)
    moved = torch.einsum("hqp,hpe->hqe", patterns.to(values.dtype), values)
    weights_o = attn.W_O.to(moved.dtype)  # type: ignore[attr-defined]
    written = torch.einsum("hqe,hed->hqd", moved, weights_o)
    return written if per_head else written.sum(dim=0)


def propagate(
    model: object, cache: object, delta: Tensor, from_layer: int, to_layer: int
) -> Tensor:
    """Carry a perturbation from the input of ``from_layer`` to the input of ``to_layer``.

    Args:
        delta: Perturbation of ``blocks.{from_layer}.hook_resid_pre``, shaped ``(n_pos, d_model)``.
        from_layer: Where the perturbation enters.
        to_layer: Where to read it out. Must not precede ``from_layer``.

    Returns:
        Tensor of shape ``(n_pos, d_model)``, the resulting perturbation of
        ``blocks.{to_layer}.hook_resid_pre``.
    """
    total = n_layers(model)
    if not 0 <= from_layer <= to_layer <= total:
        raise ValueError(
            f"need 0 <= from_layer <= to_layer <= {total}, got {from_layer} and {to_layer}"
        )
    state = delta
    for layer in range(from_layer, to_layer):
        scale = cache[f"blocks.{layer}.ln1.hook_scale"]  # type: ignore[index]
        scale = scale[0] if scale.ndim == 3 else scale
        state = state + attention_step(model, layer, state, frozen_patterns(cache, layer), scale)
    return state


def split_by_head(
    model: object,
    cache: object,
    delta: Tensor,
    from_layer: int,
    attention_layer: int,
    to_layer: int,
) -> Tensor:
    """Split a propagated perturbation by which head at ``attention_layer`` carried it.

    At that layer the residual splits into one part per head plus a bypass, and each part then
    propagates forward on its own. The result is an exact partition: summing over the leading axis
    reproduces :func:`propagate`.

    Args:
        attention_layer: The layer whose heads the split is over. Must lie in
            ``[from_layer, to_layer)``.

    Returns:
        Tensor of shape ``(n_heads + 1, n_pos, d_model)``. The last slice is the bypass, meaning
        the part of the perturbation that attention at this layer did not touch.
    """
    if not from_layer <= attention_layer < to_layer:
        raise ValueError(
            f"attention_layer must lie in [{from_layer}, {to_layer}), got {attention_layer}"
        )
    arriving = propagate(model, cache, delta, from_layer, attention_layer)
    scale = cache[f"blocks.{attention_layer}.ln1.hook_scale"]  # type: ignore[index]
    scale = scale[0] if scale.ndim == 3 else scale
    per_head = attention_step(
        model,
        attention_layer,
        arriving,
        frozen_patterns(cache, attention_layer),
        scale,
        per_head=True,
    )
    parts = torch.cat([per_head, arriving.unsqueeze(0)], dim=0)
    return torch.stack(
        [propagate(model, cache, part, attention_layer + 1, to_layer) for part in parts]
    )


def to_transcoder_input(model: object, cache: object, delta: Tensor, layer: int) -> Tensor:
    """Carry a perturbation of ``blocks.{layer}.hook_resid_pre`` to what that layer's MLP reads.

    A transcoder feature reads the MLP input, which sits after the block's own attention and after
    ``ln2``. Both are applied here, with the normalisation scale frozen.

    Returns:
        Tensor of shape ``(n_pos, d_model)``.
    """
    scale = cache[f"blocks.{layer}.ln1.hook_scale"]  # type: ignore[index]
    scale = scale[0] if scale.ndim == 3 else scale
    mid = delta + attention_step(model, layer, delta, frozen_patterns(cache, layer), scale)
    return apply_ln2(model, cache, mid, layer)


def apply_ln2(model: object, cache: object, delta: Tensor, layer: int) -> Tensor:
    """Apply ``ln2`` to a perturbation already at ``blocks.{layer}.hook_resid_mid``.

    Separate from :func:`to_transcoder_input` so that a perturbation already split by head at this
    block's own attention can be read out without that attention step being applied twice.
    """
    ln2 = model.blocks[layer].ln2  # type: ignore[attr-defined]
    if getattr(ln2, "b", None) is not None:
        raise UnsupportedArchitecture(
            "ln2 has a bias, which does not act on a perturbation; RMSNorm is assumed here"
        )
    scale = cache[f"blocks.{layer}.ln2.hook_scale"]  # type: ignore[index]
    scale = scale[0] if scale.ndim == 3 else scale
    return delta * ln2.w.to(delta.dtype) / scale.to(delta.dtype)
