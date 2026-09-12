"""Exact reconstruction of attention scores as a bilinear form in the residual stream.

:mod:`qk_attribution.circuits` shows that ``W_Q @ W_K.T`` is not the score operator for a modern
model: rotary embeddings and query/key normalisation both intervene. Neither destroys bilinearity,
though, and this module writes down what survives. For one head,

    s(p, j) = x_p @ [ W_Q diag(w_q) R(p) R(j).T diag(w_k) W_K.T ] @ x_j
              / ( attn_scale * sigma_q[p] * sigma_k[j] )

where

* ``x`` is the residual entering attention, which is ``ln1.hook_normalized * ln1.w`` and **not**
  ``hook_normalized`` alone -- TransformerLens fires that hook before applying the learned gain;
* ``w_q`` and ``w_k`` are the QK-norm gains, which fold into the projections;
* ``R(p)`` is the rotary rotation at position ``p``, orthogonal, and with ``R(p) R(j).T`` depending
  only on the offset ``p - j``;
* ``sigma_q[p]`` and ``sigma_k[j]`` are the QK-norm RMS scales, which are per-position scalars and
  are therefore frozen exactly as circuit-tracer already freezes layernorm scales.

Everything outside the bracket is a scalar, so the operator inside it is what a feature-pair
decomposition attaches to. Query-side and key-side feature vectors project into head space once
each, and their interaction is then a ``d_head x d_head`` rotation apart.

Verified against Qwen3-0.6B: 2.5e-07 median relative error over all 448 (layer, head) pairs. See
``experiments/003-exact-qk-form.md``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from qk_attribution.circuits import (
    UnsupportedArchitecture,
    architecture_notes,
    attention_block,
    attention_scale,
    d_head,
    n_heads,
    n_kv_heads,
)


def require_supported(model: object) -> None:
    """Raise unless this module can reproduce the model's scores exactly.

    Rotary embeddings and QK-norm are handled. Attention-score soft-capping is not: it applies a
    ``tanh`` to the score itself, which is outside the bilinear form rather than a rescaling of it.

    Raises:
        UnsupportedArchitecture: if soft-capping is enabled.
    """
    cap = architecture_notes(model)["score_soft_cap"]
    if cap is not None:
        raise UnsupportedArchitecture(
            f"attention scores are soft-capped at {cap}, which is not a bilinear form; "
            "the cap would have to be linearised or frozen first"
        )


def attention_input(model: object, cache: object, layer: int) -> Tensor:
    """Return the residual entering attention at ``layer``, for one prompt.

    ``ln1.hook_normalized`` is captured *before* the learned gain is applied, so using it directly
    as the projection input is wrong by a per-channel factor. This applies the gain.

    Args:
        model: A HookedTransformer-backed model.
        cache: An ``ActivationCache`` from a single-prompt run.
        layer: Block index.

    Returns:
        Tensor of shape ``(seq, d_model)``.
    """
    ln1 = model.blocks[layer].ln1  # type: ignore[attr-defined]
    normalized = cache[f"blocks.{layer}.ln1.hook_normalized"]  # type: ignore[index]
    if normalized.ndim == 3:
        normalized = normalized[0]
    scaled = normalized * ln1.w.to(normalized.dtype)
    bias = getattr(ln1, "b", None)
    return scaled + bias.to(scaled.dtype) if bias is not None else scaled


def qk_norm_scales(model: object, cache: object, layer: int) -> tuple[Tensor, Tensor] | None:
    """Return the per-position QK-norm RMS scales, or None if the model has no QK-norm.

    TransformerLens flattens these hooks over ``(batch, pos, head)``, so the head axis is restored
    here. Query and key scales have different head counts under grouped-query attention.

    Returns:
        ``(sigma_q, sigma_k)`` shaped ``(seq, n_heads, 1)`` and ``(seq, n_kv_heads, 1)``.
    """
    if not architecture_notes(model)["qk_norm"]:
        return None
    flat_q = cache[f"blocks.{layer}.attn.q_norm.hook_scale"]  # type: ignore[index]
    flat_k = cache[f"blocks.{layer}.attn.k_norm.hook_scale"]  # type: ignore[index]
    seq = flat_q.shape[0] // n_heads(model)
    return (
        flat_q.reshape(seq, n_heads(model), 1),
        flat_k.reshape(seq, n_kv_heads(model), 1),
    )


def rotation_matrices(model: object, n_pos: int) -> Tensor:
    """Return the rotary rotation for each position as an explicit matrix.

    Built by rotating the standard basis with the model's own ``apply_rotary``, so it cannot drift
    from whatever convention that uses (adjacent-pair versus split-half, NTK scaling, and so on).

    The result costs ``n_pos * d_head**2`` floats. RoPE is relative, so only ``2 * n_pos - 1``
    distinct products exist; indexing by position rather than offset is the simpler arrangement and
    the one the score reconstruction wants.

    Returns:
        Tensor of shape ``(n_pos, d_head, d_head)``, where ``v @ result[p]`` equals rotating ``v``
        at position ``p``.
    """
    if n_pos < 1:
        raise ValueError(f"n_pos must be positive, got {n_pos}")
    attn = attention_block(model, 0)
    if not architecture_notes(model)["rotary"]:
        raise UnsupportedArchitecture("model does not use rotary position embeddings")
    size = d_head(model)
    weight = attn.W_Q  # type: ignore[attr-defined]
    basis = torch.eye(size, device=weight.device, dtype=weight.dtype)
    basis = basis.reshape(size, 1, 1, size).expand(size, n_pos, 1, size).contiguous()
    rotated = attn.apply_rotary(basis, 0, None)  # type: ignore[attr-defined]
    return rotated.squeeze(2).permute(1, 0, 2).contiguous()


def relative_rotation(rotations: Tensor, query_pos: int, key_pos: int) -> Tensor:
    """Return ``R(query_pos) @ R(key_pos).T``, the operator between a query and a key position.

    This depends only on the offset between the two positions, which is what makes caching by
    offset possible rather than by pair.
    """
    return rotations[query_pos] @ rotations[key_pos].transpose(-1, -2)


def query_projection(model: object, layer: int, head: int) -> Tensor:
    """Return ``W_Q diag(w_q)`` for one head, mapping the residual into query head space.

    The QK-norm gain folds in because it is a fixed per-channel scaling of the projected query. The
    projection bias does not; see :func:`attention_scores`.

    Returns:
        Tensor of shape ``(d_model, d_head)``.
    """
    return _projection(model, layer, head, "W_Q", "q_norm")


def key_projection(model: object, layer: int, head: int) -> Tensor:
    """Return ``W_K diag(w_k)`` for one head, mapping the residual into key head space.

    Indexing is by *query* head: TransformerLens materialises one key slice per query head, so this
    needs no grouped-query adjustment. Cached key activations do; see
    :func:`qk_attribution.circuits.kv_head_for`.

    Returns:
        Tensor of shape ``(d_model, d_head)``.
    """
    return _projection(model, layer, head, "W_K", "k_norm")


def _projection(model: object, layer: int, head: int, weight: str, norm: str) -> Tensor:
    attn = attention_block(model, layer)
    heads = n_heads(model)
    if not 0 <= head < heads:
        raise IndexError(f"head {head} out of range for {heads} heads")
    projection = getattr(attn, weight)[head]
    module = getattr(attn, norm, None)
    return projection * module.w if module is not None else projection


def attention_scores(
    model: object,
    layer: int,
    head: int,
    residual: Tensor,
    *,
    query_scale: Tensor | None = None,
    key_scale: Tensor | None = None,
    rotations: Tensor | None = None,
) -> Tensor:
    """Recompute one head's pre-softmax scores from the residual stream and the weights.

    No causal mask and no softmax are applied, so the result is comparable to
    ``hook_attn_scores`` only on positions the mask keeps.

    Args:
        model: A HookedTransformer-backed model.
        layer: Block index.
        head: Query-head index.
        residual: The attention input from :func:`attention_input`, shaped ``(seq, d_model)``.
        query_scale: QK-norm RMS scale for this head, shaped ``(seq, 1)``. Required when the model
            uses QK-norm, since it depends on the activations and cannot be derived from weights.
        key_scale: As above for the key side.
        rotations: Output of :func:`rotation_matrices`, at least ``seq`` long. Computed here if
            omitted, which is wasteful when looping over heads.

    Returns:
        Scores indexed ``[query, key]``.

    Raises:
        UnsupportedArchitecture: if the model soft-caps scores.
        ValueError: if the QK-norm scales are needed but not supplied, or shapes disagree.
    """
    require_supported(model)
    if residual.ndim != 2:
        raise ValueError(f"residual must be 2D (seq, d_model), got {tuple(residual.shape)}")
    seq = residual.shape[0]
    attn = attention_block(model, layer)
    notes = architecture_notes(model)

    query = residual @ query_projection(model, layer, head).to(residual.dtype)
    key = residual @ key_projection(model, layer, head).to(residual.dtype)
    query, key = _add_projection_bias(attn, head, query, key)

    if notes["qk_norm"]:
        if query_scale is None or key_scale is None:
            raise ValueError(
                "this model applies QK-norm, so query_scale and key_scale are required; "
                "take them from qk_norm_scales()"
            )
        query = query / _check_scale(query_scale, seq, "query_scale").to(query.dtype)
        key = key / _check_scale(key_scale, seq, "key_scale").to(key.dtype)

    if notes["rotary"]:
        if rotations is None:
            rotations = rotation_matrices(model, seq)
        if rotations.shape[0] < seq:
            raise ValueError(f"rotations cover {rotations.shape[0]} positions, need at least {seq}")
        window = rotations[:seq].to(query.dtype)
        query = torch.einsum("pa,pab->pb", query, window)
        key = torch.einsum("ja,jab->jb", key, window)

    return query @ key.transpose(-1, -2) / attention_scale(model)


def _add_projection_bias(
    attn: object, head: int, query: Tensor, key: Tensor
) -> tuple[Tensor, Tensor]:
    """Add ``b_Q`` and ``b_K``, gain-folded, where the model has them.

    A non-zero bias makes the score affine rather than bilinear in the residual. It stays exact for
    reconstruction; for attribution it contributes a constant term on each side, the way an error
    node does. Qwen3 has no attention bias, so this is inert there.
    """
    out = []
    for name, norm, vector in (("b_Q", "q_norm", query), ("b_K", "k_norm", key)):
        bias = getattr(attn, name, None)
        if bias is None:
            out.append(vector)
            continue
        module = getattr(attn, norm, None)
        gain = bias[head] * module.w if module is not None else bias[head]
        out.append(vector + gain.to(vector.dtype))
    return out[0], out[1]


def _check_scale(scale: Tensor, seq: int, name: str) -> Tensor:
    if scale.shape[0] != seq:
        raise ValueError(f"{name} covers {scale.shape[0]} positions, expected {seq}")
    if scale.ndim == 1:
        scale = scale[:, None]
    if scale.ndim != 2 or scale.shape[1] != 1:
        raise ValueError(f"{name} must be (seq, 1), got {tuple(scale.shape)}")
    return scale


def layernorm_scale(cache: object, layer: int) -> Tensor:
    """Return the RMS scale that ``ln1`` divides by, shaped ``(seq, 1)``.

    A feature's decoder direction enters the residual stream before this division, so mapping it
    into attention means dividing by the same per-position scalar. It is frozen, exactly as
    circuit-tracer already freezes it.
    """
    scale = cache[f"blocks.{layer}.ln1.hook_scale"]  # type: ignore[index]
    if scale.ndim == 3:
        scale = scale[0]
    return scale


def to_head_space(
    model: object,
    layer: int,
    head: int,
    directions: Tensor,
    positions: Tensor,
    *,
    side: str,
    rotations: Tensor | None,
    norm_scale: Tensor,
    qk_scale: Tensor | None,
) -> Tensor:
    """Project residual-stream directions into one head's rotated query or key space.

    This is the same path :func:`attention_scores` takes, applied to individual directions rather
    than to the whole residual, which is what makes the decomposition possible: each direction can
    be projected once and the interaction between any query-side and key-side pair is then a dot
    product.

    Args:
        model: A HookedTransformer-backed model.
        layer: Block index.
        head: Query-head index.
        directions: Rows in residual-stream coordinates, shaped ``(n, d_model)``, each already
            scaled by its feature's activation.
        positions: Sequence position of each row, shaped ``(n,)``, used to pick the rotation and
            the frozen scales.
        side: ``"query"`` or ``"key"``.
        rotations: From :func:`rotation_matrices`, or None for a model without rotary embeddings.
        norm_scale: From :func:`layernorm_scale`, shaped ``(seq, 1)``. Directions are given in
            residual-stream coordinates before ``ln1``, so this applies ``ln1`` in full: the
            learned gain as well as the division.
        qk_scale: QK-norm scale for this head, shaped ``(seq, 1)``, or None without QK-norm.

    Returns:
        Tensor of shape ``(n, d_head)``.
    """
    if side not in ("query", "key"):
        raise ValueError(f"side must be 'query' or 'key', got {side!r}")
    if directions.ndim != 2:
        raise ValueError(f"directions must be 2D (n, d_model), got {tuple(directions.shape)}")
    if positions.shape[0] != directions.shape[0]:
        raise ValueError(f"{positions.shape[0]} positions for {directions.shape[0]} directions")
    ln1 = model.blocks[layer].ln1  # type: ignore[attr-defined]
    if getattr(ln1, "b", None) is not None:
        raise UnsupportedArchitecture(
            "ln1 has a bias, which is an additive term rather than a per-direction scaling; "
            "carry it as a separate source instead"
        )
    projection = query_projection if side == "query" else key_projection
    dtype = directions.dtype
    scaled = directions * ln1.w.to(dtype) / norm_scale[positions].to(dtype)
    out = scaled @ projection(model, layer, head).to(dtype)
    if qk_scale is not None:
        out = out / qk_scale[positions].to(dtype)
    if rotations is not None:
        out = torch.einsum("na,nab->nb", out, rotations[positions].to(out.dtype))
    return out
