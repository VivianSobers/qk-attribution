"""Per-head QK and OV circuit matrices.

Attention splits into two weight-only bilinear forms per head. The QK circuit decides *where* a
head attends; the OV circuit decides *what* it moves once it has attended. Each collapses a pair of
projection matrices into one ``d_model x d_model`` operator that depends on weights alone, so it can
be computed once per head and reused across prompts.

For a model with learned absolute positions and no query/key normalisation:

    attention score  s(q, k) = x_q @ W_QK @ x_k / sqrt(d_head)
    head output      out(q)  = sum_k A[q, k] * (x_k @ W_OV)

Two common architecture features break the first identity, and both are detected rather than
ignored. See :func:`architecture_notes` and :class:`UnsupportedArchitecture`.

Rotary position embeddings rotate the query and key by a position-dependent angle before the dot
product. The form stays bilinear in the residual stream, but the operator acquires a dependence on
the position *offset*, becoming ``W_Q R(p_k - p_q) W_K.T`` rather than a single matrix.

Query/key normalisation applies RMSNorm to the projected query and key. That is a per-position,
per-head rescaling, so it can be absorbed as a frozen scalar in the same way circuit-tracer already
freezes layernorm scales, but it cannot be folded into the weight product.

Grouped-query attention needs no special handling for weights: TransformerLens materialises one
``W_K``/``W_V`` slice per query head, repeating key/value groups. Cached *activations* are not
expanded, so ``hook_k`` carries only ``n_key_value_heads`` slices; use :func:`kv_head_for` to map.

Performance note: ``model.W_Q`` on a HookedTransformer stacks every layer on each access, which
costs hundreds of megabytes per call. Everything here reads ``model.blocks[layer].attn.W_Q``
instead, so only one layer is ever touched.
"""

from __future__ import annotations

import torch
from torch import Tensor

#: Position schemes that leave the attention score bilinear in the residual stream. Anything else,
#: ALiBi and shortformer among them, adds a term this decomposition does not model.
_BILINEAR_POSITION_SCHEMES = frozenset({"standard", "rotary", None})


class UnsupportedArchitecture(RuntimeError):
    """Raised when a model's attention cannot be expressed as a single bilinear QK form."""


def attention_block(model: object, layer: int) -> object:
    """Return the attention submodule for one block, without stacking other layers."""
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("model has no .blocks; a HookedTransformer-backed model is required")
    if not 0 <= layer < len(blocks):
        raise IndexError(f"layer {layer} out of range for {len(blocks)} layers")
    return blocks[layer].attn


def n_layers(model: object) -> int:
    """Number of transformer blocks."""
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("model has no .blocks")
    return len(blocks)


def n_heads(model: object) -> int:
    """Number of query heads."""
    return int(attention_block(model, 0).W_Q.shape[0])  # type: ignore[attr-defined]


def n_kv_heads(model: object) -> int:
    """Number of key/value heads. Equals :func:`n_heads` unless the model uses GQA."""
    cfg = getattr(model, "cfg", None)
    value = getattr(cfg, "n_key_value_heads", None) if cfg is not None else None
    return int(value) if value else n_heads(model)


def d_head(model: object) -> int:
    """Head dimension."""
    return int(attention_block(model, 0).W_Q.shape[2])  # type: ignore[attr-defined]


def kv_head_for(model: object, head: int) -> int:
    """Map a query-head index to its key/value group index.

    Cached key and value activations carry one slice per key/value head rather than per query head,
    so this is needed whenever hook activations are indexed by head under GQA.
    """
    heads, kv = n_heads(model), n_kv_heads(model)
    if not 0 <= head < heads:
        raise IndexError(f"head {head} out of range for {heads} heads")
    if heads % kv != 0:
        raise ValueError(f"n_heads={heads} is not a multiple of n_key_value_heads={kv}")
    return head // (heads // kv)


def attention_scale(model: object) -> float:
    """Divisor applied to raw attention scores, normally ``sqrt(d_head)``.

    A model can switch scaling off entirely, in which case the divisor is 1 and using
    ``sqrt(d_head)`` anyway would scale every score and every contribution by a constant without
    anything failing.
    """
    cfg = getattr(model, "cfg", None)
    if cfg is not None and getattr(cfg, "use_attn_scale", True) is False:
        return 1.0
    scale = getattr(cfg, "attn_scale", None) if cfg is not None else None
    return float(scale) if scale else float(d_head(model) ** 0.5)


def architecture_notes(model: object) -> dict[str, object]:
    """Report attention features that affect how scores may be decomposed.

    Returns a mapping with keys ``rotary``, ``qk_norm``, ``score_soft_cap`` and ``gqa``. Callers
    that need a single bilinear QK operator should check this first, or call
    :func:`require_plain_qk`.
    """
    cfg = getattr(model, "cfg", None)
    attn = attention_block(model, 0)
    pos_type = getattr(cfg, "positional_embedding_type", None) if cfg is not None else None
    soft_cap = getattr(cfg, "attn_scores_soft_cap", None) if cfg is not None else None
    qk_norm = bool(getattr(cfg, "use_qk_norm", False)) or getattr(attn, "q_norm", None) is not None
    return {
        "rotary": pos_type == "rotary",
        "positional_embedding_type": pos_type,
        "rotary_dim": getattr(cfg, "rotary_dim", None) if cfg is not None else None,
        "qk_norm": qk_norm,
        # TransformerLens signals "disabled" with a non-positive value.
        "score_soft_cap": float(soft_cap) if soft_cap and float(soft_cap) > 0 else None,
        "gqa": n_kv_heads(model) != n_heads(model),
    }


def require_plain_qk(model: object) -> None:
    """Raise unless the model's scores equal ``x_q @ W_QK @ x_k / scale``.

    Raises:
        UnsupportedArchitecture: listing every feature that invalidates the identity.
    """
    notes = architecture_notes(model)
    problems = []
    if notes["rotary"]:
        problems.append("rotary position embeddings make the QK operator offset-dependent")
    position_scheme = notes["positional_embedding_type"]
    if position_scheme not in _BILINEAR_POSITION_SCHEMES and not notes["rotary"]:
        problems.append(
            f"positional embedding type {position_scheme!r} is not known to leave the score "
            "bilinear in the residual stream"
        )
    if notes["qk_norm"]:
        problems.append("query/key normalisation rescales q and k after projection")
    if notes["score_soft_cap"] is not None:
        problems.append(f"attention scores are soft-capped at {notes['score_soft_cap']}")
    if problems:
        raise UnsupportedArchitecture(
            "attention scores are not a single bilinear form in the residual stream: "
            + "; ".join(problems)
        )


def qk_matrix(model: object, layer: int, head: int, *, scaled: bool = False) -> Tensor:
    """Return the QK circuit ``W_Q @ W_K.T`` for one head.

    This is always the product of the two projection matrices. Whether it is also the model's score
    operator depends on the architecture; see :func:`require_plain_qk`.

    Args:
        model: A HookedTransformer-backed model.
        layer: Block index.
        head: Query-head index.
        scaled: Divide by the model's attention scale so the result is comparable to pre-softmax
            scores.

    Returns:
        Tensor of shape ``(d_model, d_model)``.
    """
    attn = attention_block(model, layer)
    heads = n_heads(model)
    if not 0 <= head < heads:
        raise IndexError(f"head {head} out of range for {heads} heads")
    out = attn.W_Q[head] @ attn.W_K[head].transpose(-1, -2)  # type: ignore[attr-defined]
    if scaled:
        out = out / attention_scale(model)
    return out


def ov_matrix(model: object, layer: int, head: int) -> Tensor:
    """Return the OV circuit ``W_V @ W_O`` for one head.

    Unlike the QK circuit this is exact for every architecture here, since nothing is applied
    between the value projection and the output projection.

    Returns:
        Tensor of shape ``(d_model, d_model)``.
    """
    attn = attention_block(model, layer)
    heads = n_heads(model)
    if not 0 <= head < heads:
        raise IndexError(f"head {head} out of range for {heads} heads")
    return attn.W_V[head] @ attn.W_O[head]  # type: ignore[attr-defined]


def attention_scores_from_residual(
    model: object,
    layer: int,
    head: int,
    residual: Tensor,
    *,
    check_architecture: bool = True,
) -> Tensor:
    """Recompute pre-softmax attention scores for one head from a residual stream.

    Exists so :func:`qk_matrix` can be checked against the model's own forward pass rather than
    trusted. No causal mask or softmax is applied.

    Args:
        model: A HookedTransformer-backed model.
        layer: Block index.
        head: Query-head index.
        residual: Post-layernorm residual entering attention, shaped ``(seq, d_model)`` or
            ``(batch, seq, d_model)``.
        check_architecture: If True, refuse models whose scores are not a plain bilinear form.
            Pass False only to measure how large the resulting discrepancy is.

    Returns:
        Scores indexed ``[query, key]``.
    """
    if residual.ndim not in (2, 3):
        raise ValueError(f"residual must be 2D or 3D, got shape {tuple(residual.shape)}")
    if check_architecture:
        require_plain_qk(model)
    w_qk = qk_matrix(model, layer, head, scaled=True).to(residual.dtype)
    return residual @ w_qk @ residual.transpose(-1, -2)


def head_output_from_residual(
    model: object,
    layer: int,
    head: int,
    residual: Tensor,
    pattern: Tensor,
) -> Tensor:
    """Recompute one head's write-back into the residual stream.

    Args:
        model: A HookedTransformer-backed model.
        layer: Block index.
        head: Query-head index.
        residual: Post-layernorm residual entering attention, shaped ``(seq, d_model)``.
        pattern: Post-softmax probabilities for this head, shaped ``(seq, seq)``, indexed
            ``[query, key]``.

    Returns:
        Tensor of shape ``(seq, d_model)``.
    """
    if residual.ndim != 2:
        raise ValueError(f"residual must be 2D (seq, d_model), got {tuple(residual.shape)}")
    if pattern.ndim != 2:
        raise ValueError(f"pattern must be 2D (query, key), got {tuple(pattern.shape)}")
    seq = residual.shape[0]
    if pattern.shape != (seq, seq):
        raise ValueError(
            f"pattern {tuple(pattern.shape)} inconsistent with residual {tuple(residual.shape)}"
        )
    w_ov = ov_matrix(model, layer, head).to(residual.dtype)
    return pattern.to(residual.dtype) @ (residual @ w_ov)


def qk_low_rank_spectrum(model: object, layer: int, head: int) -> Tensor:
    """Return singular values of a head's QK circuit, largest first.

    Anthropic report that many QK attribution matrices are approximately low-rank, which is the
    route to making feature-pair decomposition tractable at longer contexts. Exposing the spectrum
    lets that be measured per head rather than assumed.
    """
    return torch.linalg.svdvals(qk_matrix(model, layer, head).float())


def effective_rank(singular_values: Tensor, energy: float = 0.99) -> int:
    """Smallest number of singular values whose squares reach ``energy`` of the total.

    Args:
        singular_values: Descending singular values.
        energy: Fraction of squared spectrum to capture, in (0, 1].

    Returns:
        The component count, at least 1.
    """
    if not 0.0 < energy <= 1.0:
        raise ValueError(f"energy must be in (0, 1], got {energy}")
    if singular_values.ndim != 1 or singular_values.numel() == 0:
        raise ValueError("singular_values must be a non-empty 1D tensor")
    squared = singular_values.float() ** 2
    cumulative = torch.cumsum(squared, dim=0) / squared.sum()
    # Counting rather than searchsorted so the comparison stays on the input's own device. The
    # cumulative sum can land just below 1 in floating point, so at energy=1 the count would
    # otherwise run one past the end.
    return min(int((cumulative < energy).sum().item()) + 1, singular_values.numel())
