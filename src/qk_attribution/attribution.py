"""Decomposing one head's attention score into query-side by key-side source interactions.

The score is bilinear in the residual stream, so substituting a per-source breakdown at the query
and key positions expands it into one term per source pair. Both sides project into head space
once, after which the contraction is a single matrix product.

Scope is one query position and one head. ``experiments/004-low-rank-and-cost.md`` records why: over
all position pairs the contraction reaches 1.5 PFLOP per head at 512 tokens, and the intermediate
does not fit in memory.

The decomposition is only as complete as the sources it is given. Transcoder features account for
the MLP writes into the residual stream and nothing else, so at a middle layer a large part of the
residual comes from earlier attention outputs and is not covered. :func:`residual_remainder` exists
so that gap is measured rather than ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from qk_attribution.circuits import attention_scale
from qk_attribution.features import SourceSet
from qk_attribution.scores import to_head_space


@dataclass(frozen=True)
class QKAttribution:
    """Per-source-pair contributions to one head's scores from a single query position."""

    contributions: Tensor
    query_sources: SourceSet
    key_sources: SourceSet
    query_position: int

    @property
    def by_key_source(self) -> Tensor:
        """Contribution of each key-side source, summed over query-side sources."""
        return self.contributions.sum(dim=0)

    @property
    def by_query_source(self) -> Tensor:
        """Contribution of each query-side source, summed over key-side sources."""
        return self.contributions.sum(dim=1)

    def by_key_position(self, n_pos: int) -> Tensor:
        """Contribution to the score at each key position, shaped ``(n_pos,)``.

        Several key-side sources sit at the same position, and this is what compares against the
        row of the true score matrix.
        """
        totals = torch.zeros(
            n_pos, dtype=self.contributions.dtype, device=self.contributions.device
        )
        if len(self.key_sources):
            totals.index_add_(0, self.key_sources.positions, self.by_key_source)
        return totals

    def top_pairs(self, count: int) -> tuple[Tensor, Tensor]:
        """Return the ``count`` largest contributions by magnitude and their flat indices.

        Returns:
            ``(values, indices)`` where indices are into the flattened contribution matrix; use
            ``divmod(index, contributions.shape[1])`` for the query and key source.
        """
        flat = self.contributions.flatten()
        count = min(count, flat.numel())
        values, indices = flat.abs().topk(count)
        return flat[indices], indices


def qk_attribution(
    model: object,
    layer: int,
    head: int,
    query_position: int,
    query_sources: SourceSet,
    key_sources: SourceSet,
    *,
    rotations: Tensor | None,
    norm_scale: Tensor,
    query_scale: Tensor | None,
    key_scale: Tensor | None,
) -> QKAttribution:
    """Expand one head's attention score into source-pair terms for a single query position.

    Args:
        model: A HookedTransformer-backed model.
        layer: Block index of the attention layer.
        head: Query-head index.
        query_position: The position attending. Every query source must sit here.
        query_sources: Sources contributing to the residual at ``query_position``.
        key_sources: Sources at any position that may be attended to.
        rotations: From ``scores.rotation_matrices``, or None without rotary embeddings.
        norm_scale: From ``scores.layernorm_scale``.
        query_scale: QK-norm scale for this head, shaped ``(seq, 1)``, or None.
        key_scale: QK-norm scale for this head's key group, shaped ``(seq, 1)``, or None.

    Returns:
        A :class:`QKAttribution` whose contributions sum to the score contributed by the sources
        supplied. That equals the full score only when the sources reconstruct the residual.
    """
    if len(query_sources):
        elsewhere = query_sources.positions != query_position
        if bool(elsewhere.any()):
            raise ValueError(
                "every query source must sit at query_position; got positions "
                f"{query_sources.positions.unique().tolist()} for query_position {query_position}"
            )
    left = to_head_space(
        model,
        layer,
        head,
        query_sources.directions,
        query_sources.positions,
        side="query",
        rotations=rotations,
        norm_scale=norm_scale,
        qk_scale=query_scale,
    )
    right = to_head_space(
        model,
        layer,
        head,
        key_sources.directions,
        key_sources.positions,
        side="key",
        rotations=rotations,
        norm_scale=norm_scale,
        qk_scale=key_scale,
    )
    contributions = left @ right.transpose(-1, -2).to(left.dtype) / attention_scale(model)
    return QKAttribution(
        contributions=contributions,
        query_sources=query_sources,
        key_sources=key_sources,
        query_position=query_position,
    )


def residual_remainder(residual: Tensor, sources: SourceSet) -> Tensor:
    """Return the part of the residual stream the sources do not account for.

    Args:
        residual: The attention input, shaped ``(seq, d_model)``, from ``scores.attention_input``
            divided by nothing: this works in pre-layernorm coordinates, so pass the residual the
            sources were built against.
        sources: The sources whose directions are subtracted.

    Returns:
        Tensor of shape ``(seq, d_model)``. Its norm relative to the residual's is how much of the
        input the decomposition leaves unexplained.
    """
    if residual.ndim != 2:
        raise ValueError(f"residual must be 2D (seq, d_model), got {tuple(residual.shape)}")
    remainder = residual.clone()
    if len(sources):
        remainder.index_add_(0, sources.positions, -sources.directions.to(remainder.dtype))
    return remainder
