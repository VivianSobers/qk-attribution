"""Reading source directions for the QK decomposition out of a circuit-tracer graph.

A graph's ``active_features`` is an ``(n, 3)`` tensor of ``(layer, pos, feature_idx)``, and
``selected_features`` indexes into it. A per-layer transcoder feature writes its decoder row into
the residual stream at its own layer's MLP output, so attention at a later layer sees it and
attention at an earlier layer does not.

This module is the only place that knows circuit-tracer's data layout. Everything downstream takes
a :class:`SourceSet`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

LAYER, POSITION, FEATURE = 0, 1, 2

#: Sentinel layer and feature id for a source that is not a transcoder feature.
REMAINDER = -1


@dataclass(frozen=True)
class SourceSet:
    """Residual-stream directions feeding one attention layer, with their provenance.

    ``directions`` are already scaled by their activations, so they sum to the part of the residual
    stream the features account for. The other fields exist so a contribution can be traced back to
    the feature that produced it.
    """

    directions: Tensor
    positions: Tensor
    layers: Tensor
    feature_ids: Tensor
    activations: Tensor

    def __len__(self) -> int:
        return int(self.directions.shape[0])

    def concat(self, other: SourceSet) -> SourceSet:
        """Return the two source sets joined, so a decomposition over both is exhaustive."""
        return SourceSet(
            directions=torch.cat([self.directions, other.directions.to(self.directions.dtype)]),
            positions=torch.cat([self.positions, other.positions]),
            layers=torch.cat([self.layers, other.layers]),
            feature_ids=torch.cat([self.feature_ids, other.feature_ids]),
            activations=torch.cat([self.activations, other.activations]),
        )

    @property
    def is_remainder(self) -> Tensor:
        """Boolean mask marking rows that stand for what no feature explains."""
        return self.layers == REMAINDER

    def at_position(self, position: int) -> SourceSet:
        """Return the subset sitting at one sequence position."""
        keep = self.positions == position
        return SourceSet(
            directions=self.directions[keep],
            positions=self.positions[keep],
            layers=self.layers[keep],
            feature_ids=self.feature_ids[keep],
            activations=self.activations[keep],
        )


def feature_sources(
    graph: Any, transcoders: Any, *, below_layer: int, device: torch.device | None = None
) -> SourceSet:
    """Collect the activation-scaled decoder directions visible to attention at ``below_layer``.

    Args:
        graph: A circuit-tracer ``Graph``.
        transcoders: A ``TranscoderSet``, indexed by layer.
        below_layer: Only features written at a strictly earlier layer are included, since a
            feature cannot influence attention in the layer that produces it or any before it.
        device: Where to build the result. Defaults to the decoder weights' device.

    Returns:
        A :class:`SourceSet`, empty but correctly shaped when nothing qualifies.
    """
    if below_layer < 0:
        raise ValueError(f"below_layer must be non-negative, got {below_layer}")
    selected = graph.selected_features
    active = graph.active_features[selected]
    if active.ndim != 2 or active.shape[1] != 3:
        raise ValueError(
            f"active_features must be (n, 3) of (layer, pos, feature), got {tuple(active.shape)}"
        )
    keep = active[:, LAYER] < below_layer
    layers = active[keep, LAYER]
    positions = active[keep, POSITION]
    feature_ids = active[keep, FEATURE]
    activations = graph.activation_values[keep]

    template = transcoders[0].W_dec
    device = device if device is not None else template.device
    directions = torch.empty(
        (int(keep.sum()), template.shape[1]), dtype=template.dtype, device=device
    )
    # One row at a time, because the decoders are lazily loaded per layer and stacking them all
    # would materialise every transcoder at once.
    for row, (layer, feature) in enumerate(zip(layers.tolist(), feature_ids.tolist(), strict=True)):
        directions[row] = transcoders[layer].W_dec[feature].to(device)
    directions = directions * activations.to(device=device, dtype=directions.dtype).unsqueeze(-1)

    return SourceSet(
        directions=directions,
        positions=positions.to(device),
        layers=layers.to(device),
        feature_ids=feature_ids.to(device),
        activations=activations.to(device),
    )


def remainder_sources(remainder: Tensor) -> SourceSet:
    """Wrap the unexplained part of a residual stream as one source per position.

    Transcoder features cover only the MLP writes. Earlier attention outputs, the token embedding,
    transcoder errors and decoder biases are all in the residual too, and dropping them would make
    the decomposition silently incomplete. Carrying them as a single lumped direction per position
    keeps the expansion exhaustive: the contributions then sum to the true score, and the share
    carried by real features is visible rather than assumed.

    Args:
        remainder: Output of ``attribution.residual_remainder``, shaped ``(seq, d_model)``.

    Returns:
        A :class:`SourceSet` with one row per position, marked by :data:`REMAINDER`.
    """
    if remainder.ndim != 2:
        raise ValueError(f"remainder must be 2D (seq, d_model), got {tuple(remainder.shape)}")
    seq = remainder.shape[0]
    device = remainder.device
    return SourceSet(
        directions=remainder,
        positions=torch.arange(seq, device=device),
        layers=torch.full((seq,), REMAINDER, dtype=torch.int64, device=device),
        feature_ids=torch.full((seq,), REMAINDER, dtype=torch.int64, device=device),
        activations=torch.ones(seq, device=device),
    )
