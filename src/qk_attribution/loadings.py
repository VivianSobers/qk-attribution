"""Splitting an attribution-graph edge by the head that carried it.

An attribution graph says that a feature at one position influenced a feature at another, but not
which attention head moved the signal. Under the constraints the graph is built with, the answer is
exact: at any attention layer the residual stream splits into one part per head plus a bypass, each
part propagates forward independently, and the target reads their sum.

The split is per attention layer. Paths compose, so a signal can pass through heads at several
layers on its way, and there is no single head to credit. Asking for the split at one layer is a
well-posed question with an exact answer; asking for a single joint attribution across layers is
not, and this module does not pretend otherwise.

Unlike QK attribution, nothing here depends on the model's architecture beyond having attention:
the OV pathway is a plain product of the value and output projections for every model in scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from qk_attribution.circuits import n_heads
from qk_attribution.propagate import (
    apply_ln2,
    attention_step,
    frozen_patterns,
    propagate,
    split_by_head,
    to_transcoder_input,
)


@dataclass(frozen=True)
class EdgeLoadings:
    """How one edge's effect divides across the heads of a single attention layer."""

    per_head: Tensor
    bypass: Tensor
    attention_layer: int

    @property
    def total(self) -> Tensor:
        """The whole edge effect, which the parts sum to."""
        return self.per_head.sum() + self.bypass

    def ranked(self) -> tuple[Tensor, Tensor]:
        """Head indices ordered by contribution magnitude, with their signed values."""
        order = self.per_head.abs().argsort(descending=True)
        return order, self.per_head[order]


def edge_effect(
    model: object,
    cache: object,
    source: Tensor,
    source_position: int,
    source_layer: int,
    reader: Tensor,
    target_position: int,
    target_layer: int,
) -> Tensor:
    """Return the linear effect of a source direction on a target feature's pre-activation.

    Args:
        model: A HookedTransformer-backed model.
        cache: An ``ActivationCache`` from the clean run, supplying the frozen patterns and scales.
        source: Residual-stream direction written by the source, shaped ``(d_model,)``, already
            scaled by the source's activation.
        source_position: Where it is written.
        source_layer: The block whose MLP writes it. The direction enters the residual entering
            the next block.
        reader: The target feature's encoder row, shaped ``(d_model,)``.
        target_position: Where the target feature sits.
        target_layer: The block whose MLP the target feature belongs to.

    Returns:
        A scalar tensor.
    """
    delta = _seed(model, cache, source, source_position)
    arriving = propagate(model, cache, delta, source_layer + 1, target_layer)
    return to_transcoder_input(model, cache, arriving, target_layer)[target_position] @ reader


def edge_loadings(
    model: object,
    cache: object,
    source: Tensor,
    source_position: int,
    source_layer: int,
    reader: Tensor,
    target_position: int,
    target_layer: int,
    attention_layer: int,
) -> EdgeLoadings:
    """Split :func:`edge_effect` across the heads of one attention layer.

    Args:
        attention_layer: Must lie in ``[source_layer + 1, target_layer]``. The target layer's own
            attention is included, since the target reads the residual after it.

    Returns:
        An :class:`EdgeLoadings` whose parts sum to :func:`edge_effect`.
    """
    if not source_layer < attention_layer <= target_layer:
        raise ValueError(
            f"attention_layer must lie in ({source_layer}, {target_layer}], got {attention_layer}"
        )
    delta = _seed(model, cache, source, source_position)
    if attention_layer == target_layer:
        # The target reads after its own block's attention, so the split happens at the readout:
        # the parts are each head's write plus the residual that reached the block untouched.
        arriving = propagate(model, cache, delta, source_layer + 1, target_layer)
        scale = cache[f"blocks.{target_layer}.ln1.hook_scale"]  # type: ignore[index]
        scale = scale[0] if scale.ndim == 3 else scale
        written = attention_step(
            model,
            target_layer,
            arriving,
            frozen_patterns(cache, target_layer),
            scale,
            per_head=True,
        )
        parts = torch.cat([written, arriving.unsqueeze(0)], dim=0)
    else:
        parts = split_by_head(model, cache, delta, source_layer + 1, attention_layer, target_layer)
        parts = torch.stack(
            [
                part
                + attention_step(
                    model,
                    target_layer,
                    part,
                    frozen_patterns(cache, target_layer),
                    _ln1_scale(cache, target_layer),
                )
                for part in parts
            ]
        )
    readouts = torch.stack(
        [apply_ln2(model, cache, part, target_layer)[target_position] @ reader for part in parts]
    )
    return EdgeLoadings(
        per_head=readouts[: n_heads(model)],
        bypass=readouts[n_heads(model)],
        attention_layer=attention_layer,
    )


def _ln1_scale(cache: object, layer: int) -> Tensor:
    scale = cache[f"blocks.{layer}.ln1.hook_scale"]  # type: ignore[index]
    return scale[0] if scale.ndim == 3 else scale


def _seed(model: object, cache: object, source: Tensor, position: int) -> Tensor:
    """Place a source direction at one position of an otherwise empty residual stream."""
    if source.ndim != 1:
        raise ValueError(f"source must be 1D (d_model,), got {tuple(source.shape)}")
    n_pos = int(cache["blocks.0.attn.hook_pattern"].shape[-1])  # type: ignore[index]
    if not 0 <= position < n_pos:
        raise IndexError(f"position {position} out of range for {n_pos} positions")
    delta = torch.zeros(n_pos, source.shape[0], dtype=source.dtype, device=source.device)
    delta[position] = source
    return delta


@dataclass(frozen=True)
class PathLoadings:
    """Every attention layer's split of one edge, computed together.

    Row ``i`` of ``per_head`` and entry ``i`` of ``bypass`` belong to ``layers[i]``. Each row is its
    own partition of ``total``; the rows are not parts of one joint decomposition.
    """

    layers: list[int]
    per_head: Tensor
    bypass: Tensor
    total: Tensor

    def at(self, layer: int) -> EdgeLoadings:
        """The split at one attention layer, in the form :func:`edge_loadings` returns."""
        if layer not in self.layers:
            raise ValueError(f"layer {layer} is not on this path, which covers {self.layers}")
        row = self.layers.index(layer)
        return EdgeLoadings(
            per_head=self.per_head[row], bypass=self.bypass[row], attention_layer=layer
        )


def path_loadings(
    model: object,
    cache: object,
    source: Tensor,
    source_position: int,
    source_layer: int,
    reader: Tensor,
    target_position: int,
    target_layer: int,
) -> PathLoadings:
    """Split one edge at every attention layer on its path, in a single sweep each way.

    Calling :func:`edge_loadings` once per layer re-propagates every head's part from that layer to
    the target, which costs on the order of ``L**2 * n_heads`` attention steps for a path ``L``
    layers long. Everything here is linear, so the readout is a fixed linear functional of the
    perturbation at each layer. One forward sweep gives the perturbation arriving at every layer,
    one backward sweep gives that functional, and each head's loading is then a dot product. The
    cost falls to a few attention steps per layer.

    Args:
        source_layer: The block whose MLP writes the source. A token embedding is written before
            block 0 and uses ``-1``.
        target_layer: The block whose MLP the target feature belongs to.

    Returns:
        A :class:`PathLoadings` over layers ``source_layer + 1`` to ``target_layer`` inclusive, each
        agreeing with :func:`edge_loadings` for that layer.
    """
    if target_layer <= source_layer:
        raise ValueError(
            f"target_layer must come after source_layer, got {source_layer} and {target_layer}"
        )
    first = source_layer + 1

    # Gradients are taken with autograd.grad rather than backward, so nothing accumulates on a real
    # model's parameters, and grad mode is restored for callers that turned it off.
    with torch.enable_grad():
        seed = _seed(model, cache, source.detach(), source_position).requires_grad_(True)
        states = [seed]
        for layer in range(first, target_layer):
            state = states[-1]
            patterns, scale = frozen_patterns(cache, layer), _ln1_scale(cache, layer)
            states.append(state + attention_step(model, layer, state, patterns, scale))
        readout = (
            to_transcoder_input(model, cache, states[-1], target_layer)[target_position]
            @ reader.detach()
        )
        # Each entry is the readout's sensitivity to the residual entering that layer.
        sensitivities = torch.autograd.grad(readout, states)

    with torch.no_grad():
        per_head, bypass = [], []
        for offset, layer in enumerate(range(first, target_layer)):
            arriving = states[offset].detach()
            written = attention_step(
                model,
                layer,
                arriving,
                frozen_patterns(cache, layer),
                _ln1_scale(cache, layer),
                per_head=True,
            )
            onward = sensitivities[offset + 1]
            per_head.append(torch.einsum("hpd,pd->h", written, onward))
            bypass.append((arriving * onward).sum())

        # At the target's own block the split happens at the readout, as in edge_loadings.
        arriving = states[-1].detach()
        written = attention_step(
            model,
            target_layer,
            arriving,
            frozen_patterns(cache, target_layer),
            _ln1_scale(cache, target_layer),
            per_head=True,
        )
        reader = reader.detach()
        per_head.append(apply_ln2(model, cache, written, target_layer)[:, target_position] @ reader)
        bypass.append(apply_ln2(model, cache, arriving, target_layer)[target_position] @ reader)

        return PathLoadings(
            layers=list(range(first, target_layer + 1)),
            per_head=torch.stack(per_head),
            bypass=torch.stack(bypass),
            total=readout.detach(),
        )
