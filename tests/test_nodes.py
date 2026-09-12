"""Tests for adjacency node layout decoding.

These run without a GPU or any downloaded weights: the layout is pure index arithmetic, so a
stub standing in for a circuit-tracer ``Graph`` is enough.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from qk_attribution.nodes import NodeKind, NodeLayout


@dataclass
class StubGraph:
    """Minimal stand-in exposing only what NodeLayout.from_graph reads."""

    selected_features: list[int]
    input_tokens: torch.Tensor
    logit_targets: list[object]
    adjacency_matrix: torch.Tensor

    @classmethod
    def build(cls, n_features: int, n_layers: int, n_pos: int, n_logits: int) -> StubGraph:
        total = n_features + n_layers * n_pos + n_pos + n_logits
        return cls(
            selected_features=list(range(n_features)),
            input_tokens=torch.zeros(n_pos, dtype=torch.int64),
            logit_targets=[object()] * n_logits,
            adjacency_matrix=torch.zeros(total, total),
        )


# Values measured from the Qwen3-0.6B feasibility run; see experiments/001.
SPIKE = dict(n_features=3165, n_layers=28, n_pos=9, n_logits=10)


def test_derives_spike_layout_from_graph():
    layout = NodeLayout.from_graph(StubGraph.build(**SPIKE))
    assert layout == NodeLayout(**SPIKE)
    assert layout.total == 3436


def test_region_boundaries_are_contiguous_and_exhaustive():
    layout = NodeLayout(**SPIKE)
    assert layout.feature_end == 3165
    assert layout.error_end == 3165 + 28 * 9
    assert layout.token_end == layout.error_end + 9
    assert layout.total == layout.token_end + 10


def test_kind_at_every_region_boundary():
    layout = NodeLayout(**SPIKE)
    cases = [
        (0, NodeKind.FEATURE),
        (layout.feature_end - 1, NodeKind.FEATURE),
        (layout.feature_end, NodeKind.ERROR),
        (layout.error_end - 1, NodeKind.ERROR),
        (layout.error_end, NodeKind.TOKEN),
        (layout.token_end - 1, NodeKind.TOKEN),
        (layout.token_end, NodeKind.LOGIT),
        (layout.total - 1, NodeKind.LOGIT),
    ]
    for index, expected in cases:
        assert layout.kind(index) is expected, f"index {index}"


def test_every_index_classifies_without_gaps():
    layout = NodeLayout(n_features=4, n_layers=3, n_pos=2, n_logits=2)
    counts = {kind: 0 for kind in NodeKind}
    for index in range(layout.total):
        counts[layout.kind(index)] += 1
    assert counts[NodeKind.FEATURE] == 4
    assert counts[NodeKind.ERROR] == 6
    assert counts[NodeKind.TOKEN] == 2
    assert counts[NodeKind.LOGIT] == 2


def test_error_nodes_decode_layer_major():
    layout = NodeLayout(n_features=0, n_layers=3, n_pos=4, n_logits=0)
    # Layer-major means the first n_pos error nodes all belong to layer 0.
    assert layout.decode_error(0) == (0, 0)
    assert layout.decode_error(3) == (0, 3)
    assert layout.decode_error(4) == (1, 0)
    assert layout.decode_error(11) == (2, 3)


def test_error_decoding_is_a_bijection():
    layout = NodeLayout(n_features=7, n_layers=5, n_pos=3, n_logits=1)
    seen = set()
    for index in range(layout.feature_end, layout.error_end):
        pair = layout.decode_error(index)
        assert pair not in seen
        seen.add(pair)
    assert len(seen) == 5 * 3


def test_token_and_logit_decode_to_zero_based_offsets():
    layout = NodeLayout(**SPIKE)
    assert layout.decode_token(layout.error_end) == 0
    assert layout.decode_token(layout.token_end - 1) == SPIKE["n_pos"] - 1
    assert layout.decode_logit(layout.token_end) == 0
    assert layout.decode_logit(layout.total - 1) == SPIKE["n_logits"] - 1


@pytest.mark.parametrize("index", [-1, 3436, 9999])
def test_out_of_range_indices_raise(index: int):
    layout = NodeLayout(**SPIKE)
    with pytest.raises(IndexError):
        layout.kind(index)


def test_decoding_the_wrong_region_raises():
    layout = NodeLayout(**SPIKE)
    with pytest.raises(ValueError, match="not a error node"):
        layout.decode_error(0)
    with pytest.raises(ValueError, match="not a token node"):
        layout.decode_token(0)
    with pytest.raises(ValueError, match="not a feature node"):
        layout.decode_feature(layout.total - 1)


def test_inconsistent_adjacency_size_is_rejected():
    graph = StubGraph.build(**SPIKE)
    # Drop one row and column so the error region is no longer a multiple of n_pos.
    graph.adjacency_matrix = torch.zeros(3435, 3435)
    with pytest.raises(ValueError, match="not a multiple of n_pos"):
        NodeLayout.from_graph(graph)


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError, match="n_pos must be positive"):
        NodeLayout(n_features=1, n_layers=1, n_pos=0, n_logits=1)
