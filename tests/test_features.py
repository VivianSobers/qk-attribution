"""Tests for reading source directions out of a circuit-tracer graph.

The layout being decoded here is index arithmetic over a tensor of triples, so a stub graph and a
stub transcoder set are enough. Getting the layer filter or the decoder lookup wrong would produce
plausible directions attached to the wrong features, which is the kind of error that survives into
results, so each is tested separately.
"""

from __future__ import annotations

import pytest
import torch

from qk_attribution.features import SourceSet, feature_sources
from tests.stubs import StubGraph, StubTranscoders

D_MODEL, N_LAYERS = 6, 8


def transcoders() -> StubTranscoders:
    return StubTranscoders(n_layers=N_LAYERS, d_model=D_MODEL)


def test_sources_keep_only_features_below_the_layer():
    graph = StubGraph.with_features(
        [(0, 1, 5), (3, 1, 7), (3, 2, 9), (6, 0, 2)], activations=[1.0, 2.0, 3.0, 4.0]
    )
    sources = feature_sources(graph, transcoders(), below_layer=4)
    assert sources.layers.tolist() == [0, 3, 3]
    assert sources.positions.tolist() == [1, 1, 2]
    assert sources.feature_ids.tolist() == [5, 7, 9]
    assert sources.activations.tolist() == [1.0, 2.0, 3.0]
    assert len(sources) == 3


def test_direction_is_the_decoder_row_times_the_activation():
    graph = StubGraph.with_features([(2, 0, 3)], activations=[2.5])
    set_ = transcoders()
    sources = feature_sources(graph, set_, below_layer=5)
    torch.testing.assert_close(sources.directions[0], set_[2].W_dec[3] * 2.5)


def test_directions_come_from_each_features_own_layer():
    """Two features with the same index in different layers must get different directions."""
    graph = StubGraph.with_features([(1, 0, 4), (2, 0, 4)], activations=[1.0, 1.0])
    set_ = transcoders()
    sources = feature_sources(graph, set_, below_layer=5)
    torch.testing.assert_close(sources.directions[0], set_[1].W_dec[4])
    torch.testing.assert_close(sources.directions[1], set_[2].W_dec[4])


def test_only_selected_features_are_read():
    graph = StubGraph.with_features([(0, 0, 1), (1, 0, 2), (2, 0, 3)], activations=[1.0, 2.0, 3.0])
    graph.selected_features = torch.tensor([0, 2])
    graph.activation_values = torch.tensor([1.0, 3.0])
    sources = feature_sources(graph, transcoders(), below_layer=5)
    assert sources.layers.tolist() == [0, 2]


def test_empty_selection_gives_empty_tensors_with_the_right_width():
    graph = StubGraph.with_features([(6, 0, 1)], activations=[1.0])
    sources = feature_sources(graph, transcoders(), below_layer=3)
    assert sources.directions.shape == (0, D_MODEL)
    assert sources.positions.shape == (0,)
    assert len(sources) == 0


def test_below_layer_zero_selects_nothing():
    graph = StubGraph.with_features([(0, 0, 1)], activations=[1.0])
    assert len(feature_sources(graph, transcoders(), below_layer=0)) == 0


def test_negative_below_layer_is_rejected():
    graph = StubGraph.with_features([(0, 0, 1)], activations=[1.0])
    with pytest.raises(ValueError, match="below_layer"):
        feature_sources(graph, transcoders(), below_layer=-1)


def test_malformed_active_features_are_rejected():
    graph = StubGraph.with_features([(0, 0, 1)], activations=[1.0])
    graph.active_features = torch.zeros(1, 2, dtype=torch.int64)
    with pytest.raises(ValueError, match=r"must be \(n, 3\)"):
        feature_sources(graph, transcoders(), below_layer=5)


def test_at_position_selects_one_position():
    graph = StubGraph.with_features([(0, 1, 5), (1, 2, 7), (2, 1, 9)], activations=[1.0, 2.0, 3.0])
    sources = feature_sources(graph, transcoders(), below_layer=5)
    at_one = sources.at_position(1)
    assert isinstance(at_one, SourceSet)
    assert at_one.feature_ids.tolist() == [5, 9]
    assert at_one.positions.tolist() == [1, 1]


def test_at_position_with_no_matches_is_empty():
    graph = StubGraph.with_features([(0, 1, 5)], activations=[1.0])
    sources = feature_sources(graph, transcoders(), below_layer=5)
    assert len(sources.at_position(3)) == 0
