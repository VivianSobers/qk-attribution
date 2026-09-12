"""Check edge effects against circuit-tracer's own adjacency matrix.

This is the strongest check in the project: the adjacency was produced by a separate
implementation, through backward passes rather than forward propagation, in a separate process.
Agreement means the frozen-attention model here is the same one attribution graphs are built under.

Needs a CUDA device, a saved graph and the transcoder weights, so it is marked ``gpu`` and
``slow``.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("circuit_tracer")
pytest.importorskip("transformer_lens")

from circuit_tracer.graph import Graph  # noqa: E402
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402

from qk_attribution.loadings import edge_effect, edge_loadings  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
GRAPH = os.environ.get("QK_TEST_GRAPH", "spike_out/graph.pt")
SEED = 0
N_EDGES = 8
# The graph was built in bfloat16 and is recomputed here in float32, so the two agree to a few
# percent rather than exactly. See experiments/007-head-loadings.md for why this is the right
# tolerance and not a hidden disagreement.
RATIO_TOLERANCE = 0.10
PARTITION_TOLERANCE = 1e-4

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def fixtures():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if not os.path.exists(GRAPH):
        pytest.skip(f"no attribution graph at {GRAPH}")
    torch.manual_seed(SEED)
    with torch.no_grad():
        graph = Graph.from_pt(GRAPH)
        assert isinstance(graph.scan, str), "a multi-scan graph names several transcoder sets"
        transcoders, _ = load_transcoder_from_hub(
            graph.scan,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            lazy_encoder=True,
            lazy_decoder=True,
        )
        model = HookedTransformer.from_pretrained_no_processing(
            MODEL, device="cuda", dtype=torch.float32
        )
        _, cache = model.run_with_cache(graph.input_tokens.unsqueeze(0).cuda())
    return graph, transcoders, model, cache


def strongest_edges(graph, count: int):
    """The strongest feature edges that both span a layer and cross a position."""
    n_features = len(graph.selected_features)
    active = graph.active_features[graph.selected_features].cuda()
    block = graph.adjacency_matrix.cuda().float()[:n_features, :n_features]
    layers, positions = active[:, 0], active[:, 1]
    mask = (layers[:, None] > layers[None, :]) & (positions[:, None] != positions[None, :])
    flat = (block.abs() * mask).flatten().topk(count).indices
    for index in flat.tolist():
        target_index, source_index = divmod(index, n_features)
        yield (
            active[source_index].tolist(),
            active[target_index].tolist(),
            graph.activation_values[source_index].item(),
            block[target_index, source_index].item(),
        )


def test_edge_effects_reproduce_the_graph_adjacency(fixtures):
    graph, transcoders, model, cache = fixtures
    ratios = []
    with torch.no_grad():
        for source_node, target_node, activation, edge in strongest_edges(graph, N_EDGES):
            source_layer, source_position, source_feature = source_node
            target_layer, target_position, target_feature = target_node
            source = transcoders[source_layer].W_dec[source_feature].float() * activation
            reader = transcoders[target_layer].W_enc[target_feature].float()
            effect = edge_effect(
                model,
                cache,
                source,
                source_position,
                source_layer,
                reader,
                target_position,
                target_layer,
            )
            ratios.append(edge / effect.item())
    worst = max(abs(ratio - 1.0) for ratio in ratios)
    assert worst < RATIO_TOLERANCE, f"edge to effect ratios deviate by up to {worst:.3f}"


def test_head_loadings_partition_a_real_edge(fixtures):
    graph, transcoders, model, cache = fixtures
    with torch.no_grad():
        for source_node, target_node, activation, _ in strongest_edges(graph, 2):
            source_layer, source_position, source_feature = source_node
            target_layer, target_position, target_feature = target_node
            source = transcoders[source_layer].W_dec[source_feature].float() * activation
            reader = transcoders[target_layer].W_enc[target_feature].float()
            effect = edge_effect(
                model,
                cache,
                source,
                source_position,
                source_layer,
                reader,
                target_position,
                target_layer,
            )
            for attention_layer in range(source_layer + 1, target_layer + 1):
                loading = edge_loadings(
                    model,
                    cache,
                    source,
                    source_position,
                    source_layer,
                    reader,
                    target_position,
                    target_layer,
                    attention_layer,
                )
                error = ((loading.total - effect).abs() / effect.abs()).item()
                assert error < PARTITION_TOLERANCE, (
                    f"split at layer {attention_layer} lost {error:.3e} of the edge"
                )
