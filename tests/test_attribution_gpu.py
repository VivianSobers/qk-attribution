"""Check the decomposition against real Qwen3 weights and a real attribution graph.

The unit tests prove the contraction is bilinear against a stub. This proves the decomposition is
exhaustive on a real model: transcoder feature directions plus the lumped remainder must reproduce
the attention score the model actually computed. Getting the layernorm gain, the QK-norm scales,
the rotary rotation or the grouped-query mapping wrong breaks this and nothing else would notice.

Needs a CUDA device, a saved graph and about 20 GB of transcoder weights, so it is marked ``gpu``
and ``slow``. Run with ``pytest -m "gpu and slow"``.
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

from qk_attribution.attribution import qk_attribution, residual_remainder  # noqa: E402
from qk_attribution.circuits import kv_head_for  # noqa: E402
from qk_attribution.features import feature_sources, remainder_sources  # noqa: E402
from qk_attribution.scores import (  # noqa: E402
    attention_input,
    attention_scores,
    layernorm_scale,
    qk_norm_scales,
    rotation_matrices,
)

MODEL = "Qwen/Qwen3-0.6B"
GRAPH = os.environ.get("QK_TEST_GRAPH", "spike_out/graph.pt")
SEED = 0
# Measured at 3e-07 to 6e-07 over four heads; see experiments/005-decomposition-completeness.md.
TOLERANCE = 1e-5

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
            dtype=torch.float32,
            lazy_encoder=True,
            lazy_decoder=False,
        )
        model = HookedTransformer.from_pretrained_no_processing(
            MODEL, device="cuda", dtype=torch.float32
        )
        _, cache = model.run_with_cache(graph.input_tokens.unsqueeze(0).cuda())
    return graph, transcoders, model, cache


@pytest.mark.parametrize(("layer", "head"), [(7, 3), (14, 5), (20, 3), (27, 9)])
def test_features_plus_remainder_reproduce_the_model_score(fixtures, layer: int, head: int):
    graph, transcoders, model, cache = fixtures
    with torch.no_grad():
        residual = attention_input(model, cache, layer)
        seq = residual.shape[0]
        query_position = seq - 1
        scales = qk_norm_scales(model, cache, layer)
        assert scales is not None, "Qwen3 uses QK-norm; the scales must be recoverable"
        scale_q = scales[0][:, head]
        scale_k = scales[1][:, kv_head_for(model, head)]
        rotations = rotation_matrices(model, seq)
        truth = attention_scores(
            model,
            layer,
            head,
            residual,
            query_scale=scale_q,
            key_scale=scale_k,
            rotations=rotations,
        )[query_position]

        features = feature_sources(graph, transcoders, below_layer=layer)
        resid_pre = cache[f"blocks.{layer}.hook_resid_pre"][0]
        sources = features.concat(remainder_sources(residual_remainder(resid_pre, features)))
        result = qk_attribution(
            model,
            layer,
            head,
            query_position,
            query_sources=sources.at_position(query_position),
            key_sources=sources,
            rotations=rotations,
            norm_scale=layernorm_scale(cache, layer),
            query_scale=scale_q,
            key_scale=scale_k,
        )
        error = ((result.by_key_position(seq) - truth).norm() / truth.norm()).item()
    assert error < TOLERANCE, f"layer {layer} head {head}: relative error {error:.3e}"
