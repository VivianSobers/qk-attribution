"""Check the reconstructed score form against real Qwen3-0.6B weights.

The unit tests prove the rearrangement is internally right. This proves the model is what we think
it is: rotary convention, QK-norm placement, grouped-query indexing, and the layernorm gain all
have to be correct simultaneously or the scores will not match.

Needs a CUDA device and downloads about 1.5 GB of weights, so it is marked ``gpu`` and ``slow``.
Run with ``pytest -m "gpu and slow"``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")

from transformer_lens import HookedTransformer  # noqa: E402

from qk_attribution.circuits import kv_head_for, n_heads  # noqa: E402
from qk_attribution.scores import (  # noqa: E402
    attention_input,
    attention_scores,
    qk_norm_scales,
    rotation_matrices,
)

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The Eiffel Tower is located in the city of Paris, the capital of France."
SEED = 0
# Measured over all 448 (layer, head) pairs in float32; see experiments/003-exact-qk-form.md.
TOLERANCE = 1e-5

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def model_and_cache():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(SEED)
    with torch.no_grad():
        model = HookedTransformer.from_pretrained_no_processing(
            MODEL, device="cuda", dtype=torch.float32
        )
        _, cache = model.run_with_cache(model.to_tokens(PROMPT))
    return model, cache


@pytest.mark.parametrize("layer", [0, 13, 27])
def test_reconstruction_matches_model_scores(model_and_cache, layer: int):
    model, cache = model_and_cache
    with torch.no_grad():
        residual = attention_input(model, cache, layer)
        seq = residual.shape[0]
        scales = qk_norm_scales(model, cache, layer)
        assert scales is not None, "Qwen3 uses QK-norm; the scales must be recoverable"
        scale_q, scale_k = scales
        rotations = rotation_matrices(model, seq)
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=residual.device))
        truth = cache[f"blocks.{layer}.attn.hook_attn_scores"][0]

        for head in range(n_heads(model)):
            got = attention_scores(
                model,
                layer,
                head,
                residual,
                query_scale=scale_q[:, head],
                key_scale=scale_k[:, kv_head_for(model, head)],
                rotations=rotations,
            )
            want = truth[head]
            error = (got[causal] - want[causal]).norm() / want[causal].norm()
            assert error < TOLERANCE, f"layer {layer} head {head}: relative error {error:.3e}"
