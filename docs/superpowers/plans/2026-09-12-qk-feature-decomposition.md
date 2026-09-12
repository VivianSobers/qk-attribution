# QK feature decomposition implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose one head's attention score at one query position into query-side by key-side
transcoder feature interactions, with an explicit accounting of what the features do not explain.

**Architecture:** `scores.py` already reconstructs a head's score exactly from the residual stream.
The decomposition substitutes a per-source breakdown of that residual and expands the bilinear form.
Each source direction projects into head space once and is rotated by its position's rotary matrix,
after which the contraction over source pairs is a single matrix product. Sources that are not
transcoder features (earlier attention outputs, transcoder errors, embeddings, decoder biases) are
kept as one lumped direction per position so the parts always sum back to the true score.

**Tech Stack:** PyTorch, TransformerLens 3.2.1, circuit-tracer 0.5.0, pytest, ruff, pyright.

**Spec:** `docs/method.md` for the derivation, `experiments/003-exact-qk-form.md` for the verified
score form, `experiments/004-low-rank-and-cost.md` for the cost budget that sets the scope.

## Global constraints

- Python >= 3.10, ruff line length 100, target py310, lint rules `E,F,I,UP,B,SIM`, pyright standard.
- Never index cached key or value activations by query head. Use `circuits.kv_head_for`.
- Never read `model.W_Q` or any other stacked property. Use `circuits.attention_block`.
- The residual entering attention is `ln1.hook_normalized * ln1.w`, never `hook_normalized` alone.
- Every experiment run logs its seed and full config next to its results.
- Unit tests run on CPU against `tests/stubs.make_model`. Anything needing weights is marked
  `gpu` and `slow`.
- Scope is one `(layer, head, query_position)` at a time. Experiment 004 shows that all-pairs
  decomposition needs 1.5 PFLOP per head at 512 tokens and does not fit in memory.

---

## File structure

- `src/qk_attribution/scores.py` (modify): add the layernorm scale accessor and the residual
  direction to head space projection. These belong here because they are the same algebra as the
  score reconstruction, and splitting them would separate the frozen scales from their derivation.
- `src/qk_attribution/features.py` (create): read a circuit-tracer `Graph` and `TranscoderSet` into
  the per-position source directions the decomposition consumes. Owns everything that knows about
  circuit-tracer's data layout.
- `src/qk_attribution/attribution.py` (create): the contraction itself and its completeness check.
  Owns no knowledge of circuit-tracer or of TransformerLens hook names.
- `tests/stubs.py` (modify): add a stub transcoder set and a stub graph.
- `tests/test_features.py`, `tests/test_attribution.py` (create).
- `experiments/scripts/measure_feature_rank.py` (create).

---

### Task 1: layernorm scale and head-space projection

**Files:**
- Modify: `src/qk_attribution/scores.py`
- Test: `tests/test_scores.py`

**Interfaces:**
- Consumes: `scores.query_projection`, `scores.key_projection`, `scores.rotation_matrices`.
- Produces:
  - `layernorm_scale(cache, layer) -> Tensor` shaped `(seq, 1)`
  - `to_head_space(model, layer, head, directions, positions, *, side, rotations, norm_scale,
    qk_scale) -> Tensor` shaped `(n_directions, d_head)`

- [ ] **Step 1: Write the failing tests**

```python
def test_layernorm_scale_drops_the_batch_axis():
    cache = {"blocks.0.ln1.hook_scale": torch.ones(1, SEQ, 1) * 3.0}
    torch.testing.assert_close(layernorm_scale(cache, 0), torch.ones(SEQ, 1) * 3.0)


def test_to_head_space_matches_projecting_the_residual_directly():
    """A residual built from directions must project to the same head-space vectors."""
    model = rotary_model(use_qk_norm=True)
    resid = residual()
    s_q, s_k = scales(model, 0, 2, resid)
    rotations = rotation_matrices(model, SEQ)
    norm = torch.ones(SEQ, 1)
    # One direction per position, equal to that position's residual.
    positions = torch.arange(SEQ)
    got = to_head_space(
        model, 0, 2, resid, positions,
        side="query", rotations=rotations, norm_scale=norm, qk_scale=s_q,
    )
    expected = torch.einsum(
        "pa,pab->pb", (resid @ query_projection(model, 0, 2)) / s_q, rotations
    )
    torch.testing.assert_close(got, expected)


def test_to_head_space_uses_the_key_projection_for_the_key_side():
    model = make_model()
    resid = residual()
    positions = torch.arange(SEQ)
    got = to_head_space(
        model, 0, 1, resid, positions,
        side="key", rotations=None, norm_scale=torch.ones(SEQ, 1), qk_scale=None,
    )
    torch.testing.assert_close(got, resid @ key_projection(model, 0, 1))


def test_to_head_space_rejects_an_unknown_side():
    with pytest.raises(ValueError, match="side must be"):
        to_head_space(
            make_model(), 0, 0, residual(), torch.arange(SEQ),
            side="value", rotations=None, norm_scale=torch.ones(SEQ, 1), qk_scale=None,
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `scripts/sync-and-check.sh 10.10.3.83 -- -m "not slow" -k "head_space or layernorm_scale"`
Expected: FAIL with `ImportError: cannot import name 'to_head_space'`

- [ ] **Step 3: Write the implementation**

```python
def layernorm_scale(cache: object, layer: int) -> Tensor:
    """Return the RMS scale ln1 divided by, shaped ``(seq, 1)``.

    A feature's decoder direction enters the residual before this division, so mapping it into
    attention requires dividing by the same per-position scalar. It is frozen the way
    circuit-tracer freezes it.
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

    Args:
        directions: Rows in residual-stream coordinates, shaped ``(n, d_model)``, each already
            scaled by its feature's activation.
        positions: Sequence position of each row, shaped ``(n,)``, used to pick the rotation and
            the frozen scales.
        side: ``"query"`` or ``"key"``.
        rotations: From :func:`rotation_matrices`, or None for a model without rotary embeddings.
        norm_scale: Output of :func:`layernorm_scale`, shaped ``(seq, 1)``.
        qk_scale: QK-norm scale for this head, shaped ``(seq, 1)``, or None without QK-norm.

    Returns:
        Tensor of shape ``(n, d_head)``.
    """
    if side not in ("query", "key"):
        raise ValueError(f"side must be 'query' or 'key', got {side!r}")
    if directions.ndim != 2:
        raise ValueError(f"directions must be 2D, got {tuple(directions.shape)}")
    if positions.shape[0] != directions.shape[0]:
        raise ValueError(
            f"{positions.shape[0]} positions for {directions.shape[0]} directions"
        )
    projection = query_projection if side == "query" else key_projection
    out = (directions / norm_scale[positions]) @ projection(model, layer, head)
    if qk_scale is not None:
        out = out / qk_scale[positions]
    if rotations is not None:
        out = torch.einsum("na,nab->nb", out, rotations[positions].to(out.dtype))
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `scripts/sync-and-check.sh 10.10.3.83 -- -m "not slow" -k "head_space or layernorm_scale"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qk_attribution/scores.py tests/test_scores.py
git commit -m "feat: project residual directions into head space"
```

---

### Task 2: read source directions from a graph

**Files:**
- Create: `src/qk_attribution/features.py`
- Modify: `tests/stubs.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `@dataclass(frozen=True) class SourceSet` with fields `directions: Tensor` `(n, d_model)`,
    `positions: Tensor` `(n,)`, `layers: Tensor` `(n,)`, `feature_ids: Tensor` `(n,)`,
    `activations: Tensor` `(n,)`
  - `feature_sources(graph, transcoders, *, below_layer, device=None) -> SourceSet`

- [ ] **Step 1: Write the failing tests**

```python
def test_sources_keep_only_features_below_the_layer():
    graph = StubGraph.with_features(
        [(0, 1, 5), (3, 1, 7), (3, 2, 9), (6, 0, 2)], activations=[1.0, 2.0, 3.0, 4.0]
    )
    sources = feature_sources(graph, StubTranscoders(n_layers=8, d_model=6), below_layer=4)
    assert sources.layers.tolist() == [0, 3, 3]
    assert sources.positions.tolist() == [1, 1, 2]
    assert sources.feature_ids.tolist() == [5, 7, 9]
    assert sources.activations.tolist() == [1.0, 2.0, 3.0]


def test_direction_is_the_decoder_row_times_the_activation():
    graph = StubGraph.with_features([(2, 0, 3)], activations=[2.5])
    transcoders = StubTranscoders(n_layers=8, d_model=6)
    sources = feature_sources(graph, transcoders, below_layer=5)
    expected = transcoders[2].W_dec[3] * 2.5
    torch.testing.assert_close(sources.directions[0], expected)


def test_empty_selection_gives_empty_tensors_with_the_right_width():
    graph = StubGraph.with_features([(6, 0, 1)], activations=[1.0])
    sources = feature_sources(graph, StubTranscoders(n_layers=8, d_model=6), below_layer=3)
    assert sources.directions.shape == (0, 6)
    assert sources.positions.shape == (0,)


def test_below_layer_zero_selects_nothing():
    graph = StubGraph.with_features([(0, 0, 1)], activations=[1.0])
    sources = feature_sources(graph, StubTranscoders(n_layers=8, d_model=6), below_layer=0)
    assert sources.directions.shape[0] == 0
```

- [ ] **Step 2: Add the stubs these tests need**

```python
class StubTranscoders:
    """A per-layer transcoder set exposing only the decoder rows."""

    def __init__(self, n_layers: int, d_model: int, d_transcoder: int = 16, seed: int = 3) -> None:
        gen = torch.Generator().manual_seed(seed)
        self._layers = [
            SimpleNamespace(W_dec=torch.randn(d_transcoder, d_model, generator=gen))
            for _ in range(n_layers)
        ]

    def __len__(self) -> int:
        return len(self._layers)

    def __getitem__(self, layer: int) -> SimpleNamespace:
        return self._layers[layer]


@dataclass
class StubGraph:
    """A circuit-tracer Graph stand-in carrying only the fields the library reads."""

    active_features: torch.Tensor
    selected_features: torch.Tensor
    activation_values: torch.Tensor
    input_tokens: torch.Tensor

    @classmethod
    def with_features(
        cls, triples: list[tuple[int, int, int]], activations: list[float], n_pos: int = 4
    ) -> StubGraph:
        active = torch.tensor(triples, dtype=torch.int64)
        return cls(
            active_features=active,
            selected_features=torch.arange(len(triples)),
            activation_values=torch.tensor(activations),
            input_tokens=torch.zeros(n_pos, dtype=torch.int64),
        )
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `scripts/sync-and-check.sh 10.10.3.83 -- -m "not slow" tests/test_features.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'qk_attribution.features'`

- [ ] **Step 4: Write the implementation**

```python
"""Reading source directions for the QK decomposition out of a circuit-tracer graph.

A graph's ``active_features`` is an ``(n, 3)`` tensor of ``(layer, pos, feature_idx)``, and
``selected_features`` indexes into it. A per-layer transcoder feature writes its decoder row into
the residual stream at its own layer's MLP output, so it is visible to attention at every later
layer and at no earlier one.

This module is the only place that knows circuit-tracer's layout. Everything downstream takes a
:class:`SourceSet`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

LAYER, POSITION, FEATURE = 0, 1, 2


@dataclass(frozen=True)
class SourceSet:
    """Residual-stream directions that feed one attention layer, with their provenance."""

    directions: Tensor
    positions: Tensor
    layers: Tensor
    feature_ids: Tensor
    activations: Tensor

    def __len__(self) -> int:
        return int(self.directions.shape[0])


def feature_sources(
    graph: Any, transcoders: Any, *, below_layer: int, device: torch.device | None = None
) -> SourceSet:
    """Collect the activation-scaled decoder directions visible to attention at ``below_layer``.

    Args:
        graph: A circuit-tracer ``Graph``.
        transcoders: A ``TranscoderSet``, indexed by layer.
        below_layer: Only features written at a strictly earlier layer are included.
        device: Where to build the result. Defaults to the decoder weights' device.

    Returns:
        A :class:`SourceSet`, empty but correctly shaped when nothing qualifies.
    """
    selected = graph.selected_features
    active = graph.active_features[selected]
    keep = active[:, LAYER] < below_layer
    layers = active[keep, LAYER]
    positions = active[keep, POSITION]
    feature_ids = active[keep, FEATURE]
    activations = graph.activation_values[keep]

    template = transcoders[0].W_dec
    device = device or template.device
    directions = torch.empty(
        (int(keep.sum()), template.shape[1]), dtype=template.dtype, device=device
    )
    for row, (layer, feature) in enumerate(zip(layers.tolist(), feature_ids.tolist())):
        directions[row] = transcoders[layer].W_dec[feature].to(device)
    directions = directions * activations.to(device).unsqueeze(-1)

    return SourceSet(
        directions=directions,
        positions=positions.to(device),
        layers=layers.to(device),
        feature_ids=feature_ids.to(device),
        activations=activations.to(device),
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `scripts/sync-and-check.sh 10.10.3.83 -- -m "not slow" tests/test_features.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/qk_attribution/features.py tests/test_features.py tests/stubs.py
git commit -m "feat: read feature directions from graph"
```

---

### Task 3: the feature-pair contraction

**Files:**
- Create: `src/qk_attribution/attribution.py`
- Test: `tests/test_attribution.py`

**Interfaces:**
- Consumes: `scores.to_head_space`, `scores.rotation_matrices`, `scores.layernorm_scale`,
  `features.SourceSet`.
- Produces:
  - `@dataclass(frozen=True) class QKAttribution` with fields `contributions: Tensor`
    `(n_query_sources, n_key_sources)`, `query_sources: SourceSet`, `key_sources: SourceSet`,
    `query_position: int`, `key_positions: Tensor`, `explained: Tensor`, `total: Tensor`
  - `qk_attribution(model, layer, head, query_position, query_sources, key_sources, *, rotations,
    norm_scale, query_scale, key_scale, attention_scale) -> QKAttribution`

- [ ] **Step 1: Write the failing tests**

```python
def test_contributions_sum_to_the_bilinear_score_of_the_summed_directions():
    """Bilinearity is the whole method: the parts must add up to the whole."""
    model = rotary_model(use_qk_norm=True)
    query_sources, key_sources = paired_sources(model)
    result = qk_attribution(
        model, 0, 1, query_position=2,
        query_sources=query_sources, key_sources=key_sources,
        rotations=rotation_matrices(model, SEQ), norm_scale=torch.ones(SEQ, 1),
        query_scale=torch.ones(SEQ, 1), key_scale=torch.ones(SEQ, 1),
    )
    rebuilt = build_residual(query_sources, key_sources)
    direct = attention_scores(model, 0, 1, rebuilt, query_scale=torch.ones(SEQ, 1),
                              key_scale=torch.ones(SEQ, 1))
    torch.testing.assert_close(result.contributions.sum(), direct[2].sum(), rtol=1e-4, atol=1e-5)


def test_contribution_matrix_has_one_row_per_query_source():
    model = make_model()
    query_sources, key_sources = paired_sources(model)
    result = qk_attribution(
        model, 0, 0, query_position=1,
        query_sources=query_sources, key_sources=key_sources,
        rotations=None, norm_scale=torch.ones(SEQ, 1), query_scale=None, key_scale=None,
    )
    assert result.contributions.shape == (len(query_sources), len(key_sources))


def test_a_zero_activation_feature_contributes_nothing():
    model = make_model()
    query_sources, key_sources = paired_sources(model, zero_first_query=True)
    result = qk_attribution(
        model, 0, 0, query_position=1,
        query_sources=query_sources, key_sources=key_sources,
        rotations=None, norm_scale=torch.ones(SEQ, 1), query_scale=None, key_scale=None,
    )
    torch.testing.assert_close(result.contributions[0], torch.zeros(len(key_sources)))


def test_query_sources_at_other_positions_are_rejected():
    model = make_model()
    query_sources, key_sources = paired_sources(model)
    with pytest.raises(ValueError, match="query_position"):
        qk_attribution(
            model, 0, 0, query_position=99,
            query_sources=query_sources, key_sources=key_sources,
            rotations=None, norm_scale=torch.ones(SEQ, 1), query_scale=None, key_scale=None,
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `scripts/sync-and-check.sh 10.10.3.83 -- -m "not slow" tests/test_attribution.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'qk_attribution.attribution'`

- [ ] **Step 3: Write the implementation**

```python
"""Decomposing one head's attention score into query-side by key-side source interactions.

The score is bilinear in the residual stream, so substituting a per-source breakdown at the query
and key positions expands it into one term per source pair. Both sides project into head space
once, and the contraction is then a single matrix product.

Scope is one query position and one head. Experiment 004 records why: over all position pairs the
contraction reaches 1.5 PFLOP per head at 512 tokens and the intermediate does not fit in memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from qk_attribution.circuits import attention_scale as head_attention_scale
from qk_attribution.features import SourceSet
from qk_attribution.scores import to_head_space


@dataclass(frozen=True)
class QKAttribution:
    """Per-source-pair contributions to one head's attention scores from one query position."""

    contributions: Tensor
    query_sources: SourceSet
    key_sources: SourceSet
    query_position: int

    @property
    def explained(self) -> Tensor:
        """Total contribution to each key position, summed over both source axes."""
        return self.contributions.sum(dim=0)


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
    """Expand the attention score into source-pair terms for one query position.

    Args:
        query_sources: Sources at ``query_position`` only.
        key_sources: Sources at any position the query may attend to.
        rotations: From ``scores.rotation_matrices``, or None without rotary embeddings.
        norm_scale: From ``scores.layernorm_scale``.
        query_scale: QK-norm scale for this head, or None.
        key_scale: QK-norm scale for this head's key group, or None.

    Returns:
        A :class:`QKAttribution` whose ``contributions`` sum to the score contributed by the
        sources supplied, which is the full score only when the sources reconstruct the residual.
    """
    if len(query_sources) and int(query_sources.positions.min()) != query_position:
        raise ValueError(
            "every query source must sit at query_position; got positions "
            f"{query_sources.positions.unique().tolist()} for query_position {query_position}"
        )
    left = to_head_space(
        model, layer, head, query_sources.directions, query_sources.positions,
        side="query", rotations=rotations, norm_scale=norm_scale, qk_scale=query_scale,
    )
    right = to_head_space(
        model, layer, head, key_sources.directions, key_sources.positions,
        side="key", rotations=rotations, norm_scale=norm_scale, qk_scale=key_scale,
    )
    contributions = left @ right.transpose(-1, -2) / head_attention_scale(model)
    return QKAttribution(
        contributions=contributions,
        query_sources=query_sources,
        key_sources=key_sources,
        query_position=query_position,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `scripts/sync-and-check.sh 10.10.3.83 -- -m "not slow" tests/test_attribution.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qk_attribution/attribution.py tests/test_attribution.py
git commit -m "feat: expand scores into source pair terms"
```

---

### Task 4: completeness against a real model

**Files:**
- Create: `tests/test_attribution_gpu.py`

**Interfaces:**
- Consumes: everything from Tasks 1 to 3.
- Produces: nothing. This task exists to prove the decomposition is exhaustive.

The unit tests prove the contraction is bilinear. They cannot prove the source set reconstructs the
residual, because the stub's residual is whatever the test builds. On a real model the residual at
an attention layer is the embedding plus every earlier attention output plus every earlier MLP
output, and only the last of those decomposes into transcoder features. The gap is real and must be
measured rather than assumed away.

- [ ] **Step 1: Write the test**

```python
"""Measure how much of a real attention score the feature decomposition accounts for."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("circuit_tracer")

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

GRAPH = "spike_out/graph.pt"


def test_feature_terms_account_for_part_of_the_score_and_the_rest_is_named():
    """The decomposition must be exhaustive: features plus remainder equals the true score."""
    from circuit_tracer.graph import Graph

    from qk_attribution.attribution import qk_attribution
    from qk_attribution.circuits import kv_head_for
    from qk_attribution.features import feature_sources
    from qk_attribution.scores import (
        attention_input,
        attention_scores,
        layernorm_scale,
        qk_norm_scales,
        rotation_matrices,
    )

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    graph = Graph.from_pt(GRAPH)
    model, transcoders = load_model_and_transcoders()

    layer, head, query_position = 20, 3, 8
    with torch.no_grad():
        _, cache = model.run_with_cache(graph.input_tokens.unsqueeze(0).cuda())
        residual = attention_input(model, cache, layer)
        seq = residual.shape[0]
        scales = qk_norm_scales(model, cache, layer)
        assert scales is not None
        rotations = rotation_matrices(model, seq)
        truth = attention_scores(
            model, layer, head, residual,
            query_scale=scales[0][:, head],
            key_scale=scales[1][:, kv_head_for(model, head)],
            rotations=rotations,
        )[query_position]

        sources = feature_sources(graph, transcoders, below_layer=layer)
        at_query = select_positions(sources, [query_position])
        result = qk_attribution(
            model, layer, head, query_position,
            query_sources=at_query, key_sources=sources,
            rotations=rotations, norm_scale=layernorm_scale(cache, layer),
            query_scale=scales[0][:, head], key_scale=scales[1][:, kv_head_for(model, head)],
        )

    explained = torch.zeros_like(truth)
    explained.index_add_(0, sources.positions, result.contributions.sum(dim=0))
    fraction = (explained.norm() / truth.norm()).item()
    print(f"feature terms carry {fraction:.3f} of the score norm at layer {layer} head {head}")
    assert 0.0 < fraction, "features must carry some of the score"
    assert torch.isfinite(explained).all()
```

- [ ] **Step 2: Run the test**

Run: `scripts/sync-and-check.sh 10.10.3.83 -- -m "gpu and slow" tests/test_attribution_gpu.py -s`
Expected: PASS, and the printed fraction recorded in the experiment note. A fraction well below 1
is the expected outcome, because earlier attention outputs are not transcoder features. Do not
adjust the test to hide it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_attribution_gpu.py
git commit -m "test: measure decomposition completeness on Qwen3"
```

---

### Task 5: rank of the feature-space attribution matrix

**Files:**
- Create: `experiments/scripts/measure_feature_rank.py`
- Create: `experiments/005-feature-space-rank.md`

**Interfaces:**
- Consumes: Tasks 1 to 3.
- Produces: the measurement experiment 004 identified as missing.

Experiment 004 ruled out low-rank structure in the weights and found only a factor of two in the
score matrix. The object Anthropic's remark is about is the feature-pair matrix, which has not been
measured. This task measures it.

- [ ] **Step 1: Write the script**

```python
"""Measure the rank of the feature-pair contribution matrix, the object the low-rank claim is about.

For several (layer, head, query position) triples, build the contribution matrix and report its
effective rank at 0.9, 0.99 and 0.999 of squared energy, along with the relative error of its own
optimal rank-r approximation. Truncating the SVD of the target is the best any method can do, so
these errors are a lower bound on what an implementation would achieve.
"""
```

The script loads the saved graph and transcoders, loops over the triples, calls `qk_attribution`,
takes `torch.linalg.svdvals` of `result.contributions`, and calls `circuits.effective_rank` at each
energy. It writes `experiments/results/005-feature-rank-qwen3-0.6b.json` with the seed, the model
name, the triples, the spectra and the truncation errors, exactly as `measure_score_rank.py` does.

- [ ] **Step 2: Run it and record the numbers**

Run: `ssh ccbd@10.10.3.83 'cd ~/qk-attribution && . .venv/bin/activate && python experiments/scripts/measure_feature_rank.py'`
Expected: a JSON file and a printed summary. Report whatever comes out. If the matrices are not low
rank, say so; experiment 004 already records two negative results on this question and a third is
as useful as a positive one.

- [ ] **Step 3: Write the experiment note**

Follow the shape of `experiments/004-low-rank-and-cost.md`: date, machine, script path, raw output
path, config with seed, the question, the tables, and what it settles. Run the `humanizer` skill
over the prose.

- [ ] **Step 4: Commit**

```bash
git add experiments/scripts/measure_feature_rank.py experiments/results/005-feature-rank-qwen3-0.6b.json experiments/005-feature-space-rank.md
git commit -m "exp: measure feature space attribution rank"
```

---

## Out of scope

- Decomposition over all position pairs. Experiment 004 gives the arithmetic.
- A Triton kernel. The contraction is a dense matrix product that cuBLAS already runs near peak.
  A kernel is worth writing only to fuse the contraction with a top-k reduction so the
  `n_query_sources` by `n_key_sources` matrix never reaches HBM, and only once a measurement shows
  that materialisation is the bottleneck.
- Attributing through earlier attention outputs, which is what would close the completeness gap
  Task 4 measures.
- Soft-capped models. `scores.require_supported` refuses them.
