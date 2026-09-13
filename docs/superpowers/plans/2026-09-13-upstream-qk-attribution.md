# Upstream QK Attribution Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `circuit_tracer/attribution/qk_attribution.py` to the circuit-tracer fork, decomposing one head's pre-softmax attention score into (query source, key source) pairs drawn from an attribution graph, with the pairs summing to the score exactly.

**Architecture:** One module beside `head_loadings.py`. A `FrozenScores` object captures, from one clean run, what a graph holds fixed: the residual entering each block, the `ln1` scales, the QK-norm scales, and the rotary operator. Sources are the graph's transcoder features below the layer plus one remainder per position, so the expansion is exhaustive. Every source projects into head space once, and the contraction is a single matrix product. The module imports `Graph` only for type checking and detects cross-layer transcoders structurally, so it imports with torch alone and its tests run without transformer_lens or pydantic.

**Tech Stack:** Python 3.10+, PyTorch, pytest, ruff, pyright (basic).

**Spec:** `docs/method.md` in qk-attribution, with the verified behaviour it must reproduce in `experiments/003-exact-qk-form.md` (exact score form, 2.5e-07), `experiments/005-decomposition-completeness.md` (exhaustive with the remainder, 3e-07) and `experiments/011-intervening-on-the-attribution.md`. The source being ported is `src/qk_attribution/{scores,features,attribution}.py`.

## Global Constraints

- Work in `~/Documents/circuit-tracer` on branch `head-loadings`.
- ruff: line length 100, rules E, F, TID; `typing.Optional`, `Union`, `Dict`, `Tuple`, `List` are banned.
- pyright in basic mode must pass on the new files.
- The module must import with torch alone: `Graph` under `TYPE_CHECKING`, no import of `CrossLayerTranscoder`.
- Feature directions are extracted in float32 whatever the transcoder storage dtype.
- `graph.activation_values` is aligned with `active_features`, not `selected_features`.
- Commit messages: a `feat:`, `fix:`, `refactor:`, `perf:`, `docs:`, `test:` or `exp:` prefix, then 4 to 6 words. No trailers, no co-author lines, no mention of Claude in code, comments, messages or docs.
- One commit per logical change, each in a working state.
- Each commit passes `ruff check` alone, so a task adds only the imports it first uses.
- Never pipe ruff or pytest before `&&` when gating a commit; the pipe hides the exit code.
- Run tests with `cd ~/Documents/circuit-tracer && python3 -m pytest tests/test_qk_attribution.py -q`.

## File Structure

- Create `circuit_tracer/attribution/qk_attribution.py`: guards, frozen scales, head-space projection, graph sources, the contraction. One responsibility (decomposing a score), built up section by section across tasks.
- Create `tests/test_qk_attribution.py`: a stub model with a real rotary implementation, QK-norm and grouped-query keys, plus an independent reference forward pass. Imports the module as `qk` once at the top, so later tasks append tests without mid-file imports.
- Modify `circuit_tracer/__init__.py`: export `qk_attribution` and `FrozenScores` lazily, beside `head_loadings`.

---

### Task 1: Architecture guards and the attention scale

**Files:**
- Create: `circuit_tracer/attribution/qk_attribution.py`
- Create: `tests/test_qk_attribution.py`

**Interfaces:**
- Produces: `qk.UnsupportedForQK(RuntimeError)`, `qk.require_supported(model) -> None`, `qk.attention_scale(model) -> float`, `qk.REMAINDER == -1`, `qk.BILINEAR_POSITION_SCHEMES`.
- Produces (tests): `make_model(...)`, `reference_cache(model)`, `reference_scores(model, layer, head)`, `make_graph(...)`, constants `N_LAYERS, N_HEADS, D_MODEL, D_HEAD, N_POS, D_TRANSCODER`.

- [ ] **Step 1: Write the test file with its stubs and the failing guard tests**

```python
"""Tests for decomposing an attention score into pairs of attribution-graph sources.

A stub stands in for a ReplacementModel: random weights, a genuine rotary implementation, QK-norm
gains and grouped-query keys. ``reference_scores`` computes scores the way a forward pass does
(project, normalise, rotate, dot) without touching the module, so agreement with it is evidence
rather than a restatement.
"""

from types import SimpleNamespace

import pytest
import torch

from circuit_tracer.attribution import qk_attribution as qk

N_LAYERS, N_HEADS, D_MODEL, D_HEAD, N_POS, D_TRANSCODER = 3, 4, 8, 4, 6, 10


class StubAttention(SimpleNamespace):
    def apply_rotary(self, x, past_kv_pos_offset=0, attention_mask=None):
        """Rotate ``[batch, pos, head, d_head]`` by a per-position angle, pairing adjacent dims."""
        positions = torch.arange(x.shape[1], dtype=torch.float32) + past_kv_pos_offset
        angle = positions[:, None] * self.freqs[None, :]
        cos, sin = angle.cos()[None, :, None, :], angle.sin()[None, :, None, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out


class StubTranscoders(list):
    """Indexable by layer, as a per-layer TranscoderSet is."""


def rms(x: torch.Tensor) -> torch.Tensor:
    return (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()


def make_model(
    *,
    rotary: bool = True,
    qk_norm: bool = True,
    kv_heads: int = 2,
    soft_cap: float = -1.0,
    scheme: str | None = None,
    ln1_bias: bool = False,
    attn_bias: bool = False,
    use_attn_scale: bool = True,
    seed: int = 0,
) -> SimpleNamespace:
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    group = N_HEADS // kv_heads
    freqs = torch.rand(D_HEAD // 2, generator=generator)
    blocks = []
    for _ in range(N_LAYERS):
        attn = StubAttention(
            W_Q=randn(N_HEADS, D_MODEL, D_HEAD),
            W_K=randn(kv_heads, D_MODEL, D_HEAD).repeat_interleave(group, dim=0),
            b_Q=randn(N_HEADS, D_HEAD) if attn_bias else torch.zeros(N_HEADS, D_HEAD),
            b_K=torch.zeros(N_HEADS, D_HEAD),
            freqs=freqs,
        )
        if qk_norm:
            attn.q_norm = SimpleNamespace(w=torch.rand(D_HEAD, generator=generator) + 0.5)
            attn.k_norm = SimpleNamespace(w=torch.rand(D_HEAD, generator=generator) + 0.5)
        ln1 = SimpleNamespace(
            w=torch.rand(D_MODEL, generator=generator) + 0.5,
            b=randn(D_MODEL) if ln1_bias else None,
        )
        blocks.append(SimpleNamespace(attn=attn, ln1=ln1))
    cfg = SimpleNamespace(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_key_value_heads=kv_heads,
        positional_embedding_type=scheme or ("rotary" if rotary else "standard"),
        attn_scores_soft_cap=soft_cap,
        attn_scale=None,
        use_attn_scale=use_attn_scale,
    )
    model = SimpleNamespace(
        cfg=cfg,
        blocks=blocks,
        transcoders=StubTranscoders(
            SimpleNamespace(W_dec=randn(D_TRANSCODER, D_MODEL)) for _ in range(N_LAYERS)
        ),
        residuals=[randn(N_POS, D_MODEL) for _ in range(N_LAYERS)],
    )
    model.run_with_cache = lambda tokens, names_filter=None: (None, reference_cache(model))
    return model


def attention_in(model: SimpleNamespace, layer: int) -> torch.Tensor:
    x = model.residuals[layer]
    return x / rms(x) * model.blocks[layer].ln1.w


def reference_cache(model: SimpleNamespace) -> dict:
    """What a single-prompt run caches, with TransformerLens's shapes."""
    cache = {}
    group = N_HEADS // model.cfg.n_key_value_heads
    for layer, block in enumerate(model.blocks):
        x = model.residuals[layer]
        cache[f"blocks.{layer}.hook_resid_pre"] = x[None]
        cache[f"blocks.{layer}.ln1.hook_scale"] = rms(x)[None]
        if getattr(block.attn, "q_norm", None) is not None:
            inp = attention_in(model, layer)
            q = torch.einsum("pd,hde->phe", inp, block.attn.W_Q)
            k = torch.einsum("pd,hde->phe", inp, block.attn.W_K[::group])
            # TransformerLens flattens these over (batch, pos, head).
            cache[f"blocks.{layer}.attn.q_norm.hook_scale"] = rms(q).reshape(-1, 1)
            cache[f"blocks.{layer}.attn.k_norm.hook_scale"] = rms(k).reshape(-1, 1)
    return cache


def reference_scores(model: SimpleNamespace, layer: int, head: int) -> torch.Tensor:
    """Project, normalise, rotate, dot: the order a forward pass uses."""
    attn = model.blocks[layer].attn
    inp = attention_in(model, layer)
    q, k = inp @ attn.W_Q[head], inp @ attn.W_K[head]
    if getattr(attn, "q_norm", None) is not None:
        q, k = q / rms(q) * attn.q_norm.w, k / rms(k) * attn.k_norm.w
    if model.cfg.positional_embedding_type == "rotary":
        q = attn.apply_rotary(q[None, :, None, :])[0, :, 0]
        k = attn.apply_rotary(k[None, :, None, :])[0, :, 0]
    scale = 1.0 if model.cfg.use_attn_scale is False else D_HEAD**0.5
    return q @ k.T / scale


def make_graph(active, selected, activations) -> SimpleNamespace:
    """Only the fields the module reads from a circuit-tracer Graph."""
    return SimpleNamespace(
        active_features=torch.tensor(active),
        selected_features=torch.tensor(selected),
        activation_values=torch.tensor(activations),
        input_tokens=torch.arange(N_POS),
    )


def test_soft_capped_scores_are_refused():
    with pytest.raises(qk.UnsupportedForQK, match="soft-capped"):
        qk.require_supported(make_model(soft_cap=50.0))


def test_a_position_scheme_outside_the_bilinear_form_is_refused():
    with pytest.raises(qk.UnsupportedForQK, match="positional embedding type"):
        qk.require_supported(make_model(scheme="alibi"))


def test_cross_layer_transcoders_are_refused():
    """A cross-layer feature writes into several layers, so it has no single direction here."""
    model = make_model()
    model.transcoders = object()
    with pytest.raises(qk.UnsupportedForQK, match="cross-layer"):
        qk.require_supported(model)


def test_an_ln1_bias_is_refused():
    with pytest.raises(qk.UnsupportedForQK, match="ln1"):
        qk.require_supported(make_model(ln1_bias=True))


def test_a_nonzero_attention_bias_is_refused():
    with pytest.raises(qk.UnsupportedForQK, match="b_Q"):
        qk.require_supported(make_model(attn_bias=True))


def test_a_zero_attention_bias_is_accepted():
    qk.require_supported(make_model())


def test_attention_scale_defaults_to_the_square_root_of_d_head():
    assert qk.attention_scale(make_model()) == pytest.approx(D_HEAD**0.5)


def test_attention_scale_is_one_when_the_model_disables_it():
    assert qk.attention_scale(make_model(use_attn_scale=False)) == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/Documents/circuit-tracer && python3 -m pytest tests/test_qk_attribution.py -q`
Expected: collection error, `ImportError: cannot import name 'qk_attribution'`.

- [ ] **Step 3: Write the module header, guards and attention scale**

```python
"""Decomposing one attention head's score into pairs of attribution-graph sources.

An attribution graph freezes attention patterns, so it explains what attention moves and says
nothing about why a head attends where it does. The pre-softmax score is still bilinear in the
residual stream with rotary embeddings and query/key normalisation, once the normalisation scales
are frozen the way a graph already freezes layer-norm scales:

    s(p, j) = x_p @ [ W_Q diag(w_q) R(p) R(j).T diag(w_k) W_K.T ] @ x_j
              / ( attn_scale * sigma_q[p] * sigma_k[j] )

``x`` is the residual after ``ln1`` including its gain, ``w_q`` and ``w_k`` are the QK-norm gains,
``R`` is the rotary rotation and the sigmas are the frozen QK-norm RMS scales. Writing the residual
at each position as its transcoder features plus a remainder expands the score into one term per
(query source, key source) pair, and the terms sum to the score exactly.

Example:

    from circuit_tracer.attribution.qk_attribution import FrozenScores, qk_attribution

    run = FrozenScores.from_model(model, graph.input_tokens)
    result = qk_attribution(model, graph, run, layer=14, head=3, query_position=8)
    values, flat = result.top_pairs(10)
"""

from __future__ import annotations

#: Position schemes that leave the score bilinear in the residual stream.
BILINEAR_POSITION_SCHEMES = frozenset({"standard", "rotary", None})

#: Layer and feature id marking a source that stands for what no feature explains.
REMAINDER = -1


class UnsupportedForQK(RuntimeError):
    """Raised when a model's attention score is not the bilinear form decomposed here."""


def attention_scale(model) -> float:
    """The divisor applied to raw scores: 1 if the model disables it, else ``sqrt(d_head)``."""
    cfg = model.cfg
    if getattr(cfg, "use_attn_scale", True) is False:
        return 1.0
    scale = getattr(cfg, "attn_scale", None)
    return float(scale) if scale else float(model.blocks[0].attn.W_Q.shape[-1] ** 0.5)


def require_supported(model) -> None:
    """Raise unless the decomposition reproduces this model's scores exactly."""
    cfg = model.cfg
    cap = getattr(cfg, "attn_scores_soft_cap", None)
    if cap is not None and float(cap) > 0:
        raise UnsupportedForQK(
            f"attention scores are soft-capped at {cap}; the tanh sits outside the bilinear form"
        )
    scheme = getattr(cfg, "positional_embedding_type", None)
    if scheme not in BILINEAR_POSITION_SCHEMES:
        raise UnsupportedForQK(
            f"positional embedding type {scheme!r} adds a term outside the bilinear form"
        )
    transcoders = getattr(model, "transcoders", None)
    # Checked structurally so the module never imports CrossLayerTranscoder: a per-layer set is
    # indexable by layer and a cross-layer transcoder is not.
    if transcoders is not None and not hasattr(type(transcoders), "__getitem__"):
        raise UnsupportedForQK(
            "a cross-layer transcoder feature writes into several layers, so it has no single "
            "direction entering attention; only per-layer transcoders are handled"
        )
    for layer, block in enumerate(model.blocks):
        if getattr(block.ln1, "b", None) is not None:
            raise UnsupportedForQK(
                f"blocks.{layer}.ln1 has a bias, an additive term no source carries"
            )
        for name in ("b_Q", "b_K"):
            bias = getattr(block.attn, name, None)
            if bias is not None and bool(bias.any()):
                raise UnsupportedForQK(
                    f"{name} is non-zero at layer {layer}; its terms belong to no source, so the "
                    "pairs would not sum to the score"
                )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/Documents/circuit-tracer && python3 -m pytest tests/test_qk_attribution.py -q && python3 -m ruff check circuit_tracer/attribution/qk_attribution.py tests/test_qk_attribution.py && python3 -m ruff format --check circuit_tracer/attribution/qk_attribution.py tests/test_qk_attribution.py`
Expected: 8 passed; ruff clean (run `ruff format` on the two files first if the check fails).

- [ ] **Step 5: Commit**

```bash
git add circuit_tracer/attribution/qk_attribution.py tests/test_qk_attribution.py
git commit -m "feat: guard QK attribution architectures"
```

---

### Task 2: Frozen scales and the rotary operator

**Files:**
- Modify: `circuit_tracer/attribution/qk_attribution.py` (append)
- Modify: `tests/test_qk_attribution.py` (append)

**Interfaces:**
- Consumes: `make_model`, `reference_cache` from Task 1.
- Produces: `qk.rotation_matrices(model, n_pos: int) -> Tensor | None` shaped `(n_pos, d_head, d_head)`; `qk.FrozenScores` with fields `resid_pre: list[Tensor]` `(pos, d_model)`, `ln1_scales: list[Tensor]` `(pos, 1)`, `query_scales: list[Tensor | None]` `(pos, n_heads, 1)`, `key_scales: list[Tensor | None]` `(pos, n_kv_heads, 1)`, `rotations: Tensor | None`, property `n_pos: int`, classmethod `from_model(model, tokens) -> FrozenScores`; private `qk._kv_group(model, head) -> int`.

- [ ] **Step 1: Append the failing tests**

```python
def test_frozen_scores_restore_the_head_axis_of_qk_norm_scales():
    run = qk.FrozenScores.from_model(make_model(), torch.arange(N_POS))
    assert run.n_pos == N_POS
    assert run.resid_pre[0].shape == (N_POS, D_MODEL)
    assert run.ln1_scales[0].shape == (N_POS, 1)
    assert run.query_scales[0].shape == (N_POS, N_HEADS, 1)
    assert run.key_scales[0].shape == (N_POS, 2, 1)


def test_frozen_scores_without_qk_norm_carry_no_scales():
    run = qk.FrozenScores.from_model(make_model(qk_norm=False), torch.arange(N_POS))
    assert run.query_scales == [None] * N_LAYERS
    assert run.key_scales == [None] * N_LAYERS


def test_a_batched_cache_is_refused():
    model = make_model()
    batched = {name: torch.cat([value, value]) for name, value in reference_cache(model).items()}
    model.run_with_cache = lambda tokens, names_filter=None: (None, batched)
    with pytest.raises(ValueError, match="a batch of 2"):
        qk.FrozenScores.from_model(model, torch.arange(N_POS))


def test_rotations_are_orthogonal():
    rotations = qk.rotation_matrices(make_model(), N_POS)
    assert rotations is not None and rotations.shape == (N_POS, D_HEAD, D_HEAD)
    identity = torch.eye(D_HEAD).expand(N_POS, D_HEAD, D_HEAD)
    torch.testing.assert_close(rotations @ rotations.transpose(1, 2), identity, atol=1e-5, rtol=0)


def test_rotations_depend_only_on_the_offset():
    rotations = qk.rotation_matrices(make_model(), N_POS)
    assert rotations is not None
    torch.testing.assert_close(
        rotations[3] @ rotations[1].T, rotations[4] @ rotations[2].T, atol=1e-5, rtol=0
    )


def test_no_rotations_without_rotary_embeddings():
    assert qk.rotation_matrices(make_model(rotary=False), N_POS) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/Documents/circuit-tracer && python3 -m pytest tests/test_qk_attribution.py -q`
Expected: 6 failures with `AttributeError: module ... has no attribute 'FrozenScores'` / `'rotation_matrices'`; the 8 Task 1 tests still pass.

- [ ] **Step 3: Add the imports this task first uses, then append the implementation**

Below `from __future__ import annotations`, add:

```python
from dataclasses import dataclass

import torch
from torch import Tensor
```

Then append:

```python
def _kv_group(model, head: int) -> int:
    """Map a query head to its key/value group; cached key scales carry one slice per group."""
    heads = model.cfg.n_heads
    groups = getattr(model.cfg, "n_key_value_heads", None) or heads
    return head // (heads // groups)


def rotation_matrices(model, n_pos: int) -> Tensor | None:
    """Return the rotary rotation at each position as a matrix, or None without rotary embeddings.

    Built by rotating the standard basis with the model's own ``apply_rotary``, so it follows that
    implementation's convention exactly. The basis is float32 so a half-precision model does not
    lower the precision of every projection that uses it.

    Returns:
        ``(n_pos, d_head, d_head)``, where ``v @ result[p]`` rotates ``v`` at position ``p``.
    """
    if getattr(model.cfg, "positional_embedding_type", None) != "rotary":
        return None
    if n_pos < 1:
        raise ValueError(f"n_pos must be positive, got {n_pos}")
    attn = model.blocks[0].attn
    size = attn.W_Q.shape[-1]
    basis = torch.eye(size, device=attn.W_Q.device, dtype=torch.float32)
    basis = basis.reshape(size, 1, 1, size).expand(size, n_pos, 1, size).contiguous()
    rotated = attn.apply_rotary(basis, 0, None)
    return rotated.squeeze(2).permute(1, 0, 2).contiguous()


def _single(value: Tensor, name: str) -> Tensor:
    if value.ndim != 3:
        raise ValueError(f"{name} should be (batch, pos, dim), got {tuple(value.shape)}")
    if value.shape[0] != 1:
        raise ValueError(
            f"expected one prompt but {name} holds a batch of {value.shape[0]}; a graph describes "
            "a single prompt"
        )
    return value[0]


@dataclass(frozen=True)
class FrozenScores:
    """What an attribution graph holds fixed about attention, from one clean run."""

    resid_pre: list[Tensor]
    ln1_scales: list[Tensor]
    query_scales: list[Tensor | None]
    key_scales: list[Tensor | None]
    rotations: Tensor | None

    @property
    def n_pos(self) -> int:
        return int(self.resid_pre[0].shape[0])

    @classmethod
    def from_model(cls, model, tokens) -> FrozenScores:
        """Run the model once on the graph's prompt and keep the frozen quantities.

        Args:
            model: A ``ReplacementModel`` on the TransformerLens backend.
            tokens: The prompt the graph was attributed on.
        """
        qk_norm = getattr(model.blocks[0].attn, "q_norm", None) is not None
        tails: tuple[str, ...] = ("hook_resid_pre", "ln1.hook_scale")
        if qk_norm:
            tails += ("attn.q_norm.hook_scale", "attn.k_norm.hook_scale")
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: name.endswith(tails))

        heads = model.cfg.n_heads
        groups = getattr(model.cfg, "n_key_value_heads", None) or heads
        resid, ln1, queries, keys = [], [], [], []
        for layer in range(model.cfg.n_layers):
            residual = _single(cache[f"blocks.{layer}.hook_resid_pre"], "hook_resid_pre")
            n_pos = residual.shape[0]
            resid.append(residual)
            ln1.append(_single(cache[f"blocks.{layer}.ln1.hook_scale"], "ln1.hook_scale"))
            if qk_norm:
                # TransformerLens flattens these hooks over (batch, pos, head).
                q_scale = cache[f"blocks.{layer}.attn.q_norm.hook_scale"]
                k_scale = cache[f"blocks.{layer}.attn.k_norm.hook_scale"]
                queries.append(q_scale.reshape(n_pos, heads, 1))
                keys.append(k_scale.reshape(n_pos, groups, 1))
            else:
                queries.append(None)
                keys.append(None)
        return cls(resid, ln1, queries, keys, rotation_matrices(model, resid[0].shape[0]))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/Documents/circuit-tracer && python3 -m pytest tests/test_qk_attribution.py -q && python3 -m ruff check circuit_tracer/attribution/qk_attribution.py tests/test_qk_attribution.py`
Expected: 14 passed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add circuit_tracer/attribution/qk_attribution.py tests/test_qk_attribution.py
git commit -m "feat: freeze scales and rotary operators"
```

---

### Task 3: Head-space projection and exact score reconstruction

**Files:**
- Modify: `circuit_tracer/attribution/qk_attribution.py` (append)
- Modify: `tests/test_qk_attribution.py` (append)

**Interfaces:**
- Consumes: `FrozenScores`, `_kv_group`, `attention_scale`, `require_supported`.
- Produces: `qk.to_head_space(model, run: FrozenScores, layer: int, head: int, directions: Tensor, positions: Tensor, *, side: str) -> Tensor` shaped `(n, d_head)`; `qk.attention_scores(model, run: FrozenScores, layer: int, head: int) -> Tensor` shaped `(n_pos, n_pos)`.

- [ ] **Step 1: Append the failing tests**

```python
ARCHITECTURES = [
    {"rotary": False, "qk_norm": False, "kv_heads": 4},
    {"rotary": True, "qk_norm": False, "kv_heads": 4},
    {"rotary": False, "qk_norm": True, "kv_heads": 2},
    {"rotary": True, "qk_norm": True, "kv_heads": 2},
]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_scores_match_an_independent_forward_pass(architecture: dict):
    model = make_model(**architecture)
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            torch.testing.assert_close(
                qk.attention_scores(model, run, layer, head),
                reference_scores(model, layer, head),
                rtol=1e-4,
                atol=1e-5,
            )


def test_to_head_space_rejects_an_unknown_side():
    model = make_model()
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    with pytest.raises(ValueError, match="side"):
        qk.to_head_space(model, run, 0, 0, torch.zeros(1, D_MODEL), torch.zeros(1).long(), side="x")


def test_to_head_space_rejects_mismatched_positions():
    model = make_model()
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    with pytest.raises(ValueError, match="positions"):
        qk.to_head_space(
            model, run, 0, 0, torch.zeros(2, D_MODEL), torch.zeros(3).long(), side="query"
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/Documents/circuit-tracer && python3 -m pytest tests/test_qk_attribution.py -q`
Expected: 6 failures with `AttributeError: ... 'attention_scores'` / `'to_head_space'`; 14 earlier tests pass.

- [ ] **Step 3: Append the implementation**

```python
def _projection(model, layer: int, head: int, side: str) -> Tensor:
    """``W_Q diag(w_q)`` or ``W_K diag(w_k)``: the QK-norm gain is a fixed per-channel scaling."""
    attn = model.blocks[layer].attn
    weight = attn.W_Q[head] if side == "query" else attn.W_K[head]
    norm = getattr(attn, "q_norm" if side == "query" else "k_norm", None)
    return weight * norm.w if norm is not None else weight


def to_head_space(
    model,
    run: FrozenScores,
    layer: int,
    head: int,
    directions: Tensor,
    positions: Tensor,
    *,
    side: str,
) -> Tensor:
    """Carry residual-stream directions into one head's rotated query or key space.

    Applies ``ln1`` in full (gain and frozen division), the projection with its QK-norm gain, the
    frozen QK-norm scale and the rotation at each direction's position. Every step is linear with
    the scales frozen, so directions can be projected one at a time and summed afterwards.

    Args:
        directions: ``(n, d_model)``, in residual-stream coordinates before ``ln1``.
        positions: ``(n,)``, the position each direction sits at.
        side: ``"query"`` or ``"key"``.

    Returns:
        ``(n, d_head)``.
    """
    if side not in ("query", "key"):
        raise ValueError(f"side must be 'query' or 'key', got {side!r}")
    if directions.ndim != 2:
        raise ValueError(f"directions must be (n, d_model), got {tuple(directions.shape)}")
    if positions.shape[0] != directions.shape[0]:
        raise ValueError(f"{positions.shape[0]} positions for {directions.shape[0]} directions")
    dtype = directions.dtype
    gain = model.blocks[layer].ln1.w.to(dtype)
    out = directions * gain / run.ln1_scales[layer][positions].to(dtype)
    out = out @ _projection(model, layer, head, side).to(dtype)
    scales = run.query_scales[layer] if side == "query" else run.key_scales[layer]
    if scales is not None:
        index = head if side == "query" else _kv_group(model, head)
        out = out / scales[positions, index].to(dtype)
    if run.rotations is not None:
        out = torch.einsum("na,nab->nb", out, run.rotations[positions].to(dtype))
    return out


def attention_scores(model, run: FrozenScores, layer: int, head: int) -> Tensor:
    """Reconstruct one head's pre-softmax scores from the frozen run, unmasked.

    Uses the same path as the decomposition, so agreement with the model's own scores is what
    licenses reading the decomposition as exact.

    Returns:
        ``(n_pos, n_pos)``, indexed ``[query, key]``.
    """
    require_supported(model)
    residual = run.resid_pre[layer].float()
    positions = torch.arange(run.n_pos, device=residual.device)
    query = to_head_space(model, run, layer, head, residual, positions, side="query")
    key = to_head_space(model, run, layer, head, residual, positions, side="key")
    return query @ key.T / attention_scale(model)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/Documents/circuit-tracer && python3 -m pytest tests/test_qk_attribution.py -q && python3 -m ruff check circuit_tracer/attribution/qk_attribution.py tests/test_qk_attribution.py`
Expected: 20 passed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add circuit_tracer/attribution/qk_attribution.py tests/test_qk_attribution.py
git commit -m "feat: reconstruct scores in head space"
```

---

### Task 4: Sources read from a graph, with the remainder

**Files:**
- Modify: `circuit_tracer/attribution/qk_attribution.py` (append)
- Modify: `tests/test_qk_attribution.py` (append)

**Interfaces:**
- Consumes: `FrozenScores`, `REMAINDER`, `make_graph`.
- Produces: `qk.SourceSet` (fields `directions (n, d_model)`, `positions (n,)`, `layers (n,)`, `feature_ids (n,)`; `__len__`, `concat(other) -> SourceSet`, `select(keep: Tensor) -> SourceSet`, property `is_remainder -> Tensor`); `qk.feature_sources(model, graph, *, below_layer: int, dtype=torch.float32) -> SourceSet`; `qk.remainder_sources(run, sources: SourceSet, layer: int) -> SourceSet`.

- [ ] **Step 1: Append the failing tests**

```python
def test_activations_follow_active_features_on_a_pruned_graph():
    """activation_values is aligned with active_features; the two coincide only when nothing is
    pruned, which is how indexing it by selection stays invisible on small graphs."""
    model = make_model()
    graph = make_graph([(0, 1, 2), (0, 3, 4), (1, 2, 5)], [0, 2], [3.0, 99.0, 7.0])
    sources = qk.feature_sources(model, graph, below_layer=2)
    torch.testing.assert_close(sources.directions[0], model.transcoders[0].W_dec[2] * 3.0)
    torch.testing.assert_close(sources.directions[1], model.transcoders[1].W_dec[5] * 7.0)


def test_activation_values_matching_neither_table_are_rejected():
    graph = make_graph([(0, 1, 2), (0, 3, 4), (1, 2, 5)], [0, 2], [3.0, 7.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="matching neither"):
        qk.feature_sources(make_model(), graph, below_layer=2)


def test_only_features_written_below_the_layer_are_sources():
    graph = make_graph([(0, 1, 2), (1, 3, 4), (2, 2, 5)], [0, 1, 2], [1.0, 1.0, 1.0])
    sources = qk.feature_sources(make_model(), graph, below_layer=2)
    assert sources.layers.tolist() == [0, 1]
    assert sources.directions.dtype == torch.float32


def test_sources_and_the_remainder_reconstruct_the_residual():
    model = make_model()
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    graph = make_graph([(0, 1, 2), (0, 1, 3), (1, 4, 5)], [0, 1, 2], [2.0, -1.5, 0.5])
    features = qk.feature_sources(model, graph, below_layer=2)
    sources = features.concat(qk.remainder_sources(run, features, 2))
    rebuilt = torch.zeros(N_POS, D_MODEL).index_add_(0, sources.positions, sources.directions)
    torch.testing.assert_close(rebuilt, run.resid_pre[2], rtol=1e-5, atol=1e-5)
    assert int(sources.is_remainder.sum()) == N_POS
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/Documents/circuit-tracer && python3 -m pytest tests/test_qk_attribution.py -q`
Expected: 4 failures with `AttributeError: ... 'feature_sources'`; 20 earlier tests pass.

- [ ] **Step 3: Add the type-only `Graph` import, then append the implementation**

Change the imports so they read:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from circuit_tracer.graph import Graph
```

Then append:

```python
@dataclass(frozen=True)
class SourceSet:
    """Residual-stream directions feeding one attention layer, with where each came from.

    Directions are already scaled by their activations, so summing them by position rebuilds the
    part of the residual stream they account for.
    """

    directions: Tensor
    positions: Tensor
    layers: Tensor
    feature_ids: Tensor

    def __len__(self) -> int:
        return int(self.directions.shape[0])

    def concat(self, other: SourceSet) -> SourceSet:
        return SourceSet(
            directions=torch.cat([self.directions, other.directions.to(self.directions.dtype)]),
            positions=torch.cat([self.positions, other.positions]),
            layers=torch.cat([self.layers, other.layers]),
            feature_ids=torch.cat([self.feature_ids, other.feature_ids]),
        )

    def select(self, keep: Tensor) -> SourceSet:
        return SourceSet(
            directions=self.directions[keep],
            positions=self.positions[keep],
            layers=self.layers[keep],
            feature_ids=self.feature_ids[keep],
        )

    @property
    def is_remainder(self) -> Tensor:
        """Rows standing for what no transcoder feature explains."""
        return self.layers == REMAINDER


def _activations_for(graph: Graph) -> Tensor:
    """Each selected feature's activation, checking which table ``activation_values`` follows."""
    selected = graph.selected_features
    values = graph.activation_values
    if len(values) == len(graph.active_features):
        return values[selected]
    if len(values) == len(selected):
        return values
    raise ValueError(
        f"activation_values has {len(values)} entries, matching neither active_features "
        f"({len(graph.active_features)}) nor selected_features ({len(selected)})"
    )


def feature_sources(model, graph: Graph, *, below_layer: int, dtype=torch.float32) -> SourceSet:
    """Activation-scaled decoder directions of every selected feature written below a layer.

    A per-layer transcoder feature writes at its own block's MLP output, so attention at a later
    block sees it and attention at its own block does not. Directions are extracted in float32 by
    default: the contraction sums many terms that largely cancel, and bfloat16 loses most of the
    result. Decoder rows are gathered one layer at a time, since a lazily loaded decoder can re-read
    the layer from disk on every access.
    """
    active = graph.active_features[graph.selected_features]
    values = _activations_for(graph)
    keep = active[:, 0] < below_layer
    template = model.transcoders[0].W_dec
    device = template.device
    layers = active[keep, 0].to(device)
    positions = active[keep, 1].to(device)
    feature_ids = active[keep, 2].to(device)
    directions = torch.empty((int(keep.sum()), template.shape[-1]), dtype=dtype, device=device)
    for layer in layers.unique().tolist():
        rows = layers == layer
        directions[rows] = model.transcoders[layer].W_dec[feature_ids[rows]].to(dtype)
    directions = directions * values[keep].to(device=device, dtype=dtype).unsqueeze(-1)
    return SourceSet(directions, positions, layers, feature_ids)


def remainder_sources(run: FrozenScores, sources: SourceSet, layer: int) -> SourceSet:
    """What the given sources leave out of the residual entering ``layer``, one row per position.

    Transcoder features cover only MLP writes. Earlier attention outputs, embeddings, transcoder
    errors and decoder biases are in the residual too; carrying them as one lumped direction per
    position keeps the decomposition exhaustive and makes the features' share visible.
    """
    remainder = run.resid_pre[layer].to(torch.float32).clone()
    if len(sources):
        remainder.index_add_(
            0,
            sources.positions.to(remainder.device),
            -sources.directions.to(device=remainder.device, dtype=remainder.dtype),
        )
    seq, device = remainder.shape[0], remainder.device
    marker = torch.full((seq,), REMAINDER, dtype=torch.long, device=device)
    return SourceSet(remainder, torch.arange(seq, device=device), marker, marker.clone())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/Documents/circuit-tracer && python3 -m pytest tests/test_qk_attribution.py -q && python3 -m ruff check circuit_tracer/attribution/qk_attribution.py tests/test_qk_attribution.py`
Expected: 24 passed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add circuit_tracer/attribution/qk_attribution.py tests/test_qk_attribution.py
git commit -m "feat: read QK sources from graphs"
```

---

### Task 5: The contraction into source pairs

**Files:**
- Modify: `circuit_tracer/attribution/qk_attribution.py` (append)
- Modify: `tests/test_qk_attribution.py` (append)

**Interfaces:**
- Consumes: everything above.
- Produces: `qk.QKAttribution` (fields `contributions (n_query, n_key)`, `query_sources`, `key_sources`, `layer`, `head`, `query_position`; properties `by_key_source`, `by_query_source`; methods `by_key_position(n_pos) -> Tensor`, `top_pairs(count) -> tuple[Tensor, Tensor]`); `qk.qk_attribution(model, graph, run, layer: int, head: int, query_position: int) -> QKAttribution`.

- [ ] **Step 1: Append the failing tests**

```python
GRAPH_FEATURES = ([(0, 1, 2), (0, 5, 3), (1, 4, 5), (1, 5, 1), (2, 3, 7)], [0, 1, 2, 3, 4])
ACTIVATIONS = [2.0, -1.0, 0.5, 1.5, 3.0]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_contributions_sum_to_the_score_row(architecture: dict):
    """The whole claim: with the remainder carried, the pairs reproduce the score exactly."""
    model = make_model(**architecture)
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    graph = make_graph(*GRAPH_FEATURES, ACTIVATIONS)
    for layer in (1, 2):
        for query_position in (2, N_POS - 1):
            result = qk.qk_attribution(model, graph, run, layer, 1, query_position)
            expected = reference_scores(model, layer, 1)[query_position, : query_position + 1]
            got = result.by_key_position(N_POS)[: query_position + 1]
            torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)


def test_query_sources_all_sit_at_the_query_position():
    model = make_model()
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    result = qk.qk_attribution(model, make_graph(*GRAPH_FEATURES, ACTIVATIONS), run, 2, 0, 5)
    assert set(result.query_sources.positions.tolist()) == {5}
    assert int(result.query_sources.is_remainder.sum()) == 1


def test_key_sources_never_come_from_later_positions():
    model = make_model()
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    result = qk.qk_attribution(model, make_graph(*GRAPH_FEATURES, ACTIVATIONS), run, 2, 0, 3)
    assert int(result.key_sources.positions.max()) <= 3
    assert result.contributions.shape == (len(result.query_sources), len(result.key_sources))


def test_top_pairs_are_the_largest_by_magnitude():
    model = make_model()
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    result = qk.qk_attribution(model, make_graph(*GRAPH_FEATURES, ACTIVATIONS), run, 2, 3, 5)
    values, flat = result.top_pairs(3)
    torch.testing.assert_close(values, result.contributions.flatten()[flat])
    assert float(values.abs().min()) >= float(result.contributions.abs().flatten().sort().values[-3])


@pytest.mark.parametrize(
    ("layer", "head", "query_position", "fragment"),
    [(N_LAYERS, 0, 0, "layer"), (0, N_HEADS, 0, "head"), (0, 0, N_POS, "query_position")],
)
def test_out_of_range_arguments_raise(layer: int, head: int, query_position: int, fragment: str):
    model = make_model()
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    graph = make_graph(*GRAPH_FEATURES, ACTIVATIONS)
    with pytest.raises(IndexError, match=fragment):
        qk.qk_attribution(model, graph, run, layer, head, query_position)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/Documents/circuit-tracer && python3 -m pytest tests/test_qk_attribution.py -q`
Expected: 10 failures with `AttributeError: ... 'qk_attribution'`; 24 earlier tests pass.

- [ ] **Step 3: Append the implementation**

```python
@dataclass(frozen=True)
class QKAttribution:
    """One head's score from one query position, expanded into (query source, key source) terms."""

    contributions: Tensor
    query_sources: SourceSet
    key_sources: SourceSet
    layer: int
    head: int
    query_position: int

    @property
    def by_key_source(self) -> Tensor:
        return self.contributions.sum(dim=0)

    @property
    def by_query_source(self) -> Tensor:
        return self.contributions.sum(dim=1)

    def by_key_position(self, n_pos: int) -> Tensor:
        """The score at each key position, which compares against a row of the score matrix.

        Accumulated in float32: many terms of both signs land on each position.
        """
        totals = torch.zeros(n_pos, dtype=torch.float32, device=self.contributions.device)
        if len(self.key_sources):
            totals.index_add_(0, self.key_sources.positions, self.by_key_source.float())
        return totals

    def top_pairs(self, count: int) -> tuple[Tensor, Tensor]:
        """The ``count`` largest terms by magnitude, signed, with flat indices.

        Use ``divmod(index, contributions.shape[1])`` to recover the query and key source rows.
        """
        flat = self.contributions.flatten()
        _, indices = flat.abs().topk(min(count, flat.numel()))
        return flat[indices], indices


def qk_attribution(
    model,
    graph: Graph,
    run: FrozenScores,
    layer: int,
    head: int,
    query_position: int,
) -> QKAttribution:
    """Expand one head's score from one query position into source-pair terms.

    Sources are the graph's selected features written below ``layer`` plus one remainder per
    position, so the terms for each key position sum to that entry of the score matrix. Over all
    query positions at once the contraction grows too large to hold, which is why this takes one.

    Args:
        model: A ``ReplacementModel`` on the TransformerLens backend.
        graph: The attribution graph for the prompt ``run`` was captured on.
        run: A :class:`FrozenScores` for that prompt.
        layer: The attention layer.
        head: The query head.
        query_position: The attending position.
    """
    require_supported(model)
    if not 0 <= layer < model.cfg.n_layers:
        raise IndexError(f"layer {layer} out of range for {model.cfg.n_layers} layers")
    if not 0 <= head < model.cfg.n_heads:
        raise IndexError(f"head {head} out of range for {model.cfg.n_heads} heads")
    if not 0 <= query_position < run.n_pos:
        raise IndexError(f"query_position {query_position} out of range for {run.n_pos} positions")

    features = feature_sources(model, graph, below_layer=layer)
    sources = features.concat(remainder_sources(run, features, layer))
    query_sources = sources.select(sources.positions == query_position)
    key_sources = sources.select(sources.positions <= query_position)

    left = to_head_space(
        model, run, layer, head, query_sources.directions, query_sources.positions, side="query"
    )
    right = to_head_space(
        model, run, layer, head, key_sources.directions, key_sources.positions, side="key"
    )
    return QKAttribution(
        contributions=left @ right.T / attention_scale(model),
        query_sources=query_sources,
        key_sources=key_sources,
        layer=layer,
        head=head,
        query_position=query_position,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/Documents/circuit-tracer && python3 -m pytest tests/test_qk_attribution.py -q && python3 -m ruff check circuit_tracer/attribution/qk_attribution.py tests/test_qk_attribution.py`
Expected: 34 passed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add circuit_tracer/attribution/qk_attribution.py tests/test_qk_attribution.py
git commit -m "feat: decompose scores into source pairs"
```

---

### Task 6: Public export

**Files:**
- Modify: `circuit_tracer/__init__.py` (the `TYPE_CHECKING` imports, `__all__`, and `_lazy_imports`)

**Interfaces:**
- Consumes: `qk_attribution`, `FrozenScores`.
- Produces: `circuit_tracer.qk_attribution`, `circuit_tracer.FrozenScores` resolved lazily.

- [ ] **Step 1: Write the failing check**

Run: `cd ~/Documents/circuit-tracer && python3 -c "import circuit_tracer; print(circuit_tracer.FrozenScores.__name__)"`
Expected: `AttributeError: module 'circuit_tracer' has no attribute 'FrozenScores'`.

- [ ] **Step 2: Add the exports**

In the `if TYPE_CHECKING:` block add:

```python
    from circuit_tracer.attribution.qk_attribution import FrozenScores, qk_attribution
```

Append to `__all__`:

```python
    "qk_attribution",
    "FrozenScores",
```

Add to `_lazy_imports`:

```python
        "qk_attribution": ("circuit_tracer.attribution.qk_attribution", "qk_attribution"),
        "FrozenScores": ("circuit_tracer.attribution.qk_attribution", "FrozenScores"),
```

- [ ] **Step 3: Verify**

Run: `cd ~/Documents/circuit-tracer && python3 -c "import circuit_tracer; print(circuit_tracer.FrozenScores.__name__, circuit_tracer.qk_attribution.__name__)" && python3 -m pytest tests/test_qk_attribution.py -q && python3 -m ruff check . && python3 -m ruff format --check .`
Expected: `FrozenScores qk_attribution`; 34 passed; ruff clean.

- [ ] **Step 4: Commit**

```bash
git add circuit_tracer/__init__.py
git commit -m "feat: export QK attribution publicly"
```

---

## Deferred until the GPU workers are reachable

Not tasks in this plan, because nothing here can execute them: run the branch's full test suite and pyright on worker-2, and check `attention_scores` against a real Qwen3-0.6B's `hook_attn_scores` the way `experiments/003-exact-qk-form.md` did, before any of these commits are pushed.
