"""A stand-in for a HookedTransformer, rich enough to exercise the score reconstruction.

Loading real weights takes a GPU and several gigabytes of download, so the unit tests run against
this instead. It implements the pieces the library actually reads: per-block projection weights,
optional QK-norm gains, and a genuine rotary implementation whose rotation is linear, orthogonal
and relative, as RoPE's is. Checks against a real model live in the ``gpu``-marked tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor

D_MODEL, D_HEAD, N_HEADS, N_LAYERS = 6, 4, 4, 2


class StubAttention(SimpleNamespace):
    """Attention block exposing weights and a rotary implementation."""

    def apply_rotary(
        self, x: Tensor, past_kv_pos_offset: int = 0, attention_mask: object = None
    ) -> Tensor:
        """Rotate ``[batch, pos, head, d_head]`` by a per-position angle, pairing adjacent dims."""
        if self.rotary_freqs is None:
            raise AttributeError("stub has no rotary embeddings")
        positions = torch.arange(x.shape[1], dtype=torch.float32) + past_kv_pos_offset
        angle = positions[:, None] * self.rotary_freqs[None, :]
        cos = angle.cos()[None, :, None, :]
        sin = angle.sin()[None, :, None, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out


def make_model(
    *,
    d_model: int = D_MODEL,
    d_head: int = D_HEAD,
    n_heads: int = N_HEADS,
    n_layers: int = N_LAYERS,
    n_key_value_heads: int | None = None,
    positional_embedding_type: str = "standard",
    use_qk_norm: bool = False,
    attn_scores_soft_cap: float = -1.0,
    attn_scale: float | None = None,
    with_bias: bool = False,
    seed: int = 0,
) -> SimpleNamespace:
    """Build a stub exposing the attention surface the library reads.

    Args:
        n_key_value_heads: Set below ``n_heads`` to model grouped-query attention.
        positional_embedding_type: ``"rotary"`` enables :meth:`StubAttention.apply_rotary`.
        use_qk_norm: Attach ``q_norm`` and ``k_norm`` modules carrying a learned gain.
        with_bias: Give the query and key projections a non-zero bias.
    """
    gen = torch.Generator().manual_seed(seed)
    rotary = positional_embedding_type == "rotary"
    if rotary and d_head % 2:
        raise ValueError(f"rotary needs an even d_head, got {d_head}")
    freqs = torch.rand(d_head // 2, generator=gen) if rotary else None

    kv_heads = n_key_value_heads or n_heads
    group = n_heads // kv_heads

    def expanded(*shape: int) -> Tensor:
        """Key-side weights, materialised per query head the way TransformerLens does."""
        return torch.randn(kv_heads, *shape, generator=gen).repeat_interleave(group, dim=0)

    def block() -> SimpleNamespace:
        zeros = torch.zeros(n_heads, d_head)
        attn = StubAttention(
            W_Q=torch.randn(n_heads, d_model, d_head, generator=gen),
            W_K=expanded(d_model, d_head),
            W_V=expanded(d_model, d_head),
            W_O=torch.randn(n_heads, d_head, d_model, generator=gen),
            b_Q=torch.randn(n_heads, d_head, generator=gen) if with_bias else zeros.clone(),
            b_K=expanded(d_head) if with_bias else zeros.clone(),
            rotary_freqs=freqs,
        )
        if use_qk_norm:
            attn.q_norm = SimpleNamespace(w=torch.rand(d_head, generator=gen) + 0.5)
            attn.k_norm = SimpleNamespace(w=torch.rand(d_head, generator=gen) + 0.5)
        ln1 = SimpleNamespace(w=torch.rand(d_model, generator=gen) + 0.5)
        ln2 = SimpleNamespace(w=torch.rand(d_model, generator=gen) + 0.5)
        return SimpleNamespace(attn=attn, ln1=ln1, ln2=ln2)

    cfg = SimpleNamespace(
        n_key_value_heads=n_key_value_heads,
        positional_embedding_type=positional_embedding_type,
        use_qk_norm=use_qk_norm,
        attn_scores_soft_cap=attn_scores_soft_cap,
        attn_scale=attn_scale if attn_scale is not None else d_head**0.5,
        rotary_dim=d_head if rotary else None,
        eps=1e-6,
    )
    return SimpleNamespace(blocks=[block() for _ in range(n_layers)], cfg=cfg)


def reference_scores(model: SimpleNamespace, layer: int, head: int, residual: Tensor) -> Tensor:
    """Compute one head's pre-softmax scores the long way, mirroring a real forward pass.

    Deliberately written as project, normalise, rotate, dot — the order a transformer uses — so
    that it is an independent check on the reconstruction rather than a restatement of it.
    """
    attn = model.blocks[layer].attn
    cfg = model.cfg

    q = residual @ attn.W_Q[head] + attn.b_Q[head]
    k = residual @ attn.W_K[head] + attn.b_K[head]
    if cfg.use_qk_norm:
        q = q / (q.pow(2).mean(-1, keepdim=True) + cfg.eps).sqrt() * attn.q_norm.w
        k = k / (k.pow(2).mean(-1, keepdim=True) + cfg.eps).sqrt() * attn.k_norm.w
    if cfg.positional_embedding_type == "rotary":
        q = attn.apply_rotary(q[None, :, None, :])[0, :, 0]
        k = attn.apply_rotary(k[None, :, None, :])[0, :, 0]
    return q @ k.transpose(-1, -2) / float(cfg.attn_scale)


class StubTranscoders:
    """A per-layer transcoder set exposing only the decoder rows the library reads."""

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


class StubGraph:
    """A circuit-tracer Graph stand-in carrying only the fields the library reads."""

    def __init__(
        self,
        active_features: Tensor,
        selected_features: Tensor,
        activation_values: Tensor,
        input_tokens: Tensor,
    ) -> None:
        self.active_features = active_features
        self.selected_features = selected_features
        self.activation_values = activation_values
        self.input_tokens = input_tokens

    @classmethod
    def with_features(
        cls,
        triples: list[tuple[int, int, int]],
        activations: list[float],
        n_pos: int = 4,
    ) -> StubGraph:
        """Build a graph whose selected features are exactly ``triples``."""
        return cls(
            active_features=torch.tensor(triples, dtype=torch.int64).reshape(-1, 3),
            selected_features=torch.arange(len(triples)),
            activation_values=torch.tensor(activations),
            input_tokens=torch.zeros(n_pos, dtype=torch.int64),
        )
