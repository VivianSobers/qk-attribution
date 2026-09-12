"""Decoding of attribution-graph adjacency indices into typed nodes.

A circuit-tracer adjacency matrix is square and indexed by a flat node ordering with four
contiguous regions, in this order:

    [ active features ] [ MLP error nodes ] [ token embeddings ] [ logit targets ]

Error nodes are laid out layer-major and position-minor, so ``divmod(i - n_features, n_pos)``
recovers ``(layer, pos)``. Everything downstream of this module needs to know which region an
index falls in, so the arithmetic lives here once rather than being repeated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class NodeKind(Enum):
    """Which region of the adjacency ordering a node index belongs to."""

    FEATURE = "feature"
    ERROR = "error"
    TOKEN = "token"
    LOGIT = "logit"


@dataclass(frozen=True)
class NodeLayout:
    """Index boundaries of the four node regions in an adjacency matrix.

    Construct with :meth:`from_graph` rather than by hand; the field values must agree with the
    graph that produced the adjacency matrix or every decoded index will be wrong.
    """

    n_features: int
    n_layers: int
    n_pos: int
    n_logits: int

    def __post_init__(self) -> None:
        for name in ("n_features", "n_layers", "n_pos", "n_logits"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.n_pos == 0:
            raise ValueError("n_pos must be positive; an empty prompt has no graph")

    @property
    def feature_end(self) -> int:
        """One past the last feature index."""
        return self.n_features

    @property
    def error_end(self) -> int:
        """One past the last error index."""
        return self.n_features + self.n_layers * self.n_pos

    @property
    def token_end(self) -> int:
        """One past the last token-embedding index."""
        return self.error_end + self.n_pos

    @property
    def total(self) -> int:
        """Total node count, equal to each side of the adjacency matrix."""
        return self.token_end + self.n_logits

    @classmethod
    def from_graph(cls, graph: Any) -> NodeLayout:
        """Derive the layout from a circuit-tracer ``Graph``.

        ``n_layers`` is recovered from the adjacency size rather than read from the model config,
        so that the result is consistent with the matrix actually present even if a config is
        missing or stale. The derivation is then checked against the matrix shape.
        """
        n_features = len(graph.selected_features)
        n_pos = len(graph.input_tokens)
        n_logits = len(graph.logit_targets)
        total = int(graph.adjacency_matrix.shape[0])
        if n_pos == 0:
            raise ValueError("n_pos must be positive; an empty prompt has no graph")

        remainder = total - n_features - n_pos - n_logits
        if remainder < 0 or remainder % n_pos != 0:
            raise ValueError(
                "adjacency size is inconsistent with the graph: "
                f"total={total}, n_features={n_features}, n_pos={n_pos}, n_logits={n_logits} "
                f"leaves {remainder} error nodes, which is not a multiple of n_pos"
            )

        layout = cls(
            n_features=n_features,
            n_layers=remainder // n_pos,
            n_pos=n_pos,
            n_logits=n_logits,
        )
        if layout.total != total:
            raise ValueError(f"derived layout totals {layout.total} but matrix side is {total}")
        return layout

    def kind(self, index: int) -> NodeKind:
        """Return which region ``index`` falls in."""
        self._check(index)
        if index < self.feature_end:
            return NodeKind.FEATURE
        if index < self.error_end:
            return NodeKind.ERROR
        if index < self.token_end:
            return NodeKind.TOKEN
        return NodeKind.LOGIT

    def decode_feature(self, index: int) -> int:
        """Return the position of ``index`` within the feature region."""
        self._expect(index, NodeKind.FEATURE)
        return index

    def decode_error(self, index: int) -> tuple[int, int]:
        """Return ``(layer, pos)`` for an error node."""
        self._expect(index, NodeKind.ERROR)
        layer, pos = divmod(index - self.n_features, self.n_pos)
        return layer, pos

    def decode_token(self, index: int) -> int:
        """Return the sequence position of a token-embedding node."""
        self._expect(index, NodeKind.TOKEN)
        return index - self.error_end

    def decode_logit(self, index: int) -> int:
        """Return the rank of a logit node within ``graph.logit_targets``."""
        self._expect(index, NodeKind.LOGIT)
        return index - self.token_end

    def _check(self, index: int) -> None:
        if not 0 <= index < self.total:
            raise IndexError(f"node index {index} out of range for {self.total} nodes")

    def _expect(self, index: int, kind: NodeKind) -> None:
        actual = self.kind(index)
        if actual is not kind:
            raise ValueError(f"node {index} is a {actual.value} node, not a {kind.value} node")
