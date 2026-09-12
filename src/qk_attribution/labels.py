"""Human-readable labels for transcoder features.

A QK attribution is a list of feature-pair contributions, and a feature is an integer until
something says what it responds to. The transcoder repositories publish that: for each feature, the
tokens it promotes and suppresses, how often it fires, and the examples it fires hardest on.

Those files are large, about 1.3 GB per layer for Qwen3-0.6B, but they are addressable. An index
gives the byte range of every feature's own chunk, so a label costs one range request rather than a
download. Fetched chunks are cached on disk, since the same features recur across prompts.

Parsing is separated from fetching so the parsing can be tested without a network.
"""

from __future__ import annotations

import gzip
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

INDEX_FILE = "features/index.json.gz"
DEFAULT_CACHE = Path.home() / ".cache" / "qk-attribution" / "features"


@dataclass(frozen=True)
class FeatureCard:
    """What a transcoder feature responds to and what it promotes."""

    layer: int
    index: int
    top_logits: list[str] = field(default_factory=list)
    bottom_logits: list[str] = field(default_factory=list)
    top_tokens: list[tuple[str, float]] = field(default_factory=list)
    activation_frequency: float = 0.0
    act_max: float = 0.0

    @property
    def is_dead(self) -> bool:
        """True when the feature barely fires, so its label means little."""
        return self.act_max <= 1e-6 or self.activation_frequency <= 0.0

    def label(self, width: int = 4) -> str:
        """A short description, built from what the feature fires on and what it promotes."""
        if self.is_dead:
            return f"L{self.layer}#{self.index} (inactive)"
        fires = ", ".join(repr(token) for token, _ in self.top_tokens[:width]) or "?"
        promotes = ", ".join(repr(token) for token in self.top_logits[:width]) or "?"
        return f"L{self.layer}#{self.index} fires on {fires} -> promotes {promotes}"


def parse_chunk(raw: bytes) -> dict[str, Any]:
    """Decode one feature's chunk: a little-endian length, then gzipped JSON.

    Raises:
        ValueError: if the chunk is truncated or its length prefix disagrees with its size.
    """
    if len(raw) < 4:
        raise ValueError(f"chunk is {len(raw)} bytes, too short to hold a length prefix")
    length = struct.unpack("<I", raw[:4])[0]
    if length > len(raw) - 4:
        raise ValueError(f"chunk declares {length} bytes but only {len(raw) - 4} follow")
    return json.loads(gzip.decompress(raw[4 : 4 + length]))


def card_from_chunk(data: dict[str, Any], layer: int, index: int, top: int = 8) -> FeatureCard:
    """Build a :class:`FeatureCard` from a decoded chunk.

    Top tokens are gathered across every example rather than from the first one, since a feature's
    single strongest example is often less telling than what recurs.
    """
    scored: dict[str, float] = {}
    for quantile in data.get("examples_quantiles") or []:
        for example in quantile.get("examples") or []:
            tokens = example.get("tokens") or []
            activations = example.get("tokens_acts_list") or []
            for token, activation in zip(tokens, activations, strict=False):
                if activation > scored.get(token, 0.0):
                    scored[token] = float(activation)
    ranked = sorted(scored.items(), key=lambda pair: -pair[1])[:top]
    return FeatureCard(
        layer=layer,
        index=index,
        top_logits=[str(token) for token in (data.get("top_logits") or [])],
        bottom_logits=[str(token) for token in (data.get("bottom_logits") or [])],
        top_tokens=ranked,
        activation_frequency=float(data.get("activation_frequency") or 0.0),
        act_max=float(data.get("act_max") or 0.0),
    )


class FeatureStore:
    """Fetches feature cards for one transcoder set, caching each chunk on disk."""

    def __init__(self, scan: str, cache_dir: Path | None = None) -> None:
        self.scan = scan
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
        self._index: dict[str, Any] | None = None
        self._cards: dict[tuple[int, int], FeatureCard] = {}

    @property
    def index(self) -> dict[str, Any]:
        """The per-layer offset index, downloaded once."""
        index = self._index
        if index is None:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(self.scan, INDEX_FILE)
            with gzip.open(path, "rt") as handle:
                index = json.load(handle)
            self._index = index
        return index

    def byte_range(self, layer: int, index: int) -> tuple[str, int, int]:
        """Return the file name and half-open byte range holding one feature's chunk."""
        entry = self.index.get(str(layer))
        if entry is None:
            raise KeyError(f"no feature metadata for layer {layer} in {self.scan}")
        offsets = entry["offsets"]
        if not 0 <= index < len(offsets) - 1:
            raise IndexError(
                f"feature {index} out of range for {len(offsets) - 1} features in layer {layer}"
            )
        return entry["filename"], int(offsets[index]), int(offsets[index + 1])

    def card(self, layer: int, index: int) -> FeatureCard:
        """Return the card for one feature, from memory, disk cache, or the hub."""
        key = (layer, index)
        if key in self._cards:
            return self._cards[key]
        path = self.cache_dir / self.scan.replace("/", "_") / f"{layer}_{index}.json"
        if path.exists():
            data = json.loads(path.read_text())
        else:
            data = self._download(layer, index)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Written via a temporary file: an interrupted write would otherwise leave truncated
            # JSON that every later read fails on, with nothing to invalidate it.
            temporary = path.with_suffix(".partial")
            temporary.write_text(json.dumps(data))
            temporary.replace(path)
        card = card_from_chunk(data, layer, index)
        self._cards[key] = card
        return card

    def _download(self, layer: int, index: int) -> dict[str, Any]:
        import requests
        from huggingface_hub import get_token, hf_hub_url

        filename, start, end = self.byte_range(layer, index)
        url = hf_hub_url(self.scan, filename, subfolder="features")
        headers = {"Range": f"bytes={start}-{end - 1}"}
        token = get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = requests.get(url, headers=headers, timeout=120)
        response.raise_for_status()
        if response.status_code != 206:
            # A server that ignores the Range header returns the whole file with status 200, and
            # every feature in the layer would then decode to the first one's card and be cached
            # under its own name.
            raise RuntimeError(
                f"expected a partial response for {filename} bytes {start}-{end - 1}, "
                f"got status {response.status_code} with {len(response.content)} bytes"
            )
        return parse_chunk(response.content)
