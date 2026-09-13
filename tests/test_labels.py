"""Tests for feature label parsing.

Fetching is separated from parsing precisely so this runs without a network. The chunk format is a
length prefix followed by gzipped JSON, and getting either half wrong would either raise or, worse,
attach one feature's label to another feature's contribution.
"""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import pytest

from qk_attribution.labels import FeatureCard, FeatureStore, card_from_chunk, parse_chunk


def make_chunk(data: dict) -> bytes:
    payload = gzip.compress(json.dumps(data).encode())
    return struct.pack("<I", len(payload)) + payload


EXAMPLE = {
    "index": 7,
    "top_logits": [" Paris", " France", " Lyon"],
    "bottom_logits": [" cat"],
    "activation_frequency": 0.004,
    "act_max": 12.5,
    "examples_quantiles": [
        {
            "quantile_name": "top",
            "examples": [
                {"tokens": [" the", " capital", " of"], "tokens_acts_list": [0.0, 9.0, 1.0]},
                {"tokens": [" capital", " city"], "tokens_acts_list": [12.5, 2.0]},
            ],
        }
    ],
}


def test_parse_chunk_round_trips():
    assert parse_chunk(make_chunk(EXAMPLE)) == EXAMPLE


def test_parse_chunk_ignores_trailing_bytes():
    """Ranges are taken between offsets, so a chunk may carry padding after its payload."""
    assert parse_chunk(make_chunk(EXAMPLE) + b"\x00" * 32) == EXAMPLE


def test_parse_chunk_rejects_a_truncated_chunk():
    with pytest.raises(ValueError, match="too short"):
        parse_chunk(b"\x01\x02")


def test_parse_chunk_rejects_a_short_payload():
    raw = make_chunk(EXAMPLE)
    with pytest.raises(ValueError, match="declares"):
        parse_chunk(raw[:-10])


def test_card_takes_the_highest_activation_per_token():
    card = card_from_chunk(EXAMPLE, layer=3, index=7)
    assert card.layer == 3
    assert card.index == 7
    assert card.top_tokens[0] == (" capital", 12.5)
    assert dict(card.top_tokens)[" of"] == 1.0
    assert card.top_logits == [" Paris", " France", " Lyon"]


def test_card_orders_tokens_by_activation():
    card = card_from_chunk(EXAMPLE, layer=0, index=0)
    activations = [activation for _, activation in card.top_tokens]
    assert activations == sorted(activations, reverse=True)


def test_card_handles_a_chunk_with_no_examples():
    card = card_from_chunk({"top_logits": ["a"]}, layer=1, index=2)
    assert card.top_tokens == []
    assert card.act_max == 0.0
    assert card.is_dead


def test_label_names_what_it_fires_on_and_promotes():
    label = card_from_chunk(EXAMPLE, layer=3, index=7).label(width=2)
    assert "L3#7" in label
    assert "' capital'" in label
    assert "' Paris'" in label


def test_dead_features_are_labelled_as_inactive():
    card = FeatureCard(layer=1, index=2, act_max=0.0, activation_frequency=0.0)
    assert card.is_dead
    assert card.label() == "L1#2 (inactive)"


def test_a_barely_active_feature_counts_as_dead():
    card = FeatureCard(layer=1, index=2, act_max=1e-9, activation_frequency=1e-9)
    assert card.is_dead


def test_byte_range_spans_one_feature():
    store = FeatureStore("someone/set")
    store._index = {"0": {"filename": "layer_0.bin", "offsets": [0, 100, 250, 400]}}
    assert store.byte_range(0, 0) == ("layer_0.bin", 0, 100)
    assert store.byte_range(0, 2) == ("layer_0.bin", 250, 400)


def test_byte_range_rejects_an_unknown_layer():
    store = FeatureStore("someone/set")
    store._index = {"0": {"filename": "layer_0.bin", "offsets": [0, 100]}}
    with pytest.raises(KeyError, match="layer 5"):
        store.byte_range(5, 0)


@pytest.mark.parametrize("index", [-1, 3, 99])
def test_byte_range_rejects_an_out_of_range_feature(index: int):
    store = FeatureStore("someone/set")
    store._index = {"0": {"filename": "layer_0.bin", "offsets": [0, 100, 250, 400]}}
    with pytest.raises(IndexError, match="out of range"):
        store.byte_range(0, index)


def test_cards_are_read_from_the_disk_cache(tmp_path: Path):
    """A cached chunk must not trigger a download; the store has no network in this test."""
    store = FeatureStore("someone/set", cache_dir=tmp_path)
    path = tmp_path / "someone_set" / "3_7.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(EXAMPLE))
    card = store.card(3, 7)
    assert card.top_logits == [" Paris", " France", " Lyon"]


def test_cards_are_memoised(tmp_path: Path):
    store = FeatureStore("someone/set", cache_dir=tmp_path)
    path = tmp_path / "someone_set" / "1_1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(EXAMPLE))
    first = store.card(1, 1)
    path.unlink()
    assert store.card(1, 1) is first


def test_a_full_response_to_a_range_request_is_refused(tmp_path: Path, monkeypatch):
    """A server ignoring Range returns the whole file; every feature would decode to the first."""
    store = FeatureStore("someone/set", cache_dir=tmp_path)
    store._index = {"0": {"filename": "layer_0.bin", "offsets": [0, 100, 250]}}

    class Response:
        status_code = 200
        content = make_chunk(EXAMPLE)

        def raise_for_status(self):
            return None

    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: Response())
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_url", lambda *a, **k: "https://example/x")
    with pytest.raises(RuntimeError, match="expected a partial response"):
        store.card(0, 1)


def test_an_interrupted_cache_write_leaves_no_file(tmp_path: Path):
    """The cache is written through a temporary file, so a crash cannot leave truncated JSON."""
    store = FeatureStore("someone/set", cache_dir=tmp_path)
    path = tmp_path / "someone_set" / "0_0.json"
    path.parent.mkdir(parents=True)
    (path.parent / "0_0.partial").write_text("{truncated")
    path.write_text(json.dumps(EXAMPLE))
    assert store.card(0, 0).top_logits == [" Paris", " France", " Lyon"]


class Recorder:
    """Stands in for ``requests.get``, replaying a scripted sequence of outcomes."""

    def __init__(self, outcomes: list):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Response:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        import requests

        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}", response=self)


def install(monkeypatch, store: FeatureStore, get) -> list[float]:
    """Point the store at a scripted transport and capture what it would sleep for."""
    import requests

    store._index = {"0": {"filename": "layer_0.bin", "offsets": [0, 100, 250]}}
    slept: list[float] = []
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_url", lambda *a, **k: "https://example/x")
    monkeypatch.setattr("qk_attribution.labels.sleep", slept.append)
    return slept


def test_a_dropped_connection_is_retried(tmp_path: Path, monkeypatch):
    """Label fetches failed intermittently over a long run and those heads were dropped."""
    import requests

    ok = Response(206, make_chunk(EXAMPLE))
    store = FeatureStore("someone/set", cache_dir=tmp_path)
    get = Recorder([requests.ConnectionError("reset"), requests.ConnectionError("reset"), ok])
    slept = install(monkeypatch, store, get)
    assert store.card(0, 1).top_logits == [" Paris", " France", " Lyon"]
    assert get.calls == 3
    assert len(slept) == 2


def test_a_server_error_is_retried(tmp_path: Path, monkeypatch):
    store = FeatureStore("someone/set", cache_dir=tmp_path)
    get = Recorder([Response(503), Response(206, make_chunk(EXAMPLE))])
    install(monkeypatch, store, get)
    assert store.card(0, 1).act_max == 12.5
    assert get.calls == 2


def test_a_missing_feature_is_not_retried(tmp_path: Path, monkeypatch):
    """A 404 will not become a 200; retrying it only slows the run down."""
    import requests

    store = FeatureStore("someone/set", cache_dir=tmp_path)
    get = Recorder([Response(404)])
    install(monkeypatch, store, get)
    with pytest.raises(requests.HTTPError):
        store.card(0, 1)
    assert get.calls == 1


def test_a_full_response_is_not_retried(tmp_path: Path, monkeypatch):
    """Ignoring Range is a property of the server, not a transient fault."""
    store = FeatureStore("someone/set", cache_dir=tmp_path)
    get = Recorder([Response(200, make_chunk(EXAMPLE))])
    install(monkeypatch, store, get)
    with pytest.raises(RuntimeError, match="expected a partial response"):
        store.card(0, 1)
    assert get.calls == 1


def test_retries_are_bounded_and_the_last_error_is_raised(tmp_path: Path, monkeypatch):
    import requests

    store = FeatureStore("someone/set", cache_dir=tmp_path, attempts=3)
    get = Recorder([requests.Timeout("slow")])
    slept = install(monkeypatch, store, get)
    with pytest.raises(requests.Timeout):
        store.card(0, 1)
    assert get.calls == 3
    assert len(slept) == 2


def test_backoff_doubles_between_attempts(tmp_path: Path, monkeypatch):
    import requests

    store = FeatureStore("someone/set", cache_dir=tmp_path, attempts=4, backoff=0.5)
    get = Recorder([requests.ConnectionError("reset")])
    slept = install(monkeypatch, store, get)
    with pytest.raises(requests.ConnectionError):
        store.card(0, 1)
    assert slept == [0.5, 1.0, 2.0]


def test_a_single_attempt_never_sleeps(tmp_path: Path, monkeypatch):
    import requests

    store = FeatureStore("someone/set", cache_dir=tmp_path, attempts=1)
    get = Recorder([requests.ConnectionError("reset")])
    slept = install(monkeypatch, store, get)
    with pytest.raises(requests.ConnectionError):
        store.card(0, 1)
    assert slept == []


@pytest.mark.parametrize("attempts", [0, -1])
def test_a_non_positive_attempt_count_is_rejected(tmp_path: Path, attempts: int):
    with pytest.raises(ValueError, match="attempts"):
        FeatureStore("someone/set", cache_dir=tmp_path, attempts=attempts)
