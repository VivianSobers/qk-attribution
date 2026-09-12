"""Shared loading for the measurement scripts.

Each script started out hardcoding Qwen3-0.6B and the graph from experiment 001. Comparing across
models and transcoder sets means those become arguments, and the loading is identical in every
case, so it lives here once.

Transcoder weights are loaded lazily. The 4B set is 56 GiB and the 14B sets are 125 GiB, so
materialising every layer is not an option on a 24 GB card; only the layers a script actually
indexes are ever read.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch
from circuit_tracer.graph import Graph
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
from transformer_lens import HookedTransformer

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


@dataclass
class Fixtures:
    """Everything a measurement needs, loaded once."""

    graph: Any
    transcoders: Any
    model: Any
    cache: Any
    seq: int
    model_name: str
    graph_path: str
    seed: int
    model_dtype: torch.dtype
    transcoder_dtype: torch.dtype

    def header(self) -> str:
        """A one-line record of what was loaded, for the log alongside the results."""
        return (
            f"model={self.model_name} graph={self.graph_path} seed={self.seed} "
            f"seq={self.seq} model_dtype={self.model_dtype} "
            f"transcoder_dtype={self.transcoder_dtype} scan={self.graph.scan} "
            f"graph_dtype={self.graph.cfg.dtype} features={len(self.graph.selected_features)}"
        )

    def config(self) -> dict[str, Any]:
        """The same record as a dictionary, to be written next to the results."""
        return {
            "model": self.model_name,
            "graph": self.graph_path,
            "scan": self.graph.scan,
            "seed": self.seed,
            "seq": self.seq,
            "model_dtype": str(self.model_dtype),
            "transcoder_dtype": str(self.transcoder_dtype),
            "graph_dtype": str(self.graph.cfg.dtype),
            "n_features": len(self.graph.selected_features),
        }


def parser(description: str) -> argparse.ArgumentParser:
    """Build the argument parser every measurement script shares."""
    parsed = argparse.ArgumentParser(description=description)
    parsed.add_argument("--model", default=None, help="inferred from the graph if omitted")
    parsed.add_argument("--graph", default="spike_out/graph.pt")
    parsed.add_argument("--out", required=True, help="where to write the results as JSON")
    parsed.add_argument("--seed", type=int, default=0)
    parsed.add_argument("--model-dtype", choices=sorted(DTYPES), default="float32")
    parsed.add_argument("--transcoder-dtype", choices=sorted(DTYPES), default="bfloat16")
    return parsed


def load(args: argparse.Namespace) -> Fixtures:
    """Load the graph, its transcoders, the model, and a cached clean run."""
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    graph = Graph.from_pt(args.graph)
    if not isinstance(graph.scan, str):
        raise ValueError(f"a multi-scan graph names several transcoder sets: {graph.scan}")
    model_name = args.model or model_for_scan(graph.scan)
    transcoder_dtype = DTYPES[args.transcoder_dtype]
    model_dtype = DTYPES[args.model_dtype]

    transcoders, _ = load_transcoder_from_hub(
        graph.scan,
        device=torch.device("cuda"),
        dtype=transcoder_dtype,
        lazy_encoder=True,
        lazy_decoder=True,
    )
    model = HookedTransformer.from_pretrained_no_processing(
        model_name, device="cuda", dtype=model_dtype
    )
    tokens = graph.input_tokens.unsqueeze(0).cuda()
    _, cache = model.run_with_cache(tokens)

    fixtures = Fixtures(
        graph=graph,
        transcoders=transcoders,
        model=model,
        cache=cache,
        seq=int(tokens.shape[1]),
        model_name=model_name,
        graph_path=args.graph,
        seed=args.seed,
        model_dtype=model_dtype,
        transcoder_dtype=transcoder_dtype,
    )
    print(fixtures.header())
    return fixtures


def model_for_scan(scan: str) -> str:
    """Name the model a transcoder set was trained against.

    Saves having to remember which graph goes with which model when comparing several.
    """
    for size in ("0.6b", "1.7b", "4b", "8b", "14b"):
        if f"qwen3-{size}-" in scan:
            return f"Qwen/Qwen3-{size.upper()}"
    raise ValueError(f"cannot infer a model from scan {scan!r}; pass --model")


def sampled_layers(model: Any, count: int = 4) -> list[int]:
    """Evenly spaced layers across a model's depth, so comparisons hold across model sizes.

    Layer 0 is excluded: no transcoder feature is written before it, so there is nothing to
    decompose there.
    """
    depth = model.cfg.n_layers
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    step = depth / (count + 1)
    return sorted({max(1, min(depth - 1, round((index + 1) * step))) for index in range(count)})


def sampled_heads(model: Any, count: int = 4) -> list[tuple[int, int]]:
    """One (layer, head) pair per sampled layer, spread across the head index too."""
    heads = model.cfg.n_heads
    return [
        (layer, (index * 5 + 3) % heads) for index, layer in enumerate(sampled_layers(model, count))
    ]
