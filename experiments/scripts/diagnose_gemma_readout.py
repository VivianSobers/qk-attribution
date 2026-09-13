"""Test one hypothesis about the remaining Gemma-2-2B disagreement after the ln1_post fix.

Hypothesis: the GemmaScope transcoders read ``ln2.hook_normalized``, which in TransformerLens is the
input divided by its RMS scale *before* the norm's gain ``w`` is applied, so the readout should
not multiply by ``w``. The module treats that hook as equivalent to the MLP input and applies ``w``.

Four readouts per edge, from the same propagated perturbation, crossing the two possible causes:
with or without ln1_post applied, and with or without ln2's gain in the readout.
"""

import importlib.util
import sys
from types import SimpleNamespace

import torch
from circuit_tracer.graph import Graph
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
from transformer_lens import HookedTransformer


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


torch.set_grad_enabled(False)
torch.manual_seed(0)
unfixed = load("hl_unfixed", "head_loadings_branch.py")
fixed = load("hl_fixed", "fixed/head_loadings_branch.py")

graph = Graph.from_pt("spike_out/gemma-2-2b-float32.pt")
transcoders, _ = load_transcoder_from_hub(
    graph.scan,
    device=torch.device("cuda"),
    dtype=torch.float32,
    lazy_encoder=True,
    lazy_decoder=True,
)
backbone = HookedTransformer.from_pretrained_no_processing(
    "google/gemma-2-2b", device="cuda", dtype=torch.float32
)
model = SimpleNamespace(
    cfg=backbone.cfg, blocks=backbone.blocks, W_E=backbone.W_E, transcoders=transcoders
)
tokens = graph.input_tokens.unsqueeze(0).cuda()
runs = {"unfixed": unfixed.FrozenRun.from_model(backbone, tokens)}
runs["fixed"] = fixed.FrozenRun.from_model(backbone, tokens)
modules = {"unfixed": unfixed, "fixed": fixed}


def readouts(target, source):
    out = {}
    for name, module in modules.items():
        run = runs[name]
        vector, position, layer = module.source_vector(model, graph, source)
        reader, target_position, target_layer = module.reader_vector(model, graph, target)
        delta = module._seed(vector, position, run.n_pos)
        arriving = module._propagate(model, delta, run, layer + 1, target_layer)
        mid = arriving + module._attention_step(model, target_layer, arriving, run)
        norm = model.blocks[target_layer].ln2
        scale = run.ln2_scales[target_layer]
        out[f"{name}, with w"] = float((mid * norm.w / scale)[target_position] @ reader)
        out[f"{name}, no w"] = float((mid / scale)[target_position] @ reader)
    return out


layout = unfixed.NodeLayout.from_graph(graph)
adjacency = graph.adjacency_matrix
layers = graph.active_features[graph.selected_features][:, 0]
block_adj = adjacency[: layout.n_features, : layout.n_features]
mask = (block_adj.abs() > 1e-4) & (layers.unsqueeze(1) > layers.unsqueeze(0))
pairs = torch.nonzero(mask)
strongest = block_adj[mask].abs().argsort(descending=True)[:20].tolist()
sampled = torch.randperm(len(pairs))[:20].tolist()

ratios: dict[str, list[float]] = {}
print(" target source src tgt  adjacency | unfixed,w  unfixed,-w  fixed,w  fixed,-w")
for index in strongest + sampled:
    target, source = (int(x) for x in pairs[index])
    expected = float(adjacency[target, source])
    got = readouts(target, source)
    for key, value in got.items():
        ratios.setdefault(key, []).append(value / expected)
    print(
        f"{target:7d} {source:6d} {int(layers[source]):3d} {int(layers[target]):3d} {expected:+10.5f} | "
        + "  ".join(f"{got[k] / expected:9.4f}" for k in got)
    )

print("\nsummary over", len(strongest) + len(sampled), "edges (20 strongest, 20 random, seed 0)")
for key, values in ratios.items():
    t = torch.tensor(values)
    dev = (t - 1).abs()
    print(
        f"{key:16s} median {t.median():.6f}  within 1e-3 {int((dev < 1e-3).sum()):2d}"
        f"  within 1% {int((dev < 0.01).sum()):2d}  max |ratio-1| {dev.max():.2e}"
    )
