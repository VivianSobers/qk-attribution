"""Report the real active-feature counts, which set the QK contraction cost."""
import torch

g = torch.load("spike_out/graph.pt", map_location="cpu", weights_only=False)
active = g["active_features"]
n_pos = len(g["input_tokens"])
cfg = g["cfg"]
print("prompt:", repr(g["input_string"]), "| n_pos", n_pos, "| n_layers", cfg.n_layers)
print("active_features", tuple(active.shape), "dtype", active.dtype)
for col in range(active.shape[1]):
    values = active[:, col]
    print(f"  col {col}: min={values.min().item()} max={values.max().item()} "
          f"unique={values.unique().numel()}")
# Column with max == n_layers-1 is the layer axis; the one with max == n_pos-1 is position.
layer_col = next(c for c in range(3) if active[:, c].max().item() == cfg.n_layers - 1)
pos_col = next(c for c in range(3) if c != layer_col and active[:, c].max().item() == n_pos - 1)
per_pos = torch.bincount(active[:, pos_col], minlength=n_pos)
per_layer = torch.bincount(active[:, layer_col], minlength=cfg.n_layers)
print("layer col", layer_col, "pos col", pos_col)
print("features per position:", per_pos.tolist())
print("features per layer:", per_layer.tolist())
print(f"total {active.shape[0]}, mean per position {active.shape[0] / n_pos:.1f}")
# Query-side features at a position are those in layers below the attention layer.
for layer in (0, 7, 14, 21, 27):
    below = ((active[:, layer_col] < layer)).sum().item()
    print(f"features below layer {layer:2d}: {below} total, {below / n_pos:.1f} per position")
