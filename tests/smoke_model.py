"""Remote-friendly model load and inference smoke test."""
import sys
import torch
from checkpoint import load_model

path = sys.argv[1]
model, device, _ = load_model(path)
output = model(torch.zeros(1, 1, 28, 28, device=device))
assert tuple(output.shape) == (1, 10)
assert bool(torch.isfinite(output).all())
print(f"device={device} shape={tuple(output.shape)} finite=True")
