import sys
import torch
sd = torch.load("weights/best_ema.pt", map_location="cpu", weights_only=False)
if "state_dict" in sd: sd = sd["state_dict"]
tail_w = sd.get("tail.weight", sd.get("module.tail.weight", None))
if tail_w is not None:
    print(f"tail.weight abs sum: {tail_w.abs().sum().item()}")
else:
    print("tail.weight not found in state_dict")
