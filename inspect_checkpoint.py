import torch
import sys

def inspect_ckpt(path):
    print(f"\nInspecting {path}:")
    try:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Failed to load: {e}")
        return
        
    if "state_dict" in sd: sd = sd["state_dict"]
    
    # Handle the fact that EMA weights might be nested or direct
    keys = list(sd.keys())
    print(f"Top level keys: {keys[:10]}...")
    
    tail_w_key = next((k for k in keys if "tail.weight" in k), None)
    if tail_w_key:
        tail_w = sd[tail_w_key]
        print(f"Found {tail_w_key}, shape: {tail_w.shape}, dtype: {tail_w.dtype}")
        print(f"  abs sum: {tail_w.abs().sum().item():.6e}")
        print(f"  norm:    {tail_w.float().norm().item():.6e}")
        print(f"  max val: {tail_w.abs().max().item():.6e}")
    else:
        print("NO tail.weight found in state dict!")
        
    if "step" in sd:
        print(f"Step: {sd['step']}")

inspect_ckpt("runs/v1/best_ema.pt")
inspect_ckpt("runs/v1/final_raw.pt")
