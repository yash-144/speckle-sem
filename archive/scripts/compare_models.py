import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np

# Load the validation logic from the current project
from train import build_val, validate, parse_pairs, gaussian_window

import time

import torch.nn.functional as F

def measure_latency(model, device, num_runs=50):
    # End-to-end inference latency on a standard input (e.g., 512x512)
    # Includes CPU->GPU, padding, forward, unpadding, GPU->CPU
    probe_cpu = torch.rand(1, 1, 512, 512)
    
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            t = probe_cpu.to(device)
            h, w = t.shape[-2:]
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            if pad_h > 0 or pad_w > 0:
                t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
            out = model(t)
            if pad_h > 0 or pad_w > 0:
                out = out[..., :2*h, :2*w]
            _ = out.cpu()
            
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(num_runs):
            t = probe_cpu.to(device)
            h, w = t.shape[-2:]
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            if pad_h > 0 or pad_w > 0:
                t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
            out = model(t)
            if pad_h > 0 or pad_w > 0:
                out = out[..., :2*h, :2*w]
            _ = out.cpu()
            
        if device.type == "cuda": torch.cuda.synchronize()
        return (time.time() - t0) / num_runs * 1000

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build validation sets (it will use your local val/kla_gt symlinks)
    val_specs = parse_pairs("kla:/kaggle/input/datasets/yashgoyaldev/sem-dataset/train/train/GT,layouts_holdout:val/layouts_holdout,set5_ood:val/set5_ood")
    valsets = build_val(val_specs, scale=2)
    win = gaussian_window(11, 1.5, device)

    # ---------------------------------------------------------
    # 1. Load the v1 Model (Flat Stack) from sem-model-v2
    # ---------------------------------------------------------
    from model import build_model as build_v1_model
    v1_model = build_v1_model({"c": 64, "n_blocks": 16}).to(device)
    sd_v1 = torch.load("weights/best_ema.pt", map_location="cpu")
    if "state_dict" in sd_v1: sd_v1 = sd_v1["state_dict"]
    sd_v1 = {k.replace("module.", "", 1): v for k, v in sd_v1.items()}
    sd_v1.pop("step", None)
    v1_model.load_state_dict(sd_v1)
    
    print("\n" + "="*50)
    print("Evaluating v1 Model (Flat Stack - sem-model-v2)")
    print("="*50)
    res_v1 = validate(v1_model, valsets, device, win, lpips_nets=("alex", "vgg"))
    for split, metrics in res_v1.items():
        print(f"  {split}: {metrics['psnr']:.2f}dB / SSIM: {metrics['ssim']:.4f} / LPIPS-alex: {metrics['lpips_alex']:.4f} / LPIPS-vgg: {metrics['lpips_vgg']:.4f}")
    
    v1_latency = measure_latency(v1_model, device)
    print(f"  Latency: {v1_latency:.2f} ms")

    # ---------------------------------------------------------
    # 2. Load the U-Net Model (NAFNetSR) from unet_model.py
    # ---------------------------------------------------------
    from archive.unet_model import NAFNetSR  # The U-Net style model
    
    unet_model = NAFNetSR(channels=1, width=32, scale=2, hr_blocks=0).to(device)
    
    unet_ckpt_path = "model_A_p96.pth"
    if not os.path.exists(unet_ckpt_path):
        print(f"\nWARNING: '{unet_ckpt_path}' not found!")
        print("Please ensure model_A_p96.pth is in the working directory.")
        return

    sd_unet = torch.load(unet_ckpt_path, map_location="cpu")
    if "state_dict" in sd_unet: sd_unet = sd_unet["state_dict"]
    elif "model" in sd_unet: sd_unet = sd_unet["model"]
    sd_unet = {k.replace("module.", "", 1): v for k, v in sd_unet.items()}
    unet_model.load_state_dict(sd_unet)

    class ClampingWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x, clamp=False):
            out = self.m(x)
            return out.clamp(0.0, 1.0) if clamp else out

    unet_model = ClampingWrapper(unet_model)

    print("\n" + "="*50)
    print("Evaluating U-Net Model (NAFNetSR - sem-model)")
    print("="*50)
    res_unet = validate(unet_model, valsets, device, win, lpips_nets=("alex", "vgg"))
    for split, metrics in res_unet.items():
        print(f"  {split}: {metrics['psnr']:.2f}dB / SSIM: {metrics['ssim']:.4f} / LPIPS-alex: {metrics['lpips_alex']:.4f} / LPIPS-vgg: {metrics['lpips_vgg']:.4f}")

    unet_latency = measure_latency(unet_model, device)
    print(f"  Latency: {unet_latency:.2f} ms")
        
    print("\n" + "="*50)
    print("WINNER ANALYSIS")
    print("="*50)
    
    for split in res_v1.keys():
        print(f"\n--- Split: {split} ---")
        v1_m = res_v1[split]
        unet_m = res_unet[split]
        
        psnr_diff = v1_m["psnr"] - unet_m["psnr"]
        if psnr_diff > 0:
            print(f"PSNR:       v1 Model wins by {psnr_diff:.2f} dB (higher is better)")
        elif psnr_diff < 0:
            print(f"PSNR:       U-Net Model wins by {abs(psnr_diff):.2f} dB (higher is better)")
        else:
            print("PSNR:       Tie!")
            
        ssim_diff = v1_m["ssim"] - unet_m["ssim"]
        if ssim_diff > 0:
            print(f"SSIM:       v1 Model wins by {ssim_diff:.4f} (higher is better)")
        elif ssim_diff < 0:
            print(f"SSIM:       U-Net Model wins by {abs(ssim_diff):.4f} (higher is better)")
        else:
            print("SSIM:       Tie!")
            
        lpips_a_diff = unet_m["lpips_alex"] - v1_m["lpips_alex"]
        if lpips_a_diff > 0:
            print(f"LPIPS-alex: v1 Model wins by {abs(lpips_a_diff):.4f} (lower is better)")
        elif lpips_a_diff < 0:
            print(f"LPIPS-alex: U-Net Model wins by {abs(lpips_a_diff):.4f} (lower is better)")
        else:
            print("LPIPS-alex: Tie!")
            
        lpips_v_diff = unet_m["lpips_vgg"] - v1_m["lpips_vgg"]
        if lpips_v_diff > 0:
            print(f"LPIPS-vgg:  v1 Model wins by {abs(lpips_v_diff):.4f} (lower is better)")
        elif lpips_v_diff < 0:
            print(f"LPIPS-vgg:  U-Net Model wins by {abs(lpips_v_diff):.4f} (lower is better)")
        else:
            print("LPIPS-vgg:  Tie!")

    print("\n--- Latency (End-to-End Inference: CPU -> Pad -> GPU -> Unpad -> CPU) ---")
    lat_diff = unet_latency - v1_latency
    if lat_diff > 0:
        print(f"v1 Model is faster by {lat_diff:.2f} ms ({v1_latency:.2f} vs {unet_latency:.2f})")
    else:
        print(f"U-Net Model is faster by {abs(lat_diff):.2f} ms ({unet_latency:.2f} vs {v1_latency:.2f})")

if __name__ == "__main__":
    main()
