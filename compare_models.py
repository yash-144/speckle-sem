import sys
import os
import torch
import numpy as np

# Load the validation logic from the current project
from train import build_val, validate, parse_pairs, gaussian_window

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build validation sets (it will use your local val/kla_gt symlinks)
    val_specs = parse_pairs("kla:/kaggle/input/datasets/yashgoyaldev/sem-dataset/train/train/GT")
    valsets = build_val(val_specs, scale=2, max_n=64)
    win = gaussian_window(11, 1.5, device)

    # ---------------------------------------------------------
    # 1. Load the v1 Model (Flat Stack) from sem-model-v2
    # ---------------------------------------------------------
    from model import build_model as build_v1_model
    v1_model = build_v1_model({"c": 64, "n_blocks": 16}).to(device)
    sd_v1 = torch.load("weights/v1/best_ema.pt", map_location="cpu")
    if "state_dict" in sd_v1: sd_v1 = sd_v1["state_dict"]
    sd_v1 = {k.replace("module.", "", 1): v for k, v in sd_v1.items()}
    sd_v1.pop("step", None)
    v1_model.load_state_dict(sd_v1)
    
    print("\n" + "="*50)
    print("Evaluating v1 Model (Flat Stack - sem-model-v2)")
    print("="*50)
    res_v1 = validate(v1_model, valsets, device, win)
    for k, (db, ssim_val, lp_val) in res_v1.items():
        print(f"  {k}: {db:.2f}dB / SSIM: {ssim_val:.4f} / LPIPS: {lp_val:.4f}")

    # ---------------------------------------------------------
    # 2. Load the U-Net Model (NAFNetSR) from unet_model.py
    # ---------------------------------------------------------
    from unet_model import NAFNetSR  # The U-Net style model
    
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
    res_unet = validate(unet_model, valsets, device, win)
    for k, (db, ssim_val, lp_val) in res_unet.items():
        print(f"  {k}: {db:.2f}dB / SSIM: {ssim_val:.4f} / LPIPS: {lp_val:.4f}")
        
    print("\n" + "="*50)
    print("WINNER ANALYSIS (KLA SPLIT)")
    print("="*50)
    v1_kla_psnr, v1_kla_ssim, v1_kla_lpips = res_v1["kla"]
    unet_kla_psnr, unet_kla_ssim, unet_kla_lpips = res_unet["kla"]
    
    psnr_diff = v1_kla_psnr - unet_kla_psnr
    if psnr_diff > 0:
        print(f"PSNR:  v1 Model wins by {psnr_diff:.2f} dB (higher is better)")
    elif psnr_diff < 0:
        print(f"PSNR:  U-Net Model wins by {abs(psnr_diff):.2f} dB (higher is better)")
    else:
        print("PSNR:  Tie!")
        
    ssim_diff = v1_kla_ssim - unet_kla_ssim
    if ssim_diff > 0:
        print(f"SSIM:  v1 Model wins by {ssim_diff:.4f} (higher is better)")
    elif ssim_diff < 0:
        print(f"SSIM:  U-Net Model wins by {abs(ssim_diff):.4f} (higher is better)")
    else:
        print("SSIM:  Tie!")
        
    lpips_diff = unet_kla_lpips - v1_kla_lpips
    if lpips_diff > 0:
        print(f"LPIPS: v1 Model wins by {abs(lpips_diff):.4f} (lower is better)")
    elif lpips_diff < 0:
        print(f"LPIPS: U-Net Model wins by {abs(lpips_diff):.4f} (lower is better)")
    else:
        print("LPIPS: Tie!")

if __name__ == "__main__":
    main()
