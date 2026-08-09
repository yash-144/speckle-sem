import os
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from model import build_model
from lpips_metric import lpips_score

def compute_bicubic_set5():
    print("--- 1. Computing Bicubic PSNR on Set5 OOD ---")
    val_dir = "val/set5_ood"
    from degradation_engine import Degrader
    deg = Degrader()
    
    psnrs = []
    for f in os.listdir(val_dir):
        if not f.endswith(".npy"): continue
        gt = np.load(os.path.join(val_dir, f))
        if gt.ndim == 3: gt = gt[..., 0]
        
        lr, gt_mod = deg(gt)
        lr_t = torch.from_numpy(lr)[None, None]
        bicubic = F.interpolate(lr_t, scale_factor=2, mode="bicubic", align_corners=False)
        bicubic = bicubic.clamp(0, 1)[0, 0].numpy()
        
        mse = np.mean((gt_mod - bicubic)**2)
        psnr = -10 * np.log10(max(mse, 1e-10))
        psnrs.append(psnr)
        
    avg_psnr = np.mean(psnrs)
    print(f"Bicubic PSNR on set5_ood: {avg_psnr:.2f} dB")
    return avg_psnr

def compute_flops_and_vram():
    print("\n--- 2 & 3. Computing GFLOPs and Peak VRAM ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.eval()
    
    # Batch size 1 for FLOPs, Batch size 8 for VRAM throughput baseline
    dummy_in_1 = torch.randn(1, 1, 256, 256).to(device)
    dummy_in_8 = torch.randn(8, 1, 256, 256).to(device)
    
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, dummy_in_1)
        gflops = flops.total() / 1e9
        print(f"GFLOPs (256x256 input): {gflops:.2f}")
    except ImportError:
        print("fvcore not installed. (Run !pip install fvcore for GFLOPs)")
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                _ = model(dummy_in_8)
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak VRAM (bs=8, fp16, 256x256 input): {peak_mb:.1f} MB")
    else:
        print("CUDA not available, skipping VRAM.")

def compute_latency():
    print("\n--- 4. Computing 256->512 Latency (evaluate.py proxy) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.eval()
    
    bs = 8
    dummy_in = torch.randn(bs, 1, 256, 256).to(device)
    
    if device.type == "cuda":
        with torch.no_grad(), torch.amp.autocast('cuda'):
            # Warmup
            for _ in range(5):
                _ = model(dummy_in)
            torch.cuda.synchronize()
            
            # Time
            t0 = time.time()
            iters = 50
            for _ in range(iters):
                _ = model(dummy_in)
            torch.cuda.synchronize()
            
            el = (time.time() - t0) / (iters * bs)
            print(f"256->512 Throughput Latency: {el*1000:.2f} ms/image")

def main():
    compute_bicubic_set5()
    compute_flops_and_vram()
    compute_latency()

if __name__ == "__main__":
    main()
