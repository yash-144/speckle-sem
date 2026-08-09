import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys

from train import parse_pairs, build_val
from model import build_model as build_v1_model
from unet_model import NAFNetSR

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_specs = parse_pairs("layouts_unseen:val/layouts_holdout")
    valsets = build_val(val_specs, scale=2)
    pairs = valsets["layouts_unseen"]
    
    print("Loading models...")
    # Load v1
    v1_model = build_v1_model({"c": 64, "n_blocks": 16}).to(device)
    sd_v1 = torch.load("weights/v1/best_ema.pt", map_location="cpu")
    if "state_dict" in sd_v1: sd_v1 = sd_v1["state_dict"]
    sd_v1 = {k.replace("module.", "", 1): v for k, v in sd_v1.items()}
    v1_model.load_state_dict(sd_v1, strict=False)
    v1_model.eval()

    # Load U-Net
    unet_model = NAFNetSR(channels=1, width=32, scale=2, hr_blocks=0).to(device)
    sd_unet = torch.load("model_A_p96.pth", map_location="cpu")
    if "state_dict" in sd_unet: sd_unet = sd_unet["state_dict"]
    elif "model" in sd_unet: sd_unet = sd_unet["model"]
    sd_unet = {k.replace("module.", "", 1): v for k, v in sd_unet.items()}
    unet_model.load_state_dict(sd_unet, strict=False)
    unet_model.eval()

    # Pick top 5 pairs
    samples = pairs[:5]
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    
    col_titles = ["Bicubic", "GT", "v1 Model", "U-Net"]
    for i in range(4):
        axes[0, i].set_title(col_titles[i], fontsize=16)

    print("Generating visual crops...")
    with torch.no_grad():
        for row, (lr, gt) in enumerate(samples):
            lr_t = lr.unsqueeze(0).to(device)
            
            # Predict
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out_v1 = v1_model(lr_t).clamp(0, 1)[0].cpu()
                out_unet = unet_model(lr_t).clamp(0, 1)[0].cpu()
                
            bicubic = F.interpolate(lr_t, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)[0].cpu()
            
            # Center crop 128x128
            def crop(t):
                h, w = t.shape[1], t.shape[2]
                # Fallback if image is smaller than 128x128
                if h <= 128 or w <= 128:
                    return t[0].numpy()
                return t[0, h//2 - 64: h//2 + 64, w//2 - 64: w//2 + 64].numpy()
                
            imgs = [crop(bicubic), crop(gt), crop(out_v1), crop(out_unet)]
            for col, img in enumerate(imgs):
                ax = axes[row, col]
                ax.imshow(img, cmap="gray")
                ax.axis("off")
                
    plt.tight_layout()
    plt.savefig("hallucination_check.png", dpi=150)
    print("Saved hallucination_check.png")

if __name__ == "__main__":
    main()
