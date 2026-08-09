#!/usr/bin/env python3
"""
pack_data.py

Packs a directory of images into a single (N, H, W) float32 .npy file.
This allows the dataloader to use mmap_mode='r' and bypass OS filesystem I/O
bottlenecks, which is crucial for maximizing Kaggle GPU utilization.
"""

import os
import glob
import numpy as np
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source directory containing images")
    parser.add_argument("--out", required=True, help="Output .npy file path")
    parser.add_argument("--skip", type=int, default=0, help="Number of sorted files to skip from the beginning (e.g. 64 to exclude validation set)")
    args = parser.parse_args()

    fs = glob.glob(os.path.join(args.src, "**", "*"), recursive=True)
    fs = [f for f in fs if os.path.isfile(f) and f.lower().endswith((".png", ".jpg", ".npy", ".tif", ".bmp", ".jpeg"))]
    fs.sort()
    
    if args.skip > 0:
        print(f"Skipping first {args.skip} files (e.g. held out for validation).")
        fs = fs[args.skip:]

    print(f"Packing {len(fs)} images from {args.src} -> {args.out}")
    if not fs:
        return
    
    arrays = []
    for f in fs:
        if f.lower().endswith(".npy"):
            a = np.load(f)
            if a.ndim == 3: a = a[..., 0]
            arrays.append(a.astype(np.float32))
        else:
            a = cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
            if a is None: continue
            if a.ndim == 3: a = a[..., 0]
            if a.dtype == np.uint8: scale = 255.0
            elif a.dtype == np.uint16: scale = 65535.0
            else: scale = 1.0
            arrays.append((a.astype(np.float32) / scale))
            
    sizes = set([a.shape for a in arrays])
    if len(sizes) != 1:
        print(f"Error: dataset has mixed shapes {sizes}. Packer requires uniform shapes.")
        return

    stacked = np.stack(arrays, axis=0)
    print(f"Stacked shape: {stacked.shape}, dtype: {stacked.dtype}, size: {stacked.nbytes / 1024 / 1024:.1f} MB")
    
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    np.save(args.out, stacked)
    print("Done.")

if __name__ == "__main__":
    main()
