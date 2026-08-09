import os
import numpy as np
import glob
from PIL import Image

def main():
    in_dir = "val/set5_ood"
    out_dir = "val/set5_ood"
    
    images = glob.glob(os.path.join(in_dir, "*.png"))
    if not images:
        print(f"No images found in {in_dir}")
        return
        
    print(f"Degrading {len(images)} images to build true OOD split...")
    for path in images:
        img = Image.open(path).convert('L')
        img = np.array(img)
        
        h, w = img.shape
        new_h = (h // 16) * 16
        new_w = (w // 16) * 16
        img = img[:new_h, :new_w]
        
        img_norm = img.astype(np.float32) / 255.0
        
        base = os.path.basename(path).replace('.png', '.npy')
        out_path = os.path.join(out_dir, base)
        np.save(out_path, img_norm)
        print(f"Saved {out_path} ({new_h}x{new_w})")

if __name__ == "__main__":
    main()
