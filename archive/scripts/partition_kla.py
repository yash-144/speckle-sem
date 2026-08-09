import os
import glob
import random
import shutil

def main():
    src_dir = "val/kla_gt"
    dst_dir = "val/kla_holdout"
    
    os.makedirs(dst_dir, exist_ok=True)
    
    images = glob.glob(os.path.join(src_dir, "*.npy"))
    
    if len(images) < 2000 and os.path.exists(dst_dir) and len(os.listdir(dst_dir)) > 0:
        print(f"kla_holdout already populated with {len(os.listdir(dst_dir))} images.")
        return
        
    random.seed(42)
    random.shuffle(images)
    
    holdout = images[:320]
    for path in holdout:
        base = os.path.basename(path)
        shutil.move(path, os.path.join(dst_dir, base))
        
    print(f"Moved {len(holdout)} images to {dst_dir}. {len(images)-len(holdout)} images left in {src_dir}.")

if __name__ == "__main__":
    main()
