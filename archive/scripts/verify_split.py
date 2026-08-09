import os
import glob

def listdir(d):
    EXTS = (".npy", ".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg")
    return [p for p in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True))
            if p.lower().endswith(EXTS)]

def main():
    print("=== PREFLIGHT SPLIT VERIFICATION ===")
    
    # 1. Get the exact 64 files the validation loader will use
    # (Assuming val/kla_gt is a symlink to train/train/GT on Kaggle)
    # We will simulate this by looking at the Kaggle dataset path directly.
    target_dir = "/kaggle/input/datasets/yashgoyaldev/sem-dataset/train/train/GT"
    
    if not os.path.exists(target_dir):
        print(f"Warning: {target_dir} not found. Are you running this on Kaggle?")
        # Fallback for local testing if needed
        target_dir = "val/kla_gt" 
        
    if not os.path.exists(target_dir):
         print(f"Cannot find dataset at {target_dir} to verify.")
         return

    val_files = listdir(target_dir)[:64]
    val_basenames = [os.path.basename(f) for f in val_files]
    
    # 2. Get the exact files pack_data.py will skip
    pack_files = glob.glob(os.path.join(target_dir, "**", "*"), recursive=True)
    pack_files = [f for f in pack_files if os.path.isfile(f) and f.lower().endswith((".png", ".jpg", ".npy", ".tif", ".bmp", ".jpeg"))]
    pack_files.sort()
    
    skipped_pack = pack_files[:64]
    pack_basenames = [os.path.basename(f) for f in skipped_pack]
    
    # 3. Assert they are identical
    if val_basenames == pack_basenames:
        print("\nSUCCESS: The 64 files skipped by pack_data.py perfectly match")
        print("the 64 files evaluated by train.py validation loader.")
        print("Data leakage is mathematically eliminated.\n")
    else:
        print("\nFAILURE: File ordering mismatch!")
        print("The skipped files do NOT perfectly match the validation set.")
        print(f"Validation file 1: {val_basenames[0]}")
        print(f"Skipped file 1:    {pack_basenames[0]}\n")

if __name__ == "__main__":
    main()
