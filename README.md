# Speckle SEM Image Restoration

This repository contains the solution for the KLA Semiconductor Inspection Hackathon. Our pipeline implements a 16-block NAFNet architecture designed to rapidly denoise and super-resolve Scanning Electron Microscope (SEM) imagery affected by quantum shot noise (speckle) and structural degradation.

## Quickstart: Inference

The `evaluate.py` script is designed for zero-configuration, robust inference against any directory of degraded SEM images. It is built to seamlessly handle `.png`, `.npy`, `.tif`, and `.jpg` inputs of arbitrary dimensions and depths without memory crashes or manual tiling.

### 1. Environment Setup

```bash
git clone https://github.com/yash-144/speckle-sem.git
cd speckle-sem
pip install -r requirements.txt
```

### 2. Run Inference

Point the evaluation script at a directory containing the test images. The restored images will be written to the output directory using the identical filenames, bit-depths, and formats as the inputs.

```bash
python evaluate.py --input_dir /path/to/test_data --output_dir /path/to/predictions
```

**Note:** The model uses `weights/best_ema.pt` by default. If it encounters a file it cannot read or a memory exception it cannot handle via automatic batch-size backoff, it will gracefully fallback to emitting a bicubic upsampled version of the input, guaranteeing a complete submission.

## Training Pipeline

The training pipeline uses on-the-fly synthetic degradations mapped closely to real-world KLA detector noise profiles (speckle). It trains over a dual-source dataset combining authentic KLA organic captures and procedural semiconductor layout generations to guarantee robust generalization.

### Data Generation and Packing
To maximize I/O throughput on constrained compute environments, training images should be packed into `mmap`-friendly `.npy` structures.

```bash
# 1. Generate OOD procedural holdouts and layout primitives
python layout_generator.py --n 2000 --out data/layouts --size 256
python layout_generator.py --n 100 --out val/layouts_holdout --size 256

# 2. Pack data arrays
python pack_data.py --src data/kla_gt --out data/kla_packed.npy
python pack_data.py --src data/layouts --out data/layouts_packed.npy
```

### Execution
```bash
python train.py \
  --sources "data/kla_packed.npy:60,data/layouts_packed.npy:40" \
  --val "kla:val/kla_gt,layouts_unseen:val/layouts_holdout" \
  --steps 40000 --bs 16 --patch 128 --workers 4 \
  --val_every 500 --save_every 1000 --out runs/v1
```
