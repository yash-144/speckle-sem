#!/usr/bin/env python3
"""
evaluate.py

Runs the trained restoration model over a directory of degraded images and
writes restored outputs. This is the file KLA's benchmarking team executes
AS-IS on an H100 - it must run with no manual edits.

    python evaluate.py --input_dir /path/to/test --output_dir /path/to/out

Accepts --input_dir/--output_dir, --input/--output, -i/-o, or two positional
arguments, because we do not control how it will be invoked.

Design choices that exist purely for robustness:
  * The network is fully convolutional with no downsampling, so any input
    size works with no padding or tiling logic.
  * Output filenames, extensions and bit depths are preserved exactly.
    Renaming breaks filename-based ground-truth matching.
  * Per-image try/except with a bicubic fallback - one unreadable file must
    not zero the whole submission.
  * Automatic batch-size backoff on OOM, down to single images.
  * Dependencies are torch, numpy, opencv-python only.

Deps: torch, numpy, opencv-python. model.py must sit alongside this file.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model

EXTS = (".npy", ".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg")
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "weights", "best_ema.pt")


def find_inputs(d):
    if os.path.isfile(d):
        return [d]
    fs = [p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
          if os.path.isfile(p) and p.lower().endswith(EXTS)]
    return sorted(fs)


def read_image(path):
    """Return (float32 HxW array, metadata for writing it back out)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        a = np.load(path)
        if a.ndim == 3:
            a = a[..., 0]
        return a.astype(np.float32), {"kind": "npy", "dtype": a.dtype}

    import cv2
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if a is None:
        raise IOError(f"unreadable: {path}")
    if a.ndim == 3:
        a = a[..., 0]
    dt = a.dtype
    if dt == np.uint8:
        scale = 255.0
    elif dt == np.uint16:
        scale = 65535.0
    else:
        scale = 1.0
    return a.astype(np.float32) / scale, {"kind": "img", "dtype": dt,
                                          "scale": scale}


def write_image(path, arr, meta):
    if meta["kind"] == "npy":
        np.save(path, arr.astype(np.float32))
        return
    import cv2
    dt, scale = meta["dtype"], meta["scale"]
    if dt in (np.uint8, np.uint16):
        out = np.clip(arr * scale + 0.5, 0, scale).astype(dt)
    else:
        out = arr.astype(dt)
    if not cv2.imwrite(path, out):
        raise IOError(f"failed to write {path}")


@torch.no_grad()
def run_batch(model, arrs, device, use_amp):
    t = torch.from_numpy(np.stack(arrs)[:, None]).to(device)
    h, w = t.shape[-2:]
    
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    
    if pad_h > 0 or pad_w > 0:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
        
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        out = model(t)
        if hasattr(out, "clamp"):
            out = out.clamp(0.0, 1.0)
            
    if pad_h > 0 or pad_w > 0:
        out = out[..., :2*h, :2*w]
        
    return out.float().cpu().numpy()[:, 0]


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--input_directory", "--input_dir", "--input", "-i", dest="input_dir")
    ap.add_argument("--output_directory", "--output_dir", "--output", "-o", dest="output_dir")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("pos", nargs="*", help="input_dir output_dir (positional)")
    args = ap.parse_args()

    inp, outp = args.input_dir, args.output_dir
    if inp is None and len(args.pos) >= 1:
        inp = args.pos[0]
    if outp is None and len(args.pos) >= 2:
        outp = args.pos[1]
    if not inp or not outp:
        ap.error("need an input and an output directory")
    inp, outp = inp.rstrip("/\\"), outp.rstrip("/\\")
    os.makedirs(outp, exist_ok=True)

    files = find_inputs(inp)
    if not files:
        print(f"no images found under {inp}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and not args.fp32
    torch.backends.cudnn.benchmark = True

    from pathlib import Path
    ckpt = Path(args.weights)
    if not ckpt.is_absolute():
        ckpt = Path(__file__).resolve().parent / ckpt
    if not ckpt.exists():
        raise SystemExit(f"FATAL: no checkpoint at {ckpt}")

    sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    
    sd.pop("step", None)
    
    C = sd["head.weight"].shape[0]
    B = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("body."))
    
    model = build_model({"c": C, "n_blocks": B})

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"FATAL: checkpoint/arch mismatch.\n"
            f"  missing({len(missing)}): {missing[:5]}\n"
            f"  unexpected({len(unexpected)}): {unexpected[:5]}\n"
            f"  inferred channels={C} blocks={B}")
    print(f"loaded {ckpt} ({ckpt.stat().st_size/1e6:.2f} MB), "
          f"channels={C} blocks={B}")
    model.eval().to(device)

    with torch.no_grad():
        probe = torch.rand(1, 1, 64, 64, device=device)
        bic = F.interpolate(probe, scale_factor=2, mode="bicubic", align_corners=False)
        delta = (model(probe).float() - bic).abs().max().item()
    if delta < 1e-6:
        raise SystemExit(f"FATAL: output identical to bicubic (delta {delta:.2e}) "
                         "— weights did not load.")
    print(f"self-test OK: deviates from bicubic by {delta:.4f}")

    print(f"device={device}  amp={use_amp}  images={len(files)}")

    # group by shape so batching is valid; preserves nothing about order
    groups = {}
    metas, arrays = {}, {}
    for f in files:
        try:
            a, m = read_image(f)
        except Exception as e:
            print(f"warning: {e}", file=sys.stderr)
            continue
        arrays[f], metas[f] = a, m
        groups.setdefault(a.shape, []).append(f)

    t0 = time.time()
    done, failed = 0, 0
    for shape, fs in groups.items():
        bs = max(1, args.batch)
        
        # Warmup for this shape bucket
        if device.type == "cuda":
            with torch.no_grad():
                for _ in range(2):
                    _ = run_batch(model, [arrays[fs[0]]], device, use_amp)
            torch.cuda.synchronize()
            
        i = 0
        while i < len(fs):
            chunk = fs[i:i + bs]
            try:
                outs = run_batch(model, [arrays[f] for f in chunk], device, use_amp)
                if device.type == "cuda": torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs > 1:
                    bs = max(1, bs // 2)
                    print(f"  OOM at {shape}, dropping batch to {bs}",
                          file=sys.stderr)
                    continue
                outs = None
            except Exception as e:
                print(f"warning: inference failed on {chunk[0]}: {e}",
                      file=sys.stderr)
                outs = None

            for j, f in enumerate(chunk):
                dst = os.path.join(outp, os.path.basename(f))
                try:
                    if outs is not None:
                        arr = outs[j]
                    else:  # last-resort fallback: never emit nothing
                        t = torch.from_numpy(arrays[f])[None, None]
                        arr = F.interpolate(t, scale_factor=2, mode="bicubic",
                                            align_corners=False
                                            ).clamp(0, 1)[0, 0].numpy()
                        failed += 1
                    write_image(dst, arr, metas[f])
                    done += 1
                except Exception as e:
                    print(f"warning: could not write {dst}: {e}", file=sys.stderr)
            i += len(chunk)

    if device.type == "cuda":
        torch.cuda.synchronize()
    el = time.time() - t0
    print(f"wrote {done}/{len(files)} images in {el:.2f}s "
          f"({1000*el/max(done,1):.1f} ms/image)"
          + (f"  [{failed} bicubic fallbacks]" if failed else ""))
    with open(os.path.join(outp, "_timing.json"), "w") as fh:
        json.dump({"images": done, "total_seconds": el,
                   "ms_per_image": 1000 * el / max(done, 1),
                   "device": str(device), "amp": use_amp,
                   "fallbacks": failed}, fh, indent=2)


if __name__ == "__main__":
    main()
