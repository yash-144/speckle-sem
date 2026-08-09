#!/usr/bin/env python3
"""
verify_outputs.py -- run this after every evaluate.py run. It answers the three
questions the timing log cannot:

  1. Did the checkpoint actually load, or is evaluate.py emitting pure bicubic?
     (strict=False + mismatched --channels/--blocks fails SILENTLY. The
     zero-init tail means an unloaded model is bitwise identical to bicubic,
     which makes this test decisive rather than heuristic.)

  2. Does every input have exactly one output with the IDENTICAL filename and
     extension? KLA matches by filename. A rename is a zero.

  3. Do dtype and shape round-trip? uint16 in must be uint16 out, and the
     output must be exactly scale x the input.

Usage:
    python verify_outputs.py --input_dir data/test_lr --output_dir out/ --scale 2

Exit code 0 = all checks passed. Non-zero = do not submit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

EXTS = {".npy", ".png", ".tif", ".tiff"}


def load(p: Path):
    if p.suffix.lower() == ".npy":
        return np.load(p)
    a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise IOError(f"unreadable: {p}")
    return a


def to_float(a: np.ndarray) -> np.ndarray:
    """Put any dtype on a comparable float scale."""
    a = a.astype(np.float32)
    if a.dtype == np.uint8 or a.max() > 300:      # uint8 / uint16 heuristic
        a = a / (255.0 if a.max() <= 255 else 65535.0)
    return a


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--bicubic_psnr_floor", type=float, default=45.0,
                    help="output-vs-bicubic PSNR above this = checkpoint "
                         "almost certainly did not load")
    args = ap.parse_args()

    idir, odir = Path(args.input_dir), Path(args.output_dir)
    ins = sorted(p for p in idir.rglob("*") if p.suffix.lower() in EXTS)
    if not ins:
        print(f"FAIL: no inputs found under {idir}")
        return 2

    fails: list[str] = []
    bic_psnrs: list[float] = []
    omin, omax = float("inf"), float("-inf")

    for ip in ins:
        op = odir / ip.name                      # identical name AND extension
        if not op.exists():
            fails.append(f"MISSING OUTPUT: {ip.name}")
            continue

        a_in, a_out = load(ip), load(op)

        if a_in.dtype != a_out.dtype:
            fails.append(f"DTYPE: {ip.name} {a_in.dtype} -> {a_out.dtype}")

        want = (a_in.shape[0] * args.scale, a_in.shape[1] * args.scale)
        if a_out.shape[:2] != want:
            fails.append(f"SHAPE: {ip.name} got {a_out.shape[:2]}, want {want}")
            continue

        f_in, f_out = to_float(a_in), to_float(a_out)
        omin, omax = min(omin, f_out.min()), max(omax, f_out.max())

        bic = cv2.resize(f_in, (want[1], want[0]), interpolation=cv2.INTER_CUBIC)
        bic_psnrs.append(psnr(np.clip(f_out, 0, 1), np.clip(bic, 0, 1)))

    extras = {p.name for p in odir.rglob("*") if p.suffix.lower() in EXTS}
    extras -= {p.name for p in ins}
    if extras:
        fails.append(f"UNEXPECTED OUTPUT FILES ({len(extras)}): "
                     f"{sorted(extras)[:5]}")

    print(f"inputs={len(ins)}  compared={len(bic_psnrs)}")
    print(f"output value range: [{omin:.4f}, {omax:.4f}]")

    if bic_psnrs:
        med = float(np.median(bic_psnrs))
        print(f"output-vs-bicubic PSNR: median {med:.2f} dB  "
              f"(min {min(bic_psnrs):.2f}, max {max(bic_psnrs):.2f})")
        if med > args.bicubic_psnr_floor:
            fails.append(
                f"OUTPUT IS ~BICUBIC (median {med:.2f} dB vs bicubic). The "
                f"checkpoint did not load. Check --channels/--blocks against "
                f"the trained model and confirm no 'missing keys' warning."
            )
        else:
            print("-> checkpoint loaded: output is meaningfully "
                  "different from bicubic.")

    if fails:
        print("\n=== FAILED ===")
        for f in fails[:40]:
            print(" ", f)
        if len(fails) > 40:
            print(f"  ... and {len(fails) - 40} more")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
