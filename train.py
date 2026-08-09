#!/usr/bin/env python3
"""
train.py

Trains RestoreNet on a weighted mixture of image sources, degrading on the fly
with the measured KLA degradation operator.

    python train.py \
      --sources "data/kla_gt:20,data/layouts:25,data/nffa:40,data/df2k:15" \
      --val "kla:val/kla_gt,layouts:val/layouts,mems:val/nffa_mems" \
      --steps 40000 --bs 32 --patch 128 --out runs/v1

Sources are "dir:weight" pairs; weights are relative sampling probabilities.
Validation sets are "name:dir" pairs of GROUND TRUTH images - they are degraded
deterministically at inference time so the numbers are comparable across runs.

Report every validation split separately. The split containing content the
model has never seen is the one that predicts the OOD half of the test set.

Deps: torch, numpy, opencv-python. (degradation_engine.py, model.py alongside.)
"""

import argparse
import glob
import math
import os
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import build_model
from degradation_engine import Degrader

EXTS = (".npy", ".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg")
BANNER_FRAC = 0.10   # NFFA instrument banner along the bottom


# ------------------------------------------------------------------------- data

def listdir(d):
    return [p for p in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True))
            if p.lower().endswith(EXTS)]


def load_gray(path, strip_banner):
    if path.lower().endswith(".npy"):
        a = np.load(path).astype(np.float32)
        if a.ndim == 3:
            a = a[..., 0]
    else:
        a = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if a is None:
            return None
        a = a.astype(np.float32) / 255.0
    if strip_banner and a.shape[0] > 64:
        a = a[: int(a.shape[0] * (1 - BANNER_FRAC))]
    return a


def minmax(x):
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class MixtureDataset(Dataset):
    """Weighted sampling across sources, on-the-fly degradation."""

    def __init__(self, sources, patch, scale=2, length=100000,
                 banner_dirs=(), prescale_dirs=()):
        self.files, self.weights, self.tags, self.mmaps = [], [], [], []
        for d, w in sources:
            if os.path.isfile(d) and d.endswith(".npy"):
                mmap = np.load(d, mmap_mode='r')
                self.files.append(mmap)
                self.mmaps.append(True)
                self.weights.append(float(w))
                self.tags.append(d)
                print(f"  {d:<40} {len(mmap):>7} images (mmap) weight {w}")
            else:
                fs = listdir(d)
                if not fs:
                    raise SystemExit(f"no images found in {d}")
                self.files.append(fs)
                self.mmaps.append(False)
                self.weights.append(float(w))
                self.tags.append(d)
                print(f"  {d:<40} {len(fs):>7} images         weight {w}")
        p = np.array(self.weights, dtype=np.float64)
        self.p = p / p.sum()
        self.patch = patch
        self.scale = scale
        self.length = length
        self.banner = {i for i, d in enumerate(self.tags)
                       if any(b in d for b in banner_dirs)}
        self.prescale = {i for i, d in enumerate(self.tags)
                         if any(b in d for b in prescale_dirs)}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.default_rng((idx * 2654435761 + os.getpid()) % (2**31))
        deg = Degrader(downsampler="cv2_cubic", scale=self.scale,
                       sigma_lo=0.055, sigma_hi=0.315,
                       gamma_speckle_prob=1.0, additive_hi=0.0,
                       kernel_jitter_prob=0.15, rng=rng)

        for _ in range(8):
            s = int(rng.choice(len(self.files), p=self.p))
            if self.mmaps[s]:
                f_idx = int(rng.integers(len(self.files[s])))
                img = self.files[s][f_idx].copy()
                if s in self.banner and img.shape[0] > 64:
                    img = img[: int(img.shape[0] * (1 - BANNER_FRAC))]
            else:
                f = self.files[s][int(rng.integers(len(self.files[s])))]
                img = load_gray(f, s in self.banner)
                if img is None:
                    continue

            # Scale matching: 2K photos show a far finer slice of a scene than
            # a 256px SEM field of view. Random pre-downscale covers both, and
            # suppresses camera sensor noise leaking into the "clean" target.
            if s in self.prescale:
                k = float(rng.uniform(1.0, 3.0))
                if k > 1.05:
                    h, w = img.shape
                    img = cv2.resize(img, (max(8, int(w / k)), max(8, int(h / k))),
                                     interpolation=cv2.INTER_AREA)

            P = self.patch * self.scale
            if img.shape[0] < P or img.shape[1] < P:
                continue

            y = int(rng.integers(0, img.shape[0] - P + 1))
            x = int(rng.integers(0, img.shape[1] - P + 1))
            gt = img[y:y + P, x:x + P].copy()

            if rng.random() < 0.5:
                gt = gt[:, ::-1]
            if rng.random() < 0.5:
                gt = gt[::-1, :]
            k = int(rng.integers(4))
            if k:
                gt = np.rot90(gt, k)
            gt = np.ascontiguousarray(gt)

            # gamma jitter BEFORE min-max, so histogram shape varies while the
            # [0,1] convention is preserved
            gt = np.power(np.clip(gt, 0, None) + 1e-6, float(rng.uniform(0.6, 1.7)))
            gt = minmax(gt)

            lr, gt = deg(gt)
            return (torch.from_numpy(lr)[None].float(),
                    torch.from_numpy(gt)[None].float())

        raise RuntimeError(f"no usable crop after 8 tries; check source image sizes "
                           f">= {self.patch * self.scale}px")


def build_val(val_specs, scale, max_n=64):
    """Fixed-seed degraded/clean pairs, identical across every run."""
    out = {}
    for name, d in val_specs:
        fs = listdir(d)[:max_n]
        pairs = []
        for i, f in enumerate(fs):
            img = load_gray(f, "nffa" in d.lower())
            if img is None:
                continue
            P = 256
            if img.shape[0] < P or img.shape[1] < P:
                continue
            y = (img.shape[0] - P) // 2
            x = (img.shape[1] - P) // 2
            gt = minmax(img[y:y + P, x:x + P].copy())
            rng = np.random.default_rng(1234 + i)
            deg = Degrader(downsampler="cv2_cubic", scale=scale,
                           sigma_lo=0.171, sigma_hi=0.171,
                           gamma_speckle_prob=1.0, additive_hi=0.0,
                           kernel_jitter_prob=0.0, rng=rng)
            lr, gt = deg(gt)
            pairs.append((torch.from_numpy(lr)[None], torch.from_numpy(gt)[None]))
        if pairs:
            out[name] = pairs
            print(f"  val[{name}]: {len(pairs)} images from {d}")
    return out


# ------------------------------------------------------------------------ losses

def gaussian_window(ws, sigma, device):
    g = torch.arange(ws, dtype=torch.float32, device=device) - ws // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    return (g.t() @ g).expand(1, 1, ws, ws).contiguous()


def ssim(a, b, win, C1=0.01 ** 2, C2=0.03 ** 2):
    p = win.shape[-1] // 2
    mu_a = F.conv2d(a, win, padding=p)
    mu_b = F.conv2d(b, win, padding=p)
    aa, bb, ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    va = F.conv2d(a * a, win, padding=p) - aa
    vb = F.conv2d(b * b, win, padding=p) - bb
    vab = F.conv2d(a * b, win, padding=p) - ab
    return (((2 * ab + C1) * (2 * vab + C2)) /
            ((aa + bb + C1) * (va + vb + C2))).mean()


def fft_loss(a, b):
    fa = torch.fft.rfft2(a.float(), norm="ortho")
    fb = torch.fft.rfft2(b.float(), norm="ortho")
    return (fa.abs() - fb.abs()).abs().mean()


class Criterion(nn.Module):
    def __init__(self, device, w_l1=1.0, w_ssim=0.2, w_fft=0.05, w_grad=0.05):
        super().__init__()
        self.win = gaussian_window(11, 1.5, device)
        self.w = (w_l1, w_ssim, w_fft, w_grad)

    def forward(self, pred, gt):
        w_l1, w_ss, w_ft, w_gd = self.w
        l1 = torch.sqrt((pred - gt) ** 2 + 1e-6).mean()          # Charbonnier
        ss = 1.0 - ssim(pred.float(), gt.float(), self.win)
        ft = fft_loss(pred, gt)
        gd = ((pred[..., 1:] - pred[..., :-1]) -
              (gt[..., 1:] - gt[..., :-1])).abs().mean() + \
             ((pred[..., 1:, :] - pred[..., :-1, :]) -
              (gt[..., 1:, :] - gt[..., :-1, :])).abs().mean()
        total = w_l1 * l1 + w_ss * ss + w_ft * ft + w_gd * gd
        return total, {"l1": l1.item(), "ssim": ss.item(),
                       "fft": ft.item(), "grad": gd.item()}


# --------------------------------------------------------------------------- ema

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1 - self.decay)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


def save_ckpt(path, step, model, opt, sched, scaler, ema, best):
    torch.save({"step": step, "model": model.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "ema": ema.shadow,
                "best": best}, path)


# -------------------------------------------------------------------- validation

@torch.no_grad()
def validate(model, valsets, device, win):
    model.eval()
    res = {}
    for name, pairs in valsets.items():
        ps, ss_ = [], []
        for lr, gt in pairs:
            lr = lr[None].to(device)
            gt = gt[None].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(lr, clamp=True).float()
            mse = F.mse_loss(out, gt).item()
            ps.append(10 * math.log10(1.0 / max(mse, 1e-12)))
            ss_.append(ssim(out, gt, win).item())
        res[name] = (float(np.mean(ps)), float(np.mean(ss_)))
    model.train()
    return res


# -------------------------------------------------------------------------- main

def parse_pairs(s, sep=":"):
    out = []
    for item in s.split(","):
        item = item.strip()
        if item:
            a, b = item.rsplit(sep, 1)
            out.append((a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True, help='"dir:weight,dir:weight"')
    ap.add_argument("--val", default="", help='"name:dir,name:dir"')
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--banner_dirs", default="nffa")
    ap.add_argument("--prescale_dirs", default="df2k,div2k,flickr")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print("sources:")
    srcs = [(d, float(w)) for d, w in parse_pairs(args.sources)]
    ds = MixtureDataset(srcs, args.patch, length=(args.steps + 1000) * args.bs,
                        banner_dirs=tuple(args.banner_dirs.split(",")),
                        prescale_dirs=tuple(args.prescale_dirs.split(",")))
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)

    valsets = {}
    if args.val:
        print("validation:")
        valsets = build_val(parse_pairs(args.val), scale=2)

    model = build_model({"c": args.channels, "n_blocks": args.blocks}).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    start_step = 0
    best = -1e9
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                            betas=(0.9, 0.9))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    warm = min(2000, args.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm)))
        * 0.999 + 0.001)
    crit = Criterion(device)
    ema = EMA(model, 0.999)
    win = gaussian_window(11, 1.5, device)

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        if "model" in ck:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            for g in opt.param_groups:
                g['lr'] = args.lr
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
            ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
            start_step, best = ck["step"] + 1, ck["best"]
            print(f"resumed at step {start_step}")
        else:
            model.load_state_dict(ck)
            print("loaded weights only (no optimizer state)")
    elif args.resume:
        print(f"no checkpoint at {args.resume}, starting fresh")

    t0 = time.time()
    for i, (lr_img, gt_img) in enumerate(dl):
        step = start_step + i
        if step >= args.steps:
            break
        lr_img = lr_img.to(device, non_blocking=True)
        gt_img = gt_img.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(lr_img)
        loss, parts = crit(pred.float(), gt_img.float())

        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, skipped")
            del pred, loss, parts
            torch.cuda.empty_cache()
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn):
            scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update(model)

        if step % 200 == 0:
            el = time.time() - t0
            print(f"step {step:>6}/{args.steps}  loss {loss.item():.4f}  "
                  f"gnorm {gn:.3f}  scale {scaler.get_scale():.0f}  "
                  f"l1 {parts['l1']:.4f}  lr {sched.get_last_lr()[0]:.2e}  {el:.0f}s")

        if valsets and step % args.val_every == 0:
            # Raw model validation
            res_raw = validate(model, valsets, device, win)
            msg_raw = f"  RAW @ {step:<3}"
            for k, (db, ssim_val) in res_raw.items():
                msg_raw += f"   {k}: {db:.2f}dB/{ssim_val:.4f}"
            print(msg_raw)

            # EMA model validation
            bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.state_dict(), strict=False)
            res = validate(model, valsets, device, win)
            model.load_state_dict(bak)

            msg = f"  EMA @ {step:<3}"
            for k, (db, ssim_val) in res.items():
                msg += f"   {k}: {db:.2f}dB/{ssim_val:.4f}"
            print(msg)
            score = float(np.mean([p for p, _ in res.values()]))
            if score > best:
                best = score
                save_ckpt(os.path.join(args.out, "best.pt"), step, model, opt,
                          sched, scaler, ema, best)
                torch.save(ema.state_dict(), os.path.join(args.out, "best_ema.pt"))
                print(f"  saved best_ema.pt  (mean {score:.3f} dB)")

        if step > 0 and step % args.save_every == 0:
            save_ckpt(os.path.join(args.out, "last.pt"), step, model, opt,
                      sched, scaler, ema, best)

    torch.save(ema.state_dict(), os.path.join(args.out, "final_ema.pt"))
    torch.save(model.state_dict(), os.path.join(args.out, "final_raw.pt"))
    print(f"done in {(time.time()-t0)/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
