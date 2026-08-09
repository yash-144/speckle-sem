"""
lpips_metric.py -- LPIPS for KLA PS01.

HARD RULE: evaluate.py must NEVER import this module. LPIPS is a
training/validation-only dependency. Keeping it out of the inference path is
what stops a missing `lpips` wheel from turning the submission gate into an
ImportError on KLA's H100.

Two entry points:
    lpips_score(pred, gt)   -> float. No grad. For validation. Mirrors what
                               evaluate.py actually writes to disk (clamped).
    LPIPSLoss()             -> nn.Module. Differentiable. For the training loss.

Design decisions that matter for this project:

  * The backbone is cached in a MODULE-LEVEL DICT, not registered as a
    submodule. If it were a child module it would land in model.state_dict()
    and inflate weights/best_ema.pt from ~2.3 MB to ~10 MB (alex) or ~60 MB
    (vgg16 features). The "commit weights directly, no Git LFS" plan depends
    on this staying small.

  * Everything runs in fp32 with autocast explicitly disabled. A fp16
    backbone forward is an inf source, and this codebase has already lost a
    day to a GradScaler/inf/clip_grad_norm_ deadlock. Not re-litigating that.

  * Grayscale is replicated to 3 channels. LPIPS backbones are ImageNet nets;
    there is no single-channel variant. Replication is what every SR paper
    does for grayscale LPIPS.

  * clamp defaults differ by entry point on purpose:
      metric -> clamp=True  (match evaluate.py's on-disk output)
      loss   -> clamp=False (clamping kills gradients on saturated pixels)

Install:  pip install lpips        # put this in requirements-train.txt ONLY
First call downloads torchvision ImageNet weights (~233 MB for alex,
~528 MB for vgg16). On Kaggle, set TORCH_HOME to a path under /kaggle/working
and run with internet enabled at least once.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Backbone cache. Plain dict -> never enters any state_dict.
_BACKBONES: dict = {}


def _get_backbone(net: str, device: torch.device):
    """Lazily build and cache a frozen LPIPS backbone."""
    key = (net, str(device))
    if key not in _BACKBONES:
        import lpips  # local import keeps this file importable without the dep

        m = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _BACKBONES[key] = m
    return _BACKBONES[key]


def _prep(x: torch.Tensor, clamp: bool) -> torch.Tensor:
    """(N,H,W) or (N,1,H,W) or (N,3,H,W) in nominal [0,1] -> (N,3,H,W) in [-1,1]."""
    if x.dim() == 2:
        x = x[None, None]
    elif x.dim() == 3:
        x = x[:, None]
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    x = x.float()
    if clamp:
        x = x.clamp(0.0, 1.0)
    return x * 2.0 - 1.0


def _call(net: str, pred: torch.Tensor, target: torch.Tensor, clamp: bool):
    m = _get_backbone(net, pred.device)
    # Force fp32. autocast(enabled=False) is a no-op if we are not inside one.
    with torch.autocast(device_type=pred.device.type, enabled=False):
        d = m(_prep(pred, clamp), _prep(target, clamp))
    return d.flatten()


@torch.no_grad()
def lpips_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "alex",
    clamp: bool = True,
) -> float:
    """Validation LPIPS. Lower is better. Returns a Python float.

    clamp=True by default so this measures the same pixels evaluate.py writes.
    """
    return _call(net, pred, target, clamp).mean().item()


class LPIPSLoss(nn.Module):
    """Differentiable LPIPS term for the training loss.

    Not a parent of the backbone -- see module docstring. Safe to hold as an
    attribute on your trainer; it contributes nothing to any state_dict.
    """

    def __init__(self, net: str = "alex", clamp: bool = False):
        super().__init__()
        self.net = net
        self.clamp = clamp

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return _call(self.net, pred, target, self.clamp).mean()

    def extra_repr(self) -> str:
        return f"net={self.net}, clamp={self.clamp}"


# --------------------------------------------------------------------------
# Smoke test:  python lpips_metric.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device="cpu").manual_seed(0)

    gt = torch.rand(2, 1, 256, 256, generator=g).to(dev)
    noisy = (gt + 0.05 * torch.randn(gt.shape, generator=g).to(dev))

    print("identical  :", lpips_score(gt, gt))            # expect ~0.0
    print("noisy      :", lpips_score(noisy, gt))         # expect > 0
    print("out-of-range input survives:",
          lpips_score(gt * 1.36, gt))                     # 1.36 is legal here

    # The load-bearing assertion: the backbone must not leak into a checkpoint.
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3)
            self.percep = LPIPSLoss()

    d = Dummy().to(dev)
    _ = d.percep(gt, gt)  # force the backbone to build
    keys = list(d.state_dict().keys())
    assert not any("percep" in k for k in keys), keys
    print("state_dict stays clean:", keys)
