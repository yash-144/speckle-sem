"""
model.py

Joint denoise + x2 super-resolution network for KLA PS01.

Design notes, all driven by the forensics:
  * Plain residual stack, NOT a U-Net. There is no downsampling anywhere, so
    the network accepts any input size with no padding logic. That keeps the
    evaluation script trivial and robust, which matters more than the ~0.1 dB
    a U-Net might buy.
  * Global bicubic residual. The net only predicts the correction on top of a
    bicubic x2 upsample, so intensity calibration is free and training
    converges in a few thousand steps instead of tens of thousands.
  * Input is fed RAW. The data is already per-image min-max normalised to
    [0,1] by KLA and the degraded input legitimately reaches ~1.36.
    Re-normalising by the input's own max would destroy that calibration.
  * Output is clamped to [0,1] at inference only. Ground truth provably
    cannot leave that range, so clamping is free PSNR. It is NOT applied
    during training, where it would kill gradients on saturated pixels.

Default config: C=64, N=16 -> ~0.5M params, single 3x3 depthwise per block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    """NAFNet block (Chen et al., ECCV 2022): no activation functions,
    just multiplicative gating and channel attention."""

    def __init__(self, c, dw_expand=2, ffn_expand=2, ksize=5):
        super().__init__()
        d = c * dw_expand
        self.conv1 = nn.Conv2d(c, d, 1)
        self.conv2 = nn.Conv2d(d, d, ksize, padding=ksize // 2, groups=d)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(d // 2, d // 2, 1))
        self.conv3 = nn.Conv2d(d // 2, c, 1)

        f = c * ffn_expand
        self.conv4 = nn.Conv2d(c, f, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(f // 2, c, 1)

        self.norm1 = nn.GroupNorm(1, c)
        self.norm2 = nn.GroupNorm(1, c)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv2(self.conv1(y))
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.sg2(self.conv4(y))
        y = self.conv5(y)
        return x + y * self.gamma


class RestoreNet(nn.Module):
    def __init__(self, c=64, n_blocks=16, scale=2, in_ch=1):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(in_ch, c, 3, padding=1)
        self.body = nn.Sequential(*[NAFBlock(c) for _ in range(n_blocks)])
        self.fuse = nn.Conv2d(c, c, 3, padding=1)
        self.tail = nn.Conv2d(c, in_ch * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x, clamp=False):
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic",
                             align_corners=False)
        f = self.head(x)
        f = self.fuse(self.body(f)) + f
        out = self.shuffle(self.tail(f)) + base
        return out.clamp(0.0, 1.0) if clamp else out


def build_model(cfg=None):
    cfg = cfg or {}
    return RestoreNet(c=cfg.get("c", 64),
                      n_blocks=cfg.get("n_blocks", 16),
                      scale=cfg.get("scale", 2))


if __name__ == "__main__":
    m = build_model()
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(2, 1, 128, 128).abs()
    y = m(x)
    print(f"params: {n/1e6:.3f}M    {tuple(x.shape)} -> {tuple(y.shape)}")
    x2 = torch.randn(1, 1, 256, 256).abs()
    print(f"                      {tuple(x2.shape)} -> {tuple(m(x2).shape)}")
