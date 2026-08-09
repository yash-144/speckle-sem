import torch, math, torch.nn.functional as F
from train import build_val, parse_pairs, gaussian_window
from model import build_model

dev = torch.device('cpu')
vs = build_val(parse_pairs("kla:val/kla_gt"), scale=2)
m = build_model().eval()
win = gaussian_window(11, 1.5, dev)
for name, pairs in vs.items():
    b, n = [], []
    for lr, gt in pairs:
        bi = F.interpolate(lr[None], scale_factor=2, mode='bicubic',
                           align_corners=False).clamp(0,1)
        net = m(lr[None], clamp=True)
        b.append(10*math.log10(1/max(F.mse_loss(bi, gt[None]).item(),1e-12)))
        n.append(10*math.log10(1/max(F.mse_loss(net, gt[None]).item(),1e-12)))
    print(f"{name}: bicubic {sum(b)/len(b):.2f} dB   untrained net {sum(n)/len(n):.2f} dB")
