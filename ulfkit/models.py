"""Backbones: a residual U-Net, the LoHiResGAN generator, and a 70x70 PatchGAN."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(c):
    return nn.GroupNorm(min(32, c // 4 if c >= 4 else 1), c)


def timestep_embedding(t, dim):
    """Sinusoidal embedding of a continuous t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None] * 1000.0
    return torch.cat([torch.cos(a), torch.sin(a)], -1)


class ResBlock(nn.Module):
    """Optional FiLM conditioning on a time embedding; without t_dim the block is
    parameter-identical to the original, so existing checkpoints still load."""

    def __init__(self, cin, cout, t_dim=None):
        super().__init__()
        self.n1, self.c1 = gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.emb = nn.Linear(t_dim, cout * 2) if t_dim else None

    def forward(self, x, temb=None):
        h = self.c1(F.silu(self.n1(x)))
        if self.emb is not None and temb is not None:
            scale, shift = self.emb(F.silu(temb)).chunk(2, 1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Residual U-Net. Predicts a residual on top of `anchor` input channels."""

    def __init__(self, in_ch=3, out_ch=1, base=64, mults=(1, 2, 4, 8), nres=2,
                 residual_from=None, t_dim=None):
        super().__init__()
        self.t_dim = t_dim
        if t_dim:
            self.tmlp = nn.Sequential(nn.Linear(t_dim, t_dim * 4), nn.SiLU(),
                                      nn.Linear(t_dim * 4, t_dim))
        # Channel(s) of the input the network adds its output to. Either a
        # (start, stop) range or an explicit list of channel indices.
        self.residual_from = residual_from
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        chs = [base * m for m in mults]
        self.downs = nn.ModuleList()
        c = base
        skips = []
        for ch in chs:
            blocks = nn.ModuleList([ResBlock(c, ch, t_dim)] + [ResBlock(ch, ch, t_dim) for _ in range(nres - 1)])
            self.downs.append(nn.ModuleList([blocks, nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
            skips.append(ch)
            c = ch
        self.mid = nn.ModuleList([ResBlock(c, c, t_dim), ResBlock(c, c, t_dim)])
        self.ups = nn.ModuleList()
        for ch in reversed(chs):
            blocks = nn.ModuleList([ResBlock(c + ch, ch, t_dim)] + [ResBlock(ch, ch, t_dim) for _ in range(nres - 1)])
            self.ups.append(nn.ModuleList([nn.ConvTranspose2d(c, c, 4, stride=2, padding=1), blocks]))
            c = ch
        self.head = nn.Sequential(gn(c), nn.SiLU(), nn.Conv2d(c, out_ch, 3, padding=1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, t=None):
        anchor = None
        if self.residual_from is not None:
            rf = self.residual_from
            anchor = x[:, rf[0]:rf[1]] if isinstance(rf, tuple) else x[:, list(rf)]
        temb = self.tmlp(timestep_embedding(t, self.t_dim)) if (self.t_dim and t is not None) else None
        h = self.stem(x)
        skips = []
        for blocks, down in self.downs:
            for b in blocks:
                h = b(h, temb)
            skips.append(h)
            h = down(h)
        for b in self.mid:
            h = b(h, temb)
        for up, blocks in self.ups:
            h = up(h)
            s = skips.pop()
            if h.shape[-1] != s.shape[-1]:
                h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = torch.cat([h, s], 1)
            for b in blocks:
                h = b(h, temb)
        out = self.head(h)
        return out if anchor is None else out + anchor


class _LRBlock(nn.Module):
    """conv/deconv + BN + ReLU + residual refinement, as in LoHiResGAN."""

    def __init__(self, cin, cout, down=True, drop=0.0):
        super().__init__()
        self.main = (nn.Conv2d(cin, cout, 4, 2, 1, bias=False) if down
                     else nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False))
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.LeakyReLU(0.2, True) if down else nn.ReLU(True)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.res = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
                                 nn.BatchNorm2d(cout), nn.ReLU(True),
                                 nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
                                 nn.BatchNorm2d(cout))

    def forward(self, x):
        h = self.drop(self.act(self.bn(self.main(x))))
        return F.relu(h + self.res(h))


class LoHiResGen(nn.Module):
    """Reimplementation of the LoHiResGAN generator (Islam et al., Sci Rep 2023).

    8 residual downsampling layers (64..512), 7 residual upsampling layers with
    dropout 0.5 on the first three, U-Net skips, tanh output.
    """

    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        dch = [64, 128, 256, 512, 512, 512, 512, 512]
        uch = [512, 512, 512, 512, 256, 128, 64]
        self.downs = nn.ModuleList()
        c = in_ch
        for ch in dch:
            self.downs.append(_LRBlock(c, ch, down=True)); c = ch
        self.ups = nn.ModuleList()
        for i, ch in enumerate(uch):
            self.ups.append(_LRBlock(c if i == 0 else c + dch[len(dch) - 1 - i], ch,
                                     down=False, drop=0.5 if i < 3 else 0.0))
            c = ch
        self.final = nn.ConvTranspose2d(c + dch[0], out_ch, 4, 2, 1)

    def forward(self, x):
        skips = []
        h = x
        for d in self.downs:
            h = d(h); skips.append(h)
        skips = skips[:-1][::-1]
        for i, u in enumerate(self.ups):
            h = u(h if i == 0 else torch.cat([h, skips[i - 1]], 1))
        h = self.final(torch.cat([h, skips[-1]], 1))
        return (torch.tanh(h) + 1) / 2      # pipeline is [0,1], not [-1,1]


class PatchGAN(nn.Module):
    """70x70 PatchGAN discriminator."""

    def __init__(self, in_ch=4, base=64):
        super().__init__()
        L = [nn.Conv2d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        c = base
        for m in (2, 4):
            L += [nn.Conv2d(c, base * m, 4, 2, 1, bias=False),
                  nn.InstanceNorm2d(base * m, affine=True), nn.LeakyReLU(0.2, True)]
            c = base * m
        L += [nn.Conv2d(c, c * 2, 4, 1, 1, bias=False),
              nn.InstanceNorm2d(c * 2, affine=True), nn.LeakyReLU(0.2, True),
              nn.Conv2d(c * 2, 1, 4, 1, 1)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)
