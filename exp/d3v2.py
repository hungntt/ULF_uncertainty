"""Direction 3-v2: posterior sampling, evidence attribution, and calibrated trust maps.

Backbone follows Yeow et al. (arXiv 2609.25319): a residual-shift diffusion in the
Haar wavelet domain, started from the 64 mT input rather than from noise, so a
posterior draw costs ~16 network evaluations. Two departures, both forced by what
Direction 1 and Finding A established on this cohort:

  * trained on the *real* paired subjects, not domain-randomised synthetic pairs -
    synthetic-only pretraining with a published-style physics operator transfers
    almost no direction here (zero-shot LL cosine -0.036);
  * evaluated against voxel-level paired ground truth, which this cohort has and
    theirs does not, so the uncertainty map can actually be validated rather than
    only displayed.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
_DWT = _IWT = None


def wave():
    global _DWT, _IWT
    if _DWT is None:
        from pytorch_wavelets import DWTForward, DWTInverse
        _DWT = DWTForward(J=1, wave="haar", mode="zero").cuda()
        _IWT = DWTInverse(wave="haar", mode="zero").cuda()
    return _DWT, _IWT


def to_wav(x):
    """(B,C,256,256) -> (B,4C,128,128) as [LL,LH,HL,HH] per input channel."""
    dwt, _ = wave()
    B, C = x.shape[:2]
    yl, yh = dwt(x.reshape(B * C, 1, *x.shape[-2:]))
    w = torch.cat([yl, yh[0][:, 0]], 1)                 # (B*C,4,128,128)
    return w.reshape(B, C * 4, *w.shape[-2:])


def from_wav(w):
    """(B,4,128,128) -> (B,1,256,256)."""
    _, iwt = wave()
    return iwt((w[:, :1], [w[:, 1:4].unsqueeze(1)]))


class Schedule:
    """Residual-shift marginal q(x_t | x_HR, x_LR) = N(x_HR + (x_LR-x_HR) b_t, g^2 b_t I)."""

    def __init__(self, T=1000, b_min=4e-4, b_max=0.9999, p=0.3, gamma=1.0):
        i = torch.linspace(0, 1, T)
        s = i ** p                                       # front-loaded warp
        self.b = (b_min * (b_max / b_min) ** s).cuda()
        self.T, self.gamma = T, gamma

    def q(self, x_hr, x_lr, idx, noise=None):
        b = self.b[idx].view(-1, 1, 1, 1)
        z = torch.randn_like(x_hr) if noise is None else noise
        return x_hr + (x_lr - x_hr) * b + self.gamma * b.sqrt() * z

    def step(self, x0, x_lr, idx):
        b = self.b[idx].view(-1, 1, 1, 1)
        return x0 + (x_lr - x0) * b + self.gamma * b.sqrt() * torch.randn_like(x0)


def subband_std(cohort, rows, bs=64):
    """Per-subband std of the 3T target, for unit-variance weighting."""
    acc, n = torch.zeros(4).cuda(), 0
    for i in range(0, len(rows), bs):
        y = torch.from_numpy(cohort.hf[rows[i:i + bs]].astype(np.float32)).cuda()
        w = to_wav(y)
        acc += w.pow(2).mean((0, 2, 3)) * len(y); n += len(y)
    return (acc / n).sqrt().clamp(min=1e-6)


def build(contrast, base=64):
    # 4 noisy subbands + 12 conditioning subbands (3-slice 64 mT slab)
    return Mo.UNet(16, 4, base=base, t_dim=128).cuda()


def train(contrast, steps=14000, bs=16, base=64, gamma=1.0, lr=1e-4,
          norm=False, cond_drop=0.0, tag=""):
    """norm: run the diffusion in unit-variance subband space, so a single gamma is
    comparable across LL..HH (their std differs ~15x here). cond_drop: fraction of
    steps with the 64 mT conditioning zeroed, which stops the network relying on the
    conditioning alone and collapsing the posterior."""
    T.seed_all(0)
    coh, tr, val = D.dev_split(contrast)
    rows = coh.rows_for(tr)
    sch = Schedule(gamma=gamma)
    net = build(contrast, base)
    sd = subband_std(coh, rows)
    print(f"  subband std (LL,LH,HL,HH): {[f'{v:.4f}' for v in sd.tolist()]}")
    w_sb = (1.0 / sd).view(1, 4, 1, 1)
    ns = w_sb if norm else torch.ones_like(w_sb)      # scale into the diffusion space

    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    sch_lr = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    for it in range(steps):
        r = np.sort(rng.choice(len(rows), bs, replace=False))
        j = rows[r]
        x = torch.from_numpy(coh.ulf[j].astype(np.float32)).cuda()
        y = torch.from_numpy(coh.hf[j].astype(np.float32)).cuda()
        if rng.random() < 0.5: x, y = torch.flip(x, [-1]), torch.flip(y, [-1])
        w_hr, w_cond = to_wav(y) * ns, to_wav(x) * ns.repeat(1, 3, 1, 1)
        w_lr = w_cond[:, 4:8]                            # centre slice = the observation
        idx = torch.randint(0, sch.T, (bs,), device="cuda")
        xt = sch.q(w_hr, w_lr, idx)
        cond = w_cond
        if cond_drop and rng.random() < cond_drop:
            cond = torch.zeros_like(w_cond)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x0 = net(torch.cat([xt, cond], 1), idx.float() / sch.T)
        # the wavelet filters are fp32, so the inverse transform stays outside autocast
        x0 = x0.float()
        l_wav = ((x0 - w_hr) * (w_sb if not norm else torch.ones_like(w_sb))).abs().mean()
        l_img = (from_wav(x0 / ns) - y).abs().mean()
        loss = l_wav + l_img
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sch_lr.step()
        if (it + 1) % 2000 == 0:
            print(f"  step {it+1:6d}  loss {loss.item():.5f}  (wav {l_wav.item():.5f} img {l_img.item():.5f})")
    torch.save({"sd": net.state_dict(), "gamma": gamma, "base": base,
                "norm": norm, "sb_scale": ns.detach().cpu()},
               f"{OUT}/d3v2_diff_{contrast}{tag}.pt")
    return net, sch


@torch.no_grad()
def sample(net, sch, ulf_slab, K=32, n_steps=16, seed=0, ns=None):
    """-> (K,B,1,256,256) image-space posterior draws."""
    g = torch.Generator(device="cuda"); g.manual_seed(seed)
    ns = torch.ones(1, 4, 1, 1, device="cuda") if ns is None else ns.cuda()
    w_cond = to_wav(ulf_slab) * ns.repeat(1, 3, 1, 1)
    w_lr = w_cond[:, 4:8]
    idxs = torch.linspace(sch.T - 1, 0, n_steps).long().cuda()
    draws = []
    for k in range(K):
        x = w_lr + sch.gamma * sch.b[idxs[0]].sqrt() * torch.randn(w_lr.shape, device="cuda", generator=g)
        for s in range(n_steps):
            t = idxs[s].repeat(len(x))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x0 = net(torch.cat([x, w_cond], 1), t.float() / sch.T)
            x0 = x0.float()
            if s < n_steps - 1:
                b = sch.b[idxs[s + 1]]
                x = x0 + (w_lr - x0) * b + sch.gamma * b.sqrt() * torch.randn(
                    x0.shape, device="cuda", generator=g)
            else:
                x = x0
        draws.append(from_wav(x / ns).clamp(0, 1))
    return torch.stack(draws)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--steps", type=int, default=14000)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--norm", action="store_true")
    ap.add_argument("--cond_drop", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    train(a.contrast, a.steps, gamma=a.gamma, base=a.base, norm=a.norm,
          cond_drop=a.cond_drop, tag=a.tag)
