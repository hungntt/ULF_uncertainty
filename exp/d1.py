"""Direction 1 — closing the reference gap.

The auxiliary 3T corpus everyone uses for unpaired/pretraining supervision (IXI
via FOMO-50k here) is spectrally *softer* than the clinical 3T target: 4.5-4.7x
less power above r=64, which is 3.8-4.9 SD below the target cohort's own
between-subject spread. Pretraining on it therefore teaches the enhancer to stop
short of the real target.

Fix, in three parts:
  1. learn the ULF forward operator F: 3T -> 64 mT from the small paired set
     (information-destroying, hence well-posed) instead of hand-crafting it;
  2. spectrally calibrate the corpus onto the target's radial spectrum before
     degrading it;
  3. pretrain on the resulting pseudo-pairs, then finetune on the real pairs.
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
CACHE = "/mnt/G/Projects/ULF-MICCAI/datasets/npy"


# ---------------------------------------------------------------- hand-crafted D
def _smooth_field(n, h, w, scale, gen, device="cuda"):
    """Low-frequency random field in [0,1], for coil shading and B0."""
    f = torch.randn(n, 1, scale, scale, device=device, generator=gen)
    f = F.interpolate(f, size=(h, w), mode="bicubic", align_corners=False)
    f = (f - f.amin((2, 3), True)) / (f.amax((2, 3), True) - f.amin((2, 3), True) + 1e-6)
    return f


def handcrafted_D(y, blur=1.5, snr_db=18.0, gen=None, physics=True, frac=None):
    """Hand-designed 64 mT forward model, i.e. what a simulation-only pipeline uses.

    Follows the operations published ULF simulators apply (ellipsoidal coil
    shading, B0-driven T2* decay, k-space bandwidth truncation, Rician noise)
    rather than the blur+noise proxy used in the predecessor project. Set
    physics=False for that simpler proxy."""
    n, c, h, w = y.shape
    dev = y.device
    supp = (y > 0).float()

    if not physics:
        k = int(2 * round(3 * blur) + 1)
        g = torch.arange(k, device=dev, dtype=torch.float32) - k // 2
        g = torch.exp(-g ** 2 / (2 * blur ** 2)); g = g / g.sum()
        x = F.conv2d(F.pad(y, (k // 2, k // 2, 0, 0), mode="reflect"),
                     g.view(1, 1, 1, k).expand(c, 1, 1, k), groups=c)
        x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"),
                     g.view(1, 1, k, 1).expand(c, 1, k, 1), groups=c)
    else:
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=dev),
                                torch.linspace(-1, 1, w, device=dev), indexing="ij")
        r2 = (yy ** 2 + xx ** 2)[None, None]
        # ellipsoidal coil sensitivity, >=30% at the edge of the FOV
        sens = 1.0 - 0.7 * r2.clamp(max=1.0)
        # B0 inhomogeneity drives a spatially varying T2* attenuation
        b0 = _smooth_field(n, h, w, 8, gen, dev)
        att = torch.exp(-(0.15 + 0.35 * b0))
        x = y * sens * att
        # bandwidth limitation / lower acquisition matrix: truncate k-space
        if frac is None:
            frac = 0.42 + 0.18 * torch.rand((), device=dev, generator=gen).item()
        X = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        ky = (torch.arange(h, device=dev) - h / 2).abs()[:, None] / (h / 2)
        kx = (torch.arange(w, device=dev) - w / 2).abs()[None] / (w / 2)
        X = X * (torch.sqrt((ky / frac) ** 2 + (kx / frac) ** 2) <= 1).float()
        x = torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1))).real
        x = x / x[supp > 0].mean().clamp(min=1e-6) * y[supp > 0].mean()

    sig = x[supp > 0].mean() if (supp > 0).any() else x.mean()
    s = sig / (10 ** (snr_db / 20))
    n1 = torch.randn(x.shape, device=dev, generator=gen) * s
    n2 = torch.randn(x.shape, device=dev, generator=gen) * s
    return torch.sqrt((x + n1) ** 2 + n2 ** 2).clamp(0, 1) * supp


# ---------------------------------------------------------------- spectral tools
def radial_bins(H=256, W=256, nb=128, device="cuda"):
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    return torch.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2).long().clamp(max=nb - 1)


def mean_radial(x, r, nb=128, bs=256):
    """x: (N,1,H,W) cpu f16 -> mean radial power (nb,) on gpu."""
    acc = torch.zeros(nb, device="cuda"); cnt = torch.bincount(r.flatten(), minlength=nb).clamp(min=1)
    n = 0
    for i in range(0, len(x), bs):
        a = torch.from_numpy(x[i:i + bs, 0].astype(np.float32)).cuda()
        P = torch.fft.fftshift(torch.fft.fft2(a), dim=(-2, -1)).abs() ** 2
        acc += torch.bincount(r.flatten(), weights=P.sum(0).flatten(), minlength=nb)
        n += len(a)
    return acc / cnt / n


def spectral_calibrate(x, gain, r, bs=256, cap=4.0):
    """Apply a radial gain g(r) to every slice, preserving the skull-strip support."""
    g = gain.clamp(max=cap)
    out = np.empty_like(x)
    for i in range(0, len(x), bs):
        a = torch.from_numpy(x[i:i + bs, 0].astype(np.float32)).cuda()
        supp = (a > 0).float()
        Xf = torch.fft.fftshift(torch.fft.fft2(a), dim=(-2, -1))
        Xf = Xf * g[r][None]
        b = torch.fft.ifft2(torch.fft.ifftshift(Xf, dim=(-2, -1))).real
        b = (b * supp).clamp(0, 1)
        out[i:i + bs, 0] = b.cpu().numpy().astype(np.float16)
    return out


# ---------------------------------------------------------------- stages
def stage_forward(contrast, steps, bs=16, n_subj=None, tag=""):
    """Train F: 3T centre slice -> 64 mT centre slice on the paired training subjects.

    n_subj restricts the operator to the same nested subset the enhancer will see,
    so a low-paired-data arm does not inherit an operator fitted on more subjects."""
    T.seed_all(0)
    coh, tr, val = D.dev_split(contrast)
    if n_subj is not None:
        tr = sorted(tr)[:n_subj]
        print(f"  forward operator fitted on {len(tr)} paired subjects")
    net = Mo.UNet(1, 1, base=32, residual_from=(0, 1)).cuda()
    rows = coh.rows_for(tr)
    xs, ys = coh.hf, coh.ulf[:, 1:2]           # 3T in, ULF out
    smp = T.Sampler(xs, ys, rows, bs=bs, seed=1)

    def val_fn(model):
        model.eval(); rs = []
        with torch.no_grad():
            for sid in val:
                idx = coh.subject_rows(sid)
                p, u = [], []
                for i in range(0, len(idx), 32):
                    j = idx[i:i + 32]
                    a = torch.from_numpy(coh.hf[j].astype(np.float32)).cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p.append(net(a).float().clamp(0, 1))
                    u.append(torch.from_numpy(coh.ulf[j][:, 1:2].astype(np.float32)).cuda())
                p, u = torch.cat(p), torch.cat(u)
                rs.append(M.volume_metrics(p, u, u))
        model.train()
        g = M.aggregate(rs)
        return {"score": g["ssim"][0], "ssim": g["ssim"][0], "psnr": g["psnr"][0]}

    info = T.train_regression(net, smp, steps, lr=2e-4, log=print, val_fn=val_fn, val_every=1000)
    torch.save(net.state_dict(), f"{OUT}/d1_forward_{contrast}{tag}.pt")

    # residual spectrum of real ULF around F(3T): used to colour the synthetic noise
    rb = radial_bins()
    net.eval(); acc = torch.zeros(128, device="cuda"); n = 0
    cnt = torch.bincount(rb.flatten(), minlength=128).clamp(min=1)
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            j = rows[i:i + 64]
            a = torch.from_numpy(coh.hf[j].astype(np.float32)).cuda()
            u = torch.from_numpy(coh.ulf[j][:, 1:2].astype(np.float32)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                res = (u - net(a).float().clamp(0, 1))
            P = torch.fft.fftshift(torch.fft.fft2(res[:, 0]), dim=(-2, -1)).abs() ** 2
            acc += torch.bincount(rb.flatten(), weights=P.sum(0).flatten(), minlength=128); n += len(j)
    np.save(f"{OUT}/d1_resid_spec_{contrast}{tag}.npy", (acc / cnt / n).cpu().numpy())
    print("forward operator done:", info["best"])
    return info


def load_forward(contrast, tag=""):
    net = Mo.UNet(1, 1, base=32, residual_from=(0, 1)).cuda()
    net.load_state_dict(torch.load(f"{OUT}/d1_forward_{contrast}{tag}.pt")); net.eval()
    return net


def ixi_subject_bounds(x, thresh=0.95):
    """FOMO-50k ships as one stack with no manifest; split it where consecutive
    slices stop correlating, so 2.5D slabs never straddle two subjects.

    The cut is unambiguous: consecutive within-subject slices correlate >0.98,
    across a boundary <0.90, and the result is 185 runs of 57-69 slices each,
    matching the central z-window used for the paired cohort."""
    a = torch.from_numpy(x[:, 0].astype(np.float32)).cuda()
    a = F.avg_pool2d(a[:, None], 4)[:, 0].flatten(1)
    a = (a - a.mean(1, keepdim=True)) / (a.std(1, keepdim=True) + 1e-6)
    c = (a[:-1] * a[1:]).mean(1)
    b = torch.cat([torch.tensor([0], device="cuda"), (c < thresh).nonzero()[:, 0] + 1,
                   torch.tensor([len(x)], device="cuda")]).unique().cpu().numpy()
    return b


def stage_pairs(contrast, mode, cap=4.0, tag="", n_subj=None):
    """Build pseudo-pairs. mode: handcrafted | learned | learned_cal."""
    T.seed_all(0)
    coh, tr, val = D.dev_split(contrast)
    if n_subj is not None:
        tr = sorted(tr)[:n_subj]          # calibrate the target spectrum on the same subset
    ixi = D.unpaired_3t(contrast)
    rb = radial_bins()

    if mode == "learned_cal":
        tgt = mean_radial(coh.hf[coh.rows_for(tr)], rb)
        src = mean_radial(ixi, rb)
        gain = torch.sqrt(tgt / src.clamp(min=1e-20))
        gain[:2] = 1.0
        print("  radial gain r=[8,32,64,96,120]:", [f"{gain[b].clamp(max=cap).item():.2f}" for b in (8, 32, 64, 96, 120)])
        ixi = spectral_calibrate(ixi, gain, rb, cap=cap)
        np.save(f"{OUT}/d1_gain_{contrast}.npy", gain.cpu().numpy())

    bounds = ixi_subject_bounds(ixi)
    print(f"  corpus split into {len(bounds)-1} contiguous runs")
    seg = np.zeros(len(ixi), dtype=np.int64)
    for k in range(len(bounds) - 1):
        seg[bounds[k]:bounds[k + 1]] = k

    net = load_forward(contrast, tag) if mode.startswith("learned") else None
    rs = torch.from_numpy(np.load(f"{OUT}/d1_resid_spec_{contrast}{tag}.npy")).cuda() if net else None

    ulf = np.empty((len(ixi), 3, 256, 256), dtype=np.float16)
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    idx = np.arange(len(ixi))
    nb = np.stack([np.clip(idx - 1, 0, None), idx, np.clip(idx + 1, None, len(ixi) - 1)], 1)
    for col in (0, 2):                       # clamp a neighbour that fell into another run
        bad = seg[nb[:, col]] != seg
        nb[bad, col] = idx[bad]

    with torch.no_grad():
        for i in range(0, len(ixi), 64):
            j = slice(i, min(i + 64, len(ixi)))
            trip = nb[j]                                            # (b,3)
            a = torch.from_numpy(ixi[trip.reshape(-1), 0].astype(np.float32)).cuda()[:, None]
            if net is None:
                d = handcrafted_D(a, gen=gen)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    d = net(a).float().clamp(0, 1)
                # colour the stochastic part with the measured ULF residual spectrum
                w = torch.randn(d.shape, device="cuda", generator=gen)
                Wf = torch.fft.fftshift(torch.fft.fft2(w[:, 0]), dim=(-2, -1))
                Wf = Wf * torch.sqrt(rs.clamp(min=0))[rb][None] / (256.0)
                nz = torch.fft.ifft2(torch.fft.ifftshift(Wf, dim=(-2, -1))).real[:, None]
                d = (d + nz).clamp(0, 1) * (a > 0)
            ulf[j] = d.reshape(-1, 3, 256, 256).cpu().numpy().astype(np.float16)
    np.save(f"{CACHE}/d1_pseudo_ulf_{contrast}_{mode}{tag}.npy", ulf)
    if mode == "learned_cal":
        np.save(f"{CACHE}/d1_pseudo_hf_{contrast}{tag}.npy", ixi)
    print(f"  wrote pseudo-pairs [{mode}] {ulf.shape}")


def stage_train(contrast, mode, pre_steps, ft_steps, bs=16, base=32, tag=None,
                n_subj=None, ptag="", seed=0):
    """Pretrain on pseudo-pairs (if any), then finetune on the real paired set."""
    # seed reaches the global RNG and both samplers; seed=0 reproduces the original run
    T.seed_all(seed)
    coh, tr, val = D.dev_split(contrast)
    if n_subj is not None:                 # nested subsets, so the curve is monotone in data
        tr = sorted(tr)[:n_subj]
        print(f"  restricted to {len(tr)} paired training subjects")
    test = D.PairedCohort("test", contrast)
    net = Mo.UNet(3, 1, base=base, residual_from=(1, 2)).cuda()

    def in_fn_of(c):
        return lambda rows: (c.ulf[rows].astype(np.float32), c.ulf[rows][:, 1:2].astype(np.float32),
                             c.hf[rows].astype(np.float32))

    def val_fn(model):
        g = M.aggregate(T.evaluate(model, coh, val, in_fn_of(coh)))
        return {"score": g["ssim"][0], "ssim": g["ssim"][0], "psnr": g["psnr"][0],
                "cosLL": g["cos_LL"][0], "rHH": g["r_HH"][0]}

    if mode == "none":
        # compute-matched control: the pretrained arms see pre_steps + ft_steps
        # optimiser steps in total, so the no-corpus arm gets the same budget on
        # the real pairs rather than being starved of training.
        ft_steps = ft_steps + pre_steps
        print(f"  no corpus: compute-matched to {ft_steps} steps on real pairs")

    pre_info = None
    if mode != "none":
        pu = np.load(f"{CACHE}/d1_pseudo_ulf_{contrast}_{mode}{ptag}.npy")
        ph = (np.load(f"{CACHE}/d1_pseudo_hf_{contrast}{ptag}.npy") if mode == "learned_cal"
              else D.unpaired_3t(contrast))
        print(f"  pretrain on {len(pu)} pseudo-pairs [{mode}]")
        smp = T.Sampler(pu, ph, np.arange(len(pu)), bs=bs, seed=1000 * seed + 2)
        pre_info = T.train_regression(net, smp, pre_steps, lr=2e-4, log=print,
                                      val_fn=val_fn, val_every=2000)
        del pu, ph

    smp = T.Sampler(coh.ulf, coh.hf, coh.rows_for(tr), bs=bs, seed=1000 * seed)
    info = T.train_regression(net, smp, ft_steps, lr=1e-4, log=print, val_fn=val_fn, val_every=1000)
    rows = T.evaluate(net, test, test.subjects, in_fn_of(test))
    agg = M.aggregate(rows)
    tag = tag or mode
    json.dump({"contrast": contrast, "mode": mode, "n_subj": len(tr),
               "pre_steps": pre_steps, "ft_steps": ft_steps,
               "dev": info["best"], "pre_dev": pre_info["best"] if pre_info else None,
               "test": agg, "test_rows": rows},
              open(f"{OUT}/d1_{tag}_{contrast}.json", "w"), indent=1, default=float)
    torch.save(net.state_dict(), f"{OUT}/d1_{tag}_{contrast}.pt")
    print(f"\n== D1 TEST [{tag} {contrast}] ==")
    for k in ["psnr", "ssim", "nrmse", "cos_LL", "r_LL", "cos_HH", "r_HH", "hfpow"]:
        print(f"  {k:8s} {agg[k][0]:.4f} +- {agg[k][1]:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["forward", "pairs", "train"])
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--mode", default="learned_cal")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--pre_steps", type=int, default=12000)
    ap.add_argument("--ft_steps", type=int, default=8000)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n_subj", type=int, default=None)
    ap.add_argument("--ptag", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "forward": stage_forward(a.contrast, a.steps, n_subj=a.n_subj, tag=a.ptag)
    elif a.stage == "pairs": stage_pairs(a.contrast, a.mode, tag=a.ptag, n_subj=a.n_subj)
    else: stage_train(a.contrast, a.mode, a.pre_steps, a.ft_steps, tag=a.tag,
                      n_subj=a.n_subj, ptag=a.ptag, seed=a.seed)
