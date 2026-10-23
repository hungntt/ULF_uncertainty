"""Is the learned forward operator a better 64 mT simulator than the hand-crafted one?

The 48 ULF-only subjects never appear in any training set, so they give an
independent answer: push the 3T corpus through each candidate operator and ask
which output lands in the real 64 mT distribution.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, models as Mo
from exp.d1 import handcrafted_D, load_forward, radial_bins, mean_radial, spectral_calibrate

OUT = "/mnt/G/Projects/ULF-MICCAI/out"


def patch_stats(x, n=4000, seed=0, bs=128):
    """Marginal + local-texture summary: intensity histogram and gradient energy."""
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(x), min(n, len(x)), replace=False))
    hs, gs = [], []
    for i in range(0, len(sel), bs):
        a = torch.from_numpy(x[sel[i:i + bs], 0 if x.shape[1] == 1 else 1].astype(np.float32)).cuda()
        m = a > 0
        hs.append(torch.histc(a[m], bins=64, min=0, max=1))
        gx = (a[:, :, 1:] - a[:, :, :-1]).abs()
        gy = (a[:, 1:] - a[:, :-1]).abs()
        gs.append(torch.stack([gx.mean(), gy.mean(), gx.std(), a[m].mean(), a[m].std()]))
    h = torch.stack(hs).sum(0); h = h / h.sum()
    return h.cpu().numpy(), torch.stack(gs).mean(0).cpu().numpy()


def jsd(p, q, eps=1e-10):
    p, q = p + eps, q + eps
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda a, b: float((a * np.log(a / b)).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def apply_op(ixi, op, net=None, rb=None, rs=None, bs=64):
    out = np.empty((len(ixi), 1, 256, 256), np.float16)
    g = torch.Generator(device="cuda"); g.manual_seed(0)
    with torch.no_grad():
        for i in range(0, len(ixi), bs):
            a = torch.from_numpy(ixi[i:i + bs, 0].astype(np.float32)).cuda()[:, None]
            if op == "handcrafted":
                d = handcrafted_D(a, gen=g)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    d = net(a).float().clamp(0, 1)
                w = torch.randn(d.shape, device="cuda", generator=g)
                Wf = torch.fft.fftshift(torch.fft.fft2(w[:, 0]), dim=(-2, -1)) * torch.sqrt(rs.clamp(min=0))[rb][None] / 256.0
                d = (d + torch.fft.ifft2(torch.fft.ifftshift(Wf, dim=(-2, -1))).real[:, None]).clamp(0, 1) * (a > 0)
            out[i:i + bs] = d.cpu().numpy().astype(np.float16)
    return out


def main(contrast="T2", n=2500):
    rb = radial_bins()
    real = D.unpaired_ulf(contrast)[:n]                      # 48 held-out ULF subjects
    ixi = D.unpaired_3t(contrast)[:n]
    net = load_forward(contrast)
    rs = torch.from_numpy(np.load(f"{OUT}/d1_resid_spec_{contrast}.npy")).cuda()
    coh, tr, _ = D.dev_split(contrast)
    gain = torch.sqrt(mean_radial(coh.hf[coh.rows_for(tr)], rb) /
                      mean_radial(D.unpaired_3t(contrast), rb).clamp(min=1e-20))
    gain[:2] = 1.0

    cands = {"hand-crafted D (blur+Rician)": apply_op(ixi, "handcrafted"),
             "learned F": apply_op(ixi, "learned", net, rb, rs),
             "learned F on calibrated corpus": apply_op(spectral_calibrate(ixi, gain, rb), "learned", net, rb, rs)}

    h_real, g_real = patch_stats(real)
    s_real = mean_radial(real[:, 1:2], rb).cpu().numpy()
    print(f"=== {contrast}: distance to the real 64 mT pool (48 unseen ULF-only subjects) ===")
    print(f"{'operator':34s} {'JSD(hist)':>10s} {'|dgrad|':>9s} {'log-spec MSE':>13s}")
    res = {}
    for nm, arr in cands.items():
        h, g = patch_stats(arr)
        s = mean_radial(arr, rb).cpu().numpy()
        lo, hi = 2, 110
        sm = float(np.mean((np.log10(s[lo:hi] + 1e-20) - np.log10(s_real[lo:hi] + 1e-20)) ** 2))
        dg = float(np.abs(g[:3] - g_real[:3]).mean())
        res[nm] = dict(jsd=jsd(h, h_real), dgrad=dg, logspec_mse=sm)
        print(f"{nm:34s} {res[nm]['jsd']:10.5f} {dg:9.5f} {sm:13.5f}")
    # reference: how far is the paired cohort's own ULF from the ULF-only pool?
    hp, gp = patch_stats(coh.ulf)
    sp = mean_radial(coh.ulf[:, 1:2], rb).cpu().numpy()
    lo, hi = 2, 110
    print(f"{'[ref] real paired-cohort ULF':34s} {jsd(hp, h_real):10.5f} "
          f"{float(np.abs(gp[:3]-g_real[:3]).mean()):9.5f} "
          f"{float(np.mean((np.log10(sp[lo:hi]+1e-20)-np.log10(s_real[lo:hi]+1e-20))**2)):13.5f}")
    res["[ref] real paired-cohort ULF"] = dict(
        jsd=jsd(hp, h_real), dgrad=float(np.abs(gp[:3] - g_real[:3]).mean()),
        logspec_mse=float(np.mean((np.log10(sp[lo:hi] + 1e-20) - np.log10(s_real[lo:hi] + 1e-20)) ** 2)))
    json.dump(res, open(f"{OUT}/d1_operator_check_{contrast}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "T2")
