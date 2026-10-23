"""Heatmap-guided re-acquisition: given budget to re-scan part of the volume, does
the QA map pick the right part?

A Hyperfine cannot improve an arbitrary blob, but it can re-run the sequence over a
slab or take extra averages. So the actionable question is *which slices to re-scan*.

Acquisition is simulated with the Direction-1 forward operator F, holding its blur
and bandwidth limits fixed and varying only the noise, so a "re-acquired" slice
differs from a standard one exactly as extra averages would (NEX=2 -> sigma/sqrt(2),
independent realisation) and no 3T detail leaks in through the back door. Everything
is trained on simulated-standard input so the two passes share one domain.

Budget accounting: re-scanning a fraction B of slices costs B extra acquisitions, so
the budget-matched uniform alternative gives every slice B extra averages
(sigma -> sigma/sqrt(1+B)).
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T
from exp.d1 import load_forward, radial_bins, handcrafted_D

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
CACHE = "/tmp/claude-17055/-mnt-G/5ec3b7fc-b916-4b70-887e-dbe4d6709b39/scratchpad/data"


@torch.no_grad()
def simulate_res(contrast, split, frac, seed, bs=12):
    """Resolution-limited acquisition: the hand-crafted physics chain with its k-space
    truncation pinned at `frac`. Noise and every other term held fixed, so standard and
    re-acquired differ only in how much k-space was sampled - which is what a thinner
    slice or larger matrix actually buys, and what the learned F cannot express."""
    coh = D.PairedCohort(split, contrast)
    g = torch.Generator(device="cuda"); g.manual_seed(seed)
    n = len(coh.hf)
    nb = np.arange(n)[:, None] + np.array([-1, 0, 1])[None]
    for s in coh.man:
        a, b = s["slice_start"], s["slice_start"] + s["slice_count"]
        nb[a:b] = np.clip(nb[a:b], a, b - 1)
    out = np.empty((n, 3, 256, 256), np.float16)
    for i in range(0, n, bs):
        trip = nb[i:i + bs].reshape(-1)
        y = torch.from_numpy(coh.hf[trip].astype(np.float32)).cuda()
        d = handcrafted_D(y, snr_db=18.0, gen=g, physics=True, frac=frac)
        out[i:i + bs] = d.reshape(-1, 3, 256, 256).cpu().numpy().astype(np.float16)
    return out


def gen_res(contrast, std_frac=0.50, hi_frac=0.75):
    for split in ("train", "test"):
        for nm, fr, sd in (("rstd", std_frac, 11), ("rhi", hi_frac, 12)):
            f = f"{CACHE}/acq_{split}_{nm}_{contrast}.npy"
            if os.path.exists(f):
                print(f"  {os.path.basename(f)} exists"); continue
            np.save(f, simulate_res(contrast, split, fr, sd))
            print(f"  wrote {os.path.basename(f)} (k-space fraction {fr})", flush=True)


@torch.no_grad()
def simulate(contrast, split, scale, seed, bs=12):
    """F(3T) + coloured noise at `scale` x the measured ULF residual amplitude,
    returned as (N,3,256,256) 2.5D slabs built from contiguous slices."""
    coh = D.PairedCohort(split, contrast)
    net = load_forward(contrast)
    rs = torch.from_numpy(np.load(f"{OUT}/d1_resid_spec_{contrast}.npy")).cuda()
    rb = radial_bins()
    g = torch.Generator(device="cuda"); g.manual_seed(seed)

    n = len(coh.hf)
    nb = np.arange(n)[:, None] + np.array([-1, 0, 1])[None]
    for s in coh.man:                                   # never cross a subject boundary
        a, b = s["slice_start"], s["slice_start"] + s["slice_count"]
        nb[a:b] = np.clip(nb[a:b], a, b - 1)
    out = np.empty((n, 3, 256, 256), np.float16)
    for i in range(0, n, bs):
        trip = nb[i:i + bs].reshape(-1)
        y = torch.from_numpy(coh.hf[trip].astype(np.float32)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d = net(y).float().clamp(0, 1)
        w = torch.randn(d.shape, device="cuda", generator=g)
        Wf = torch.fft.fftshift(torch.fft.fft2(w[:, 0]), dim=(-2, -1))
        Wf = Wf * torch.sqrt(rs.clamp(min=0))[rb][None] / 256.0 * scale
        nz = torch.fft.ifft2(torch.fft.ifftshift(Wf, dim=(-2, -1))).real[:, None]
        d = ((d + nz).clamp(0, 1) * (y > 0)).reshape(-1, 3, 256, 256)
        out[i:i + bs] = d.cpu().numpy().astype(np.float16)
    del net; torch.cuda.empty_cache()
    return out


def gen_all(contrast):
    for split in ("train", "test"):
        for nm, scale, seed in (("std", 1.0, 1), ("reacq", 1 / np.sqrt(2), 2)):
            f = f"{CACHE}/acq_{split}_{nm}_{contrast}.npy"
            if os.path.exists(f):
                print(f"  {os.path.basename(f)} exists"); continue
            np.save(f, simulate(contrast, split, scale, seed))
            print(f"  wrote {os.path.basename(f)}", flush=True)
    # budget-matched uniform alternatives
    for B in (0.10, 0.20, 0.30):
        f = f"{CACHE}/acq_test_unif{int(B*100)}_{contrast}.npy"
        if not os.path.exists(f):
            np.save(f, simulate(contrast, "test", 1 / np.sqrt(1 + B), 3))
            print(f"  wrote uniform B={B}", flush=True)


def halves(contrast):
    """16 subjects to fit the enhancer, 16 for the QA map, so the QA map only ever
    sees out-of-sample predictions without needing full cross-fitting."""
    coh, tr, _ = D.dev_split(contrast)
    tr = sorted(tr)
    return coh, tr[0::2], tr[1::2]


def train_models(contrast, steps=9000, bs=16, kind="std"):
    T.seed_all(0)
    coh, a_ids, b_ids = halves(contrast)
    std = np.load(f"{CACHE}/acq_train_{kind}_{contrast}.npy")

    print(f"[{contrast}] enhancer on {len(a_ids)} subjects (simulated standard input)")
    enh = Mo.UNet(3, 1, base=32, residual_from=None).cuda()
    T.train_regression(enh, T.Sampler(std, coh.hf, coh.rows_for(a_ids), bs=bs, seed=0),
                       steps, lr=2e-4, log=None)
    torch.save(enh.state_dict(), f"{OUT}/d3acq_enh_{kind}_{contrast}.pt")

    print(f"[{contrast}] QA map on the other {len(b_ids)} subjects")
    enh.eval()
    rows_b = coh.rows_for(b_ids)
    pred = np.zeros_like(coh.hf)
    with torch.no_grad():
        for i in range(0, len(rows_b), 32):
            j = rows_b[i:i + 32]
            x = torch.from_numpy(std[j].astype(np.float32)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred[j] = enh(x).float().clamp(0, 1).cpu().numpy().astype(pred.dtype)
    qa = Mo.UNet(4, 1, base=32, residual_from=None).cuda()
    opt = torch.optim.AdamW(qa.parameters(), 2e-4, weight_decay=1e-4)
    lrs = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    for it in range(steps):
        j = rows_b[np.sort(rng.choice(len(rows_b), bs, replace=False))]
        x = torch.from_numpy(std[j].astype(np.float32)).cuda()
        p = torch.from_numpy(pred[j].astype(np.float32)).cuda()
        y = torch.from_numpy(coh.hf[j].astype(np.float32)).cuda()
        if rng.random() < 0.5: x, p, y = (torch.flip(v, [-1]) for v in (x, p, y))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = qa(torch.cat([x, p], 1)).float()
        d = (p - y).abs() - q
        loss = torch.maximum(0.9 * d, -0.1 * d).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(qa.parameters(), 1.0); opt.step(); lrs.step()
        if (it + 1) % 3000 == 0:
            print(f"  step {it+1}  pinball {loss.item():.6f}", flush=True)
    torch.save(qa.state_dict(), f"{OUT}/d3acq_qa_{kind}_{contrast}.pt")


@torch.no_grad()
def run(contrast, budgets=(0.10, 0.20, 0.30), kind="std", hi="reacq"):
    T.seed_all(0)
    test = D.PairedCohort("test", contrast)
    enh = Mo.UNet(3, 1, base=32, residual_from=None).cuda()
    enh.load_state_dict(torch.load(f"{OUT}/d3acq_enh_{kind}_{contrast}.pt")); enh.eval()
    qa = Mo.UNet(4, 1, base=32, residual_from=None).cuda()
    qa.load_state_dict(torch.load(f"{OUT}/d3acq_qa_{kind}_{contrast}.pt")); qa.eval()
    std = np.load(f"{CACHE}/acq_test_{kind}_{contrast}.npy")
    reacq = np.load(f"{CACHE}/acq_test_{hi}_{contrast}.npy")
    unif = {}
    for B in budgets:
        f = f"{CACHE}/acq_test_unif{int(B*100)}_{contrast}.npy"
        unif[B] = np.load(f) if os.path.exists(f) else None

    def synth(arr, idx, bs=8):
        P = []
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            x = torch.from_numpy(arr[j].astype(np.float32)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                P.append(enh(x).float().clamp(0, 1))
        return torch.cat(P)

    res = {"contrast": contrast, "budgets": list(budgets), "subjects": {}}
    for sid in test.subjects:
        idx = test.subject_rows(sid)
        y = torch.from_numpy(test.hf[idx].astype(np.float32)).cuda()
        x0 = torch.from_numpy(std[idx].astype(np.float32)).cuda()
        m = ((x0[:, 1:2] > 0) | (y > 0)).float()
        p0 = synth(std, idx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = qa(torch.cat([x0, p0], 1)).float()
        p1 = synth(reacq, idx)
        e0 = ((p0 - y).abs() * m).sum((1, 2, 3))
        e1 = ((p1 - y).abs() * m).sum((1, 2, 3))
        w = m.sum((1, 2, 3))
        sl_q = ((q * m).sum((1, 2, 3)) / w.clamp(min=1)).cpu().numpy()   # per-slice QA
        sl_e = (e0 / w.clamp(min=1)).cpu().numpy()                        # per-slice true error
        base = (e0.sum() / w.sum()).item()
        n = len(idx)
        rng = np.random.default_rng(abs(hash(sid)) % 2**31)
        sub = {"base": base, "n_slices": int(n)}
        for B in budgets:
            k = max(1, int(round(B * n)))
            def mix(sel):
                # e0/e1 are per-slice error sums, shape (n,). Indexing with [:, None]
                # here broadcast to (n, n) and produced nonsense.
                sel = torch.from_numpy(np.isin(np.arange(n), sel)).cuda()
                return (torch.where(sel, e1, e0).sum() / w.sum()).item()
            def best_slab(score):
                c = np.convolve(score, np.ones(k), "valid")
                s = int(np.argmax(c)); return np.arange(s, s + k)
            sub[f"B{int(B*100)}"] = {
                "qa_topk":   mix(np.argsort(-sl_q)[:k]),
                "qa_slab":   mix(best_slab(sl_q)),
                "oracle":    mix(np.argsort(-sl_e)[:k]),
                "random":    float(np.mean([mix(rng.choice(n, k, replace=False)) for _ in range(5)])),
                "uniform":   (((((synth(unif[B], idx) - y).abs() * m).sum()) / w.sum()).item()
                              if unif[B] is not None else float("nan")),
                "all_reacq": (e1.sum() / w.sum()).item(),
            }
        res["subjects"][sid] = sub
        print(f"  {sid}: base {base:.5f}", flush=True)

    print(f"\n== mean brain |error| after spending the budget ({contrast}) ==")
    print(f"  {'budget':>8s} {'no re-scan':>11s} {'random':>9s} {'uniform':>9s} "
          f"{'QA slab':>9s} {'QA top-k':>9s} {'oracle':>9s} {'re-scan all':>12s}")
    base = np.mean([s["base"] for s in res["subjects"].values()])
    for B in budgets:
        g = lambda k: np.mean([s[f"B{int(B*100)}"][k] for s in res["subjects"].values()])
        print(f"  {int(B*100):7d}% {base:11.5f} {g('random'):9.5f} {g('uniform'):9.5f} "
              f"{g('qa_slab'):9.5f} {g('qa_topk'):9.5f} {g('oracle'):9.5f} {g('all_reacq'):12.5f}")
    print(f"\n  fraction of the achievable gain captured (vs re-scanning everything):")
    for B in budgets:
        g = lambda k: np.mean([s[f"B{int(B*100)}"][k] for s in res["subjects"].values()])
        full = base - g("all_reacq")
        if full <= 0: continue
        print(f"  {int(B*100):7d}% random {100*(base-g('random'))/full:5.1f}%  "
              f"uniform {100*(base-g('uniform'))/full:5.1f}%  "
              f"QA slab {100*(base-g('qa_slab'))/full:5.1f}%  "
              f"QA top-k {100*(base-g('qa_topk'))/full:5.1f}%  "
              f"oracle {100*(base-g('oracle'))/full:5.1f}%")
    json.dump(res, open(f"{OUT}/d3acq_{contrast}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "train", "run", "gen_res"])
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--kind", default="std")
    ap.add_argument("--hi", default="reacq")
    a = ap.parse_args()
    if a.stage == "gen": gen_all(a.contrast)
    elif a.stage == "gen_res": gen_res(a.contrast)
    elif a.stage == "train": train_models(a.contrast, a.steps, kind=a.kind)
    else: run(a.contrast, kind=a.kind, hi=a.hi)
