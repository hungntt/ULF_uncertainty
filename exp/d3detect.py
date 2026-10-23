"""A detectability map: would this pipeline erase a lesion placed here?

The lesion study showed that the conformal guarantee is on voxel intensity error, and
that this is not clinical salience - 4 mm lesions lose 83% of their contrast while
raising the QA score only 1.34x. The fix is not a better estimator of intensity error
but a different target.

    d(v) = fraction of a lesion's contrast that survives synthesis, if one were at v

At inference no lesion is present; the map answers a counterfactual about the
pipeline's own sensitivity. That is the quantity a reader needs ("lesions under 5 mm
would be invisible here"), and it is the quantity an ensemble's disagreement cannot
represent, because model disagreement is about intensity, not about detectability.

Labels come from simulation, so no patient cohort is needed to train it. One lesion
per image, batched over (slice, site) pairs, so a label costs one forward pass slot
and lesions never interact through the receptive field.
"""
import argparse, json, os, sys, zlib
import numpy as np
import torch
import torch.nn.functional as F_
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T
from exp.d1 import load_forward, handcrafted_D
from exp.d3qa import enhancer

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
CACHE = "/tmp/claude-17055/-mnt-G/5ec3b7fc-b916-4b70-887e-dbe4d6709b39/scratchpad/data"


def blobs(cy, cx, r, H=256, W=256):
    """(B,1,H,W) lesion profiles, one per (cy,cx) pair."""
    yy, xx = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"),
                            indexing="ij")
    d2 = ((yy[None] - cy[:, None, None]) ** 2 + (xx[None] - cx[:, None, None]) ** 2).float()
    return torch.exp(-2.0 * d2 / (r[:, None, None].float() ** 2))[:, None]


def site_pool(y, r_max=6):
    """Interior tissue sites: inside an eroded brain mask, mid-intensity, so an inserted
    lesion is the only new structure and never clips the dynamic range."""
    m = (y > 0).float()
    er = -F_.max_pool2d(-m, 2 * r_max + 1, 1, r_max)
    ok = (er[:, 0] > 0.5) & (y[:, 0] > 0.12) & (y[:, 0] < 0.45)
    return ok


@torch.no_grad()
def make_labels(contrast, split, sids, cohort, enh, fwd, n_sites=6,
                radii=(2, 3, 4, 6), cs=(-0.25, -0.12, 0.12, 0.25), bs=24, seed=0):
    """-> per-slice sparse labels: (slice_row, cy, cx, r, c, retention)."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    recs = []
    for sid in sids:
        idx = cohort.subject_rows(sid)
        for i0 in range(0, len(idx), 8):
            j = idx[i0:i0 + 8]
            y0 = torch.from_numpy(cohort.hf[j].astype(np.float32)).cuda()
            ok = site_pool(y0)
            x0 = fwd(y0, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p0 = enh(x0).float().clamp(0, 1)
            jobs = []
            for s in range(len(j)):
                cand = torch.nonzero(ok[s])
                if len(cand) < 8: continue
                for _ in range(n_sites):
                    cy, cx = cand[rng.integers(len(cand))].tolist()
                    jobs.append((s, cy, cx, int(rng.choice(radii)), float(rng.choice(cs))))
            for b0 in range(0, len(jobs), bs):
                ch = jobs[b0:b0 + bs]
                si = torch.tensor([t[0] for t in ch], device="cuda")
                cy = torch.tensor([t[1] for t in ch], device="cuda")
                cx = torch.tensor([t[2] for t in ch], device="cuda")
                rr = torch.tensor([t[3] for t in ch], device="cuda")
                cc = torch.tensor([t[4] for t in ch], device="cuda")[:, None, None, None]
                b = blobs(cy, cx, rr) * (y0[si] > 0).float()
                yl = (y0[si] + cc * b).clamp(0, 1)
                dy = yl - y0[si]
                core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
                tru = torch.stack([(dy[k][core[k]].mean() - dy[k][ring[k]].mean())
                                   for k in range(len(ch))])
                xl = fwd(yl, gen)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pl = enh(xl).float().clamp(0, 1)
                dp = pl - p0[si]
                rec = torch.stack([(dp[k][core[k]].mean() - dp[k][ring[k]].mean())
                                   for k in range(len(ch))])
                keep = tru.abs() > 0.25 * cc[:, 0, 0, 0].abs()
                # sign-preserving guard: clamping to +1e-6 would flip the ratio for
                # hypointense lesions, where tru is negative
                den = torch.where(tru.abs() < 1e-6, torch.full_like(tru, 1e-6), tru)
                ret = (rec / den).clamp(-0.5, 1.5)
                for k in range(len(ch)):
                    if keep[k]:
                        recs.append((int(j[ch[k][0]]), ch[k][1], ch[k][2],
                                     ch[k][3], ch[k][4], float(ret[k])))
        print(f"  {sid}: {len(recs)} labels", flush=True)
    return recs


def operator(contrast, which):
    if which == "learned":
        net = load_forward(contrast)
        def f(y, gen):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d = net(y).float().clamp(0, 1)
            return d.repeat(1, 3, 1, 1) * (y > 0)
    else:
        def f(y, gen):
            return handcrafted_D(y, gen=gen).repeat(1, 3, 1, 1) * (y > 0)
    return f


def train_detect(contrast, recs, cohort, enh, fwd, steps=8000, bs=6, base=32, seed=0, tag=""):
    """Dense U-Net, sparse supervision: the loss is evaluated only at labelled sites."""
    T.seed_all(seed)
    by_row = {}
    for row, cy, cx, r, c, ret in recs:
        by_row.setdefault(row, []).append((cy, cx, r, c, ret))
    rows = sorted(by_row)
    # 2 input maps for lesion geometry (radius, contrast) alongside the slab + prediction
    net = Mo.UNet(6, 1, base=base, residual_from=None).cuda()
    opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-4)
    lrs = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    for it in range(steps):
        pick = [rows[i] for i in rng.choice(len(rows), bs, replace=False)]
        y0 = torch.from_numpy(cohort.hf[np.array(pick)].astype(np.float32)).cuda()
        x0 = fwd(y0, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p0 = enh(x0).float().clamp(0, 1)
        tgt = torch.zeros(bs, 1, 256, 256, device="cuda")
        msk = torch.zeros_like(tgt)
        rmap = torch.zeros_like(tgt); cmap = torch.zeros_like(tgt)
        for k, row in enumerate(pick):
            cy, cx, r, c, ret = by_row[row][rng.integers(len(by_row[row]))]
            lo_y, hi_y = max(0, cy - r), min(256, cy + r + 1)
            lo_x, hi_x = max(0, cx - r), min(256, cx + r + 1)
            tgt[k, 0, lo_y:hi_y, lo_x:hi_x] = ret
            msk[k, 0, lo_y:hi_y, lo_x:hi_x] = 1
            rmap[k] = r / 6.0; cmap[k] = c / 0.25      # signed: carries polarity
        inp = torch.cat([x0, p0, rmap, cmap], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = net(inp).float()
        loss = ((q - tgt).abs() * msk).sum() / msk.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); lrs.step()
        if (it + 1) % 2000 == 0:
            print(f"  step {it+1:6d}  L1 {loss.item():.5f}", flush=True)
    torch.save(net.state_dict(), f"{OUT}/d3detect_{contrast}{tag}.pt")
    return net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--op", default="learned")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labtag", default=None,
                    help="label cache to reuse; defaults to --tag. Set it to share one "
                         "label set across seeds so only training stochasticity varies.")
    a = ap.parse_args()
    if a.labtag is None:
        a.labtag = a.tag
    T.seed_all(a.seed)
    coh, tr, val = D.dev_split(a.contrast)
    lab = f"{CACHE}/detect_labels_{a.contrast}{a.labtag}.npy"
    enh = enhancer(); enh.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_{a.contrast}.pt")); enh.eval()
    fwd = operator(a.contrast, a.op)
    if a.resume and os.path.exists(lab):
        recs = [tuple(r) for r in np.load(lab, allow_pickle=True)]
        print(f"  reusing {len(recs)} cached labels")
    else:
        print(f"[{a.contrast}] generating detectability labels on {len(tr)} training subjects")
        recs = make_labels(a.contrast, "train", sorted(tr), coh, enh, fwd)
        np.save(lab, np.array(recs, dtype=object), allow_pickle=True)
    print(f"  {len(recs)} labels; mean retention {np.mean([r[5] for r in recs]):.3f}")
    torch.cuda.empty_cache()
    train_detect(a.contrast, recs, coh, enh, fwd, steps=a.steps, seed=a.seed, tag=a.tag)
