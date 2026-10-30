"""Can a detectability map be made sensitive to the pipeline it audits?

The cross-architecture study found the map is 99.7% between-site and 0.0% between-
architecture: it ignores the `p` channel that identifies the pipeline. That is redundancy
by construction - with one enhancer in training, `p = enh(x0)` carries nothing `x0` does
not, so nothing in the objective could teach the map to read it.

Train across several enhancers and `p` becomes informative. The question is whether the map
then picks it up, and specifically whether it gains the ability the single-pipeline map
lacks: ranking PIPELINES, not just sites within one.

  train on : U-Net L1, pix2pix, LoHiResGAN
  test on  : U-Net L1+SSIM, U-Net absolute, D1 calibrated corpus   (never seen)

Held-out architectures, so "it can rank pipelines" means pipelines it was not trained on.
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, models as Mo, train as T
from exp.d3qa import enhancer
from exp.d3detect import operator, site_pool, blobs
from exp.d3arch import load

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
CACHE = "/mnt/G/Projects/ULF-MICCAI/datasets/npy"
TRAIN_ARCH = [("U-Net L1", "base_unet_T2", "unet_res"),
              ("cGAN / pix2pix", "base_pix2pix_T2", "unet_res"),
              ("LoHiResGAN", "base_lohiresgan_T2", "lohires")]
TEST_ARCH = [("U-Net L1+SSIM", "base_unet_ssim_T2", "unet_res"),
             ("U-Net absolute", "base_unet_noresid_T2", "unet_abs"),
             ("D1 calibrated corpus", "d1_learned_cal_T2", "unet_res")]
RADII, CS = (2, 3, 4, 6), (-0.25, -0.12, 0.12, 0.25)


@torch.no_grad()
def labels_for(net, cohort, sids, fwd, n_sites=5, bs=12, seed=0):
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    recs = []
    for sid in sids:
        idx = cohort.subject_rows(sid)
        for i0 in range(0, len(idx), 6):
            j = idx[i0:i0 + 6]
            y0 = torch.from_numpy(cohort.hf[j].astype(np.float32)).cuda()
            x0 = fwd(y0, gen)
            ok = site_pool(y0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p0 = net(x0).float().clamp(0, 1)
            jobs = []
            for s in range(len(j)):
                cand = torch.nonzero(ok[s])
                if len(cand) < 8: continue
                for _ in range(n_sites):
                    cy, cx = cand[rng.integers(len(cand))].tolist()
                    jobs.append((s, cy, cx, int(rng.choice(RADII)), float(rng.choice(CS))))
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
                good = torch.tensor([bool(core[k].sum() >= 4 and ring[k].sum() >= 8)
                                     for k in range(len(ch))], device="cuda")
                tru = torch.stack([(dy[k][core[k]].mean() - dy[k][ring[k]].mean())
                                   if good[k] else torch.tensor(0., device="cuda")
                                   for k in range(len(ch))])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pl = net(fwd(yl, gen)).float().clamp(0, 1)
                dp = pl - p0[si]
                rec = torch.stack([(dp[k][core[k]].mean() - dp[k][ring[k]].mean())
                                   if good[k] else torch.tensor(0., device="cuda")
                                   for k in range(len(ch))])
                den = torch.where(tru.abs() < 1e-6, torch.full_like(tru, 1e-6), tru)
                ret = (rec / den).clamp(-0.5, 1.5)
                for k in range(len(ch)):
                    if good[k] and tru[k].abs() > 0.25 * cc[k, 0, 0, 0].abs():
                        recs.append((int(j[ch[k][0]]), ch[k][1], ch[k][2],
                                     ch[k][3], ch[k][4], float(ret[k])))
        print(f"    {sid}: {len(recs)}", flush=True)
    return recs


def train_multi(coh, per_arch, nets, fwd, steps=10000, bs=6, seed=0, tag="_multi"):
    """One map, batches drawn across architectures so `p` carries signal."""
    T.seed_all(seed)
    flat = []
    for nm, recs in per_arch.items():
        by = {}
        for row, cy, cx, r, c, ret in recs:
            by.setdefault(row, []).append((cy, cx, r, c, ret))
        for row, v in by.items():
            flat.append((nm, row, v))
    net = Mo.UNet(6, 1, base=32, residual_from=None).cuda()
    opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-4)
    lrs = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    for it in range(steps):
        pick = [flat[i] for i in rng.choice(len(flat), bs, replace=False)]
        rows = np.array([p[1] for p in pick])
        y0 = torch.from_numpy(coh.hf[rows].astype(np.float32)).cuda()
        x0 = fwd(y0, gen)
        p0 = torch.zeros_like(y0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for k, (nm, _, _) in enumerate(pick):
                p0[k] = nets[nm](x0[k:k+1]).float().clamp(0, 1)[0]
        tgt = torch.zeros_like(y0); msk = torch.zeros_like(y0)
        rm = torch.zeros_like(y0); cm = torch.zeros_like(y0)
        for k, (nm, row, v) in enumerate(pick):
            cy, cx, r, c, ret = v[rng.integers(len(v))]
            lo_y, hi_y = max(0, cy - r), min(256, cy + r + 1)
            lo_x, hi_x = max(0, cx - r), min(256, cx + r + 1)
            tgt[k, 0, lo_y:hi_y, lo_x:hi_x] = ret
            msk[k, 0, lo_y:hi_y, lo_x:hi_x] = 1
            rm[k] = r / 6.0; cm[k] = c / 0.25
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = net(torch.cat([x0, p0, rm, cm], 1)).float()
        loss = ((q - tgt).abs() * msk).sum() / msk.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); lrs.step()
        if (it + 1) % 2000 == 0:
            print(f"  step {it+1:6d}  L1 {loss.item():.5f}", flush=True)
    torch.save(net.state_dict(), f"{OUT}/d3detect_T2{tag}.pt")
    return net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--tag", default="_multi")
    a = ap.parse_args()
    T.seed_all(0)
    coh, tr, val = D.dev_split("T2")
    fwd = operator("T2", "learned")
    nets = {nm: load(kind, path) for nm, path, kind in TRAIN_ARCH}
    per_arch = {}
    for nm, path, kind in TRAIN_ARCH:
        f = f"{CACHE}/multilab_{path}.npy"
        if os.path.exists(f):
            per_arch[nm] = [tuple(r) for r in np.load(f, allow_pickle=True)]
            print(f"  reusing {len(per_arch[nm])} labels for {nm}")
        else:
            print(f"  labels for {nm}")
            per_arch[nm] = labels_for(nets[nm], coh, tr, fwd)
            np.save(f, np.array(per_arch[nm], dtype=object), allow_pickle=True)
        print(f"  {nm}: {len(per_arch[nm])} labels, mean retention "
              f"{np.mean([r[5] for r in per_arch[nm]]):.3f}", flush=True)
    torch.cuda.empty_cache()
    train_multi(coh, per_arch, nets, fwd, steps=a.steps, tag=a.tag)
