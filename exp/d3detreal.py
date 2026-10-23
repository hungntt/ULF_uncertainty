"""A detectability map trained on the acquisition it will be deployed on.

d3realraw showed the map is tied to the forward operator: hold the estimator fixed, swap
the operator's rendering for a real 64 mT acquisition, and its site ranking halves
(spearman +0.376 -> +0.191, AUROC 0.554). The map describes the simulated pipeline.

The repair is to build the labels on the real input. A lesion is inserted into the 3T, and
only the operator's LOCAL response to it is added to the real 64 mT slab:

    x_lesioned = x_real + [ f(y + c*b) - f(y) ] * near

so the measured noise, texture and artefacts of the real acquisition survive and only the
lesion is synthetic. The operator still sets how a lesion looks, but it no longer sets how
the brain looks - which is the part the map was over-fitting to.
"""
import argparse, json, os, sys
import numpy as np, torch
import torch.nn.functional as F_
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, models as Mo, train as T
from exp.d3qa import enhancer
from exp.d3detect import operator, site_pool, blobs

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
CACHE = "/tmp/claude-17055/-mnt-G/5ec3b7fc-b916-4b70-887e-dbe4d6709b39/scratchpad/data"
RADII, CS = (2, 3, 4, 6), (-0.25, -0.12, 0.12, 0.25)


def lesion_input(x_real, y, yl, fwd, gen, cy, cx, rr):
    """Real slab plus the operator's LOCAL response to the lesion."""
    near = (blobs(cy, cx, torch.clamp(rr.float() * 4, min=24)) > 0.05).float()
    return x_real + (fwd(yl, gen) - fwd(y, gen)) * near


@torch.no_grad()
def make_labels(cohort, sids, enh, fwd, n_sites=6, bs=12, seed=0, mode="real"):
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    recs = []
    for sid in sids:
        idx = cohort.subject_rows(sid)
        for i0 in range(0, len(idx), 6):
            j = idx[i0:i0 + 6]
            y0 = torch.from_numpy(cohort.hf[j].astype(np.float32)).cuda()
            xr = (torch.from_numpy(cohort.ulf[j].astype(np.float32)).cuda()
                  if mode == "real" else fwd(y0, gen))
            ok = site_pool(y0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p0 = enh(xr).float().clamp(0, 1)          # baseline from the REAL input
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
                xl = (lesion_input(xr[si], y0[si], yl, fwd, gen, cy, cx, rr)
                      if mode == "real" else fwd(yl, gen))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pl = enh(xl).float().clamp(0, 1)
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
        print(f"  {sid}: {len(recs)} labels", flush=True)
    return recs


def train(cohort, recs, enh, steps=8000, bs=6, seed=0, tag="_realin", mode="real", fwd=None):
    T.seed_all(seed)
    by_row = {}
    for row, cy, cx, r, c, ret in recs:
        by_row.setdefault(row, []).append((cy, cx, r, c, ret))
    rows = sorted(by_row)
    net = Mo.UNet(6, 1, base=32, residual_from=None).cuda()
    opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-4)
    lrs = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(seed)
    for it in range(steps):
        pick = rng.choice(len(rows), bs, replace=False)
        j = np.array([rows[i] for i in pick])
        with torch.no_grad():
            if mode == "real":
                xr = torch.from_numpy(cohort.ulf[j].astype(np.float32)).cuda()
            else:
                g2 = torch.Generator(device="cuda"); g2.manual_seed(0)
                xr = fwd(torch.from_numpy(cohort.hf[j].astype(np.float32)).cuda(), g2)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p0 = enh(xr).float().clamp(0, 1)
        tgt = torch.zeros(bs, 1, 256, 256, device="cuda")
        msk = torch.zeros(bs, 1, 256, 256, device="cuda")
        rm = torch.zeros(bs, 1, 256, 256, device="cuda")
        cm = torch.zeros(bs, 1, 256, 256, device="cuda")
        for k, row in enumerate(j):
            cy, cx, r, c, ret = by_row[row][rng.integers(len(by_row[row]))]
            rm[k] = r / 6.0; cm[k] = c / 0.25
            tgt[k, 0, cy, cx] = ret; msk[k, 0, cy, cx] = 1.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = net(torch.cat([xr, p0, rm, cm], 1)).float()
        loss = ((q - tgt).abs() * msk).sum() / msk.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); lrs.step()
        if (it + 1) % 2000 == 0:
            print(f"  step {it+1:6d}  L1 {loss.item():.5f}", flush=True)
    torch.save(net.state_dict(), f"{OUT}/d3detect_T2{tag}.pt")
    return net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--tag", default="_realin")
    ap.add_argument("--input", default="real", choices=["real", "sim"],
                    help="sim runs THIS trainer on operator-rendered inputs - the control "
                         "that separates 'real inputs' from 'a different trainer'")
    a = ap.parse_args()
    T.seed_all(0)
    coh, tr, val = D.dev_split("T2")
    enh = enhancer(); enh.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_T2.pt")); enh.eval()
    fwd = operator("T2", "learned")
    lab = f"{CACHE}/detect_labels_T2{a.tag}.npy"
    if os.path.exists(lab):
        recs = [tuple(r) for r in np.load(lab, allow_pickle=True)]
        print(f"  reusing {len(recs)} cached labels")
    else:
        print(f"[T2] {a.input.upper()}-input detectability labels on {len(tr)} training subjects")
        recs = make_labels(coh, tr, enh, fwd, mode=a.input)
        np.save(lab, np.array(recs, dtype=object), allow_pickle=True)
    print(f"  {len(recs)} labels; mean retention {np.mean([r[5] for r in recs]):.3f}")
    torch.cuda.empty_cache()
    train(coh, recs, enh, steps=a.steps, tag=a.tag, mode=a.input, fwd=fwd)
