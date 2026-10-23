"""Did the detectability map learn the pipeline, or did it learn Gaussian blobs?

It was trained on one lesion family: isotropic hyperintense Gaussians, radius 2-6 px,
contrast 0.12/0.25, in eroded mid-intensity tissue. Every number so far is therefore
in-distribution. If the map memorised that family the result is worthless; if it
generalises to lesion types it has never seen, it has learned a property of the
synthesis.

Five held-out families, each changing one thing:
  iso        the training family, as the reference point
  hypo       negative contrast - the sign flips
  elong      anisotropic, 3:1 aspect, random orientation - shape
  ring       an annulus rather than a filled blob - topology
  faint      contrast 0.06, below the trained range
  big        radius 8, above the trained range
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F_
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, models as Mo, train as T
from exp.d3qa import enhancer
from exp.d3detect import operator, site_pool

OUT = "/mnt/G/Projects/ULF-MICCAI/out"


def profile(kind, cy, cx, r, H=256, W=256, rng=None):
    yy, xx = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"),
                            indexing="ij")
    dy = (yy[None] - cy[:, None, None]).float()
    dx = (xx[None] - cx[:, None, None]).float()
    rr = r[:, None, None].float()
    if kind == "elong":
        th = torch.tensor(rng.uniform(0, np.pi, len(cy)), device="cuda",
                          dtype=torch.float32)[:, None, None]
        u = dy * torch.cos(th) + dx * torch.sin(th)
        v = -dy * torch.sin(th) + dx * torch.cos(th)
        d2 = (u / (rr * 1.8)) ** 2 + (v / (rr / 1.8)) ** 2
        return torch.exp(-2.0 * d2)[:, None]
    d2 = (dy ** 2 + dx ** 2) / rr ** 2
    if kind == "ring":
        return torch.exp(-8.0 * (d2.sqrt() - 0.7) ** 2)[:, None]
    return torch.exp(-2.0 * d2)[:, None]


FAMILIES = {
    "iso   (training family)": dict(kind="iso",   radii=(2, 3, 4, 6), cs=(0.12, 0.25)),
    "hypo  (negative)":        dict(kind="iso",   radii=(2, 3, 4, 6), cs=(-0.12, -0.25)),
    "elong (3:1 aspect)":      dict(kind="elong", radii=(2, 3, 4, 6), cs=(0.12, 0.25)),
    "ring  (annulus)":         dict(kind="ring",  radii=(3, 4, 6),    cs=(0.12, 0.25)),
    "faint (c=0.06, unseen)":  dict(kind="iso",   radii=(2, 3, 4, 6), cs=(0.06,)),
    "big   (r=8, unseen)":     dict(kind="iso",   radii=(8,),         cs=(0.12, 0.25)),
}


@torch.no_grad()
def run_family(name, cfg, contrast, test, enh, fwd, det, n_sites=4, bs=16, seed=3):
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    recs = []
    for sid in test.subjects:
        idx = test.subject_rows(sid)
        sel = idx[np.linspace(len(idx)//5, 4*len(idx)//5, 6).astype(int)]
        for i0 in range(0, len(sel), 4):
            j = sel[i0:i0+4]
            y0 = torch.from_numpy(test.hf[j].astype(np.float32)).cuda()
            ok = site_pool(y0, r_max=8)
            x0 = fwd(y0, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p0 = enh(x0).float().clamp(0, 1)
            jobs = []
            for s in range(len(j)):
                cand = torch.nonzero(ok[s])
                if len(cand) < 8: continue
                for _ in range(n_sites):
                    cy, cx = cand[rng.integers(len(cand))].tolist()
                    jobs.append((s, cy, cx, int(rng.choice(cfg["radii"])),
                                 float(rng.choice(cfg["cs"]))))
            for b0 in range(0, len(jobs), bs):
                ch = jobs[b0:b0+bs]
                si = torch.tensor([t[0] for t in ch], device="cuda")
                cy = torch.tensor([t[1] for t in ch], device="cuda")
                cx = torch.tensor([t[2] for t in ch], device="cuda")
                rr = torch.tensor([t[3] for t in ch], device="cuda")
                cc = torch.tensor([t[4] for t in ch], device="cuda")[:, None, None, None]
                b = profile(cfg["kind"], cy, cx, rr, rng=rng) * (y0[si] > 0).float()
                yl = (y0[si] + cc * b).clamp(0, 1)
                dyl = yl - y0[si]
                core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
                good = torch.tensor([bool(core[k].sum() >= 4 and ring[k].sum() >= 8)
                                     for k in range(len(ch))], device="cuda")
                if not good.any(): continue
                tru = torch.stack([(dyl[k][core[k]].mean() - dyl[k][ring[k]].mean())
                                   if good[k] else torch.tensor(0., device="cuda")
                                   for k in range(len(ch))])
                xl = fwd(yl, gen)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pl = enh(xl).float().clamp(0, 1)
                dp = pl - p0[si]
                rec = torch.stack([(dp[k][core[k]].mean() - dp[k][ring[k]].mean())
                                   if good[k] else torch.tensor(0., device="cuda")
                                   for k in range(len(ch))])
                ret = (rec / tru.sign().clamp(min=-1) / tru.abs().clamp(min=1e-6)).clamp(-0.5, 1.5)
                # the map is queried with the magnitude of radius/contrast it understands
                for k in range(len(ch)):
                    if not good[k] or tru[k].abs() < 0.25 * abs(ch[k][4]): continue
                    rmap = torch.full_like(p0[si][k:k+1], min(ch[k][3], 6) / 6.0)
                    cmap = torch.full_like(p0[si][k:k+1], abs(ch[k][4]) / 0.25)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        d = det(torch.cat([x0[si][k:k+1], p0[si][k:k+1], rmap, cmap], 1)).float()
                    r_ = ch[k][3]
                    sl = (slice(max(0, ch[k][1]-r_), min(256, ch[k][1]+r_+1)),
                          slice(max(0, ch[k][2]-r_), min(256, ch[k][2]+r_+1)))
                    recs.append(dict(sid=sid, r=r_, c=ch[k][4],
                                     retention=float(ret[k]), detect=float(d[0, 0][sl].mean())))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--op", default="learned")
    ap.add_argument("--thr", type=float, default=0.25)
    a = ap.parse_args()
    T.seed_all(3)
    test = D.PairedCohort("test", a.contrast)
    enh = enhancer(); enh.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_{a.contrast}.pt")); enh.eval()
    det = Mo.UNet(6, 1, base=32, residual_from=None).cuda()
    det.load_state_dict(torch.load(f"{OUT}/d3detect_{a.contrast}.pt")); det.eval()
    fwd = operator(a.contrast, a.op)

    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    out = {}
    print(f"[{a.contrast}] trained on 'iso' only; every other row is a family never seen\n")
    print(f"  {'lesion family':26s} {'n':>5s} {'% erased':>9s} {'AUROC':>7s} {'spearman':>9s} {'mean ret':>9s}")
    for nm, cfg in FAMILIES.items():
        rec = run_family(nm, cfg, a.contrast, test, enh, fwd, det)
        if len(rec) < 50:
            print(f"  {nm:26s} {len(rec):5d}   too few"); continue
        ret = np.array([x["retention"] for x in rec])
        dsc = np.array([x["detect"] for x in rec])
        er = (ret < a.thr).astype(np.int8)
        au = roc_auc_score(er, -dsc) if 0 < er.sum() < len(er) else float("nan")
        rho = spearmanr(dsc, ret).statistic
        out[nm] = dict(n=len(rec), erased=float(er.mean()), auroc=float(au),
                       spearman=float(rho), mean_ret=float(ret.mean()))
        print(f"  {nm:26s} {len(rec):5d} {er.mean()*100:8.0f}% {au:7.3f} {rho:9.3f} {ret.mean():9.3f}")
    json.dump(out, open(f"{OUT}/d3ood_{a.contrast}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
