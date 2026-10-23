"""Do the architectures that lose on PSNR preserve lesions better?

The baseline table ranks plain regression above both GANs on every distortion metric,
while the GANs produce far more high-frequency energy (r_HH 0.84 and 0.79 against 0.27
for L1). The perception-distortion trade-off is usually argued over how an image
*looks*. If the energy a critic restores also carries small structures through the
pipeline, the trade-off has a clinical dimension and the ranking inverts for the thing
a reader cares about.

Identical lesions through every trained T2 enhancer, same forward operator, same sites.
"""
import argparse, json, sys
import numpy as np
import torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, models as Mo, train as T
from ulfkit.metrics import paired_test
from exp.d3detect import operator, site_pool, blobs

OUT = "/mnt/G/Projects/ULF-MICCAI/out"

ARCHS = [
    ("U-Net L1",            "base_unet_T2",          "unet_res"),
    ("U-Net L1+SSIM",       "base_unet_ssim_T2",     "unet_res"),
    ("U-Net absolute",      "base_unet_noresid_T2",  "unet_abs"),
    ("cGAN / pix2pix",      "base_pix2pix_T2",       "unet_res"),
    ("LoHiResGAN",          "base_lohiresgan_T2",    "lohires"),
    ("D1 calibrated corpus", "d1_learned_cal_T2",    "unet_res"),
]


def load(kind, path):
    if kind == "lohires":
        n = Mo.LoHiResGen(3, 1).cuda()
    elif kind == "unet_abs":
        n = Mo.UNet(3, 1, base=32, residual_from=None).cuda()
    else:
        n = Mo.UNet(3, 1, base=32, residual_from=(1, 2)).cuda()
    n.load_state_dict(torch.load(f"{OUT}/{path}.pt")); n.eval()
    return n


@torch.no_grad()
def run(contrast="T2", n_sites=5, radii=(2, 3, 4, 6), cs=(0.12, 0.25), seed=11):
    T.seed_all(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    test = D.PairedCohort("test", contrast)
    fwd = operator(contrast, "learned")
    nets = []
    for nm, path, kind in ARCHS:
        try:
            nets.append((nm, load(kind, path)))
        except FileNotFoundError:
            print(f"  (missing {path}, skipped)")
    rows = []
    for sid in test.subjects:
        idx = test.subject_rows(sid)
        sel = idx[np.linspace(len(idx)//5, 4*len(idx)//5, 8).astype(int)]
        for i0 in range(0, len(sel), 4):
            j = sel[i0:i0+4]
            y0 = torch.from_numpy(test.hf[j].astype(np.float32)).cuda()
            ok = site_pool(y0)
            x0 = fwd(y0, gen)
            base = {}
            for nm, net in nets:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    base[nm] = net(x0).float().clamp(0, 1)
            jobs = []
            for s in range(len(j)):
                cand = torch.nonzero(ok[s])
                if len(cand) < 8: continue
                for _ in range(n_sites):
                    cy, cx = cand[rng.integers(len(cand))].tolist()
                    jobs.append((s, cy, cx, int(rng.choice(radii)), float(rng.choice(cs))))
            for b0 in range(0, len(jobs), 12):
                ch = jobs[b0:b0+12]
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
                xl = fwd(yl, gen)
                for nm, net in nets:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pl = net(xl).float().clamp(0, 1)
                    dp = pl - base[nm][si]
                    rec = torch.stack([(dp[k][core[k]].mean() - dp[k][ring[k]].mean())
                                       if good[k] else torch.tensor(0., device="cuda")
                                       for k in range(len(ch))])
                    for k in range(len(ch)):
                        if not good[k] or tru[k].abs() < 0.25 * cc[k, 0, 0, 0].abs():
                            continue
                        rows.append(dict(arch=nm, sid=sid, r=ch[k][3], c=ch[k][4],
                                         retention=float((rec[k] / tru[k]).clamp(-0.5, 1.5))))
        print(f"  {sid}: {len(rows)} measurements", flush=True)
    return rows


def report(rows, thr=0.25):
    archs = [nm for nm, _, _ in ARCHS if any(x["arch"] == nm for x in rows)]
    sids = sorted({x["sid"] for x in rows})
    ref = json.load(open(f"{OUT}/base_identity_T2.json"))  # only for the table header
    print(f"\n== lesion survival by enhancer architecture (identical lesions, same operator) ==")
    print(f"  {'architecture':22s} {'retention':>16s} {'% erased':>10s}")
    per = {}
    for nm in archs:
        R = [x for x in rows if x["arch"] == nm]
        ret = np.array([np.mean([x["retention"] for x in R if x["sid"] == s]) for s in sids])
        er = np.array([np.mean([x["retention"] < thr for x in R if x["sid"] == s]) for s in sids])
        per[nm] = (ret, er)
        print(f"  {nm:22s} {ret.mean():10.3f} ±{ret.std(ddof=1):.3f} {er.mean()*100:9.1f}%")
    anchor = "U-Net L1"
    if anchor in per:
        print(f"\n  against {anchor}:")
        for nm in archs:
            if nm == anchor: continue
            d = per[nm][0].mean() - per[anchor][0].mean()
            print(f"    {nm:22s} retention {d:+.3f}  (p={paired_test(per[nm][0], per[anchor][0]):.4f})")
    print(f"\n  % erased by lesion size:")
    print(f"  {'architecture':22s}" + "".join(f"{2*r:>8d} mm" for r in (2, 3, 4, 6)))
    for nm in archs:
        R = [x for x in rows if x["arch"] == nm]
        cells = []
        for r in (2, 3, 4, 6):
            v = [x["retention"] < thr for x in R if x["r"] == r]
            cells.append(f"{np.mean(v)*100:8.0f}%" if v else "       -")
        print(f"  {nm:22s}" + "".join(cells))


if __name__ == "__main__":
    rows = run()
    json.dump(rows, open(f"{OUT}/d3arch_T2.json", "w"), indent=1, default=float)
    report(rows)
