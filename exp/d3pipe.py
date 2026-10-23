"""Which input route destroys fewer lesions in the synthesised 3T T1?

Direction 2 found the 64 mT T1 acquisition near-redundant: 64 mT T2 predicts 3T T1
better than 64 mT T1 does. The lesion study then found the T1-from-T1 route erases 77%
of lesions against 35% on T2. Those two facts together suggest the route matters more
than the model, which is a protocol question, not a modelling one.

A lesion is a physical object that appears in both contrasts, so it is inserted into
the 3T T1 and the 3T T2 of the same subject at the same place. Default polarity is the
clinically common one - T2-hyperintense, T1-hypointense at half the magnitude, as for
white-matter disease, oedema and infarct - so the T1 route sees the darker, harder
lesion and the T2 route the brighter one. Retention is always measured against the
3T T1 lesion, because that is the output a reader looks at.

  A  64 mT T1          -> 3T T1   (base_unet_noresid_T1)
  B  64 mT T2          -> 3T T1   (d2_T2toT1)
  C  64 mT T1 and T2   -> 3T T1   (d2_bothtoT1)
"""
import argparse, json, sys
import numpy as np
import torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, models as Mo, train as T
from ulfkit.metrics import paired_test
from exp.d3qa import enhancer
from exp.d3detect import operator, site_pool, blobs

OUT = "/mnt/G/Projects/ULF-MICCAI/out"


def d2_model(tag):
    net = Mo.UNet(8, 1, base=32, residual_from=[1]).cuda()
    net.load_state_dict(torch.load(f"{OUT}/d2_{tag}.pt")); net.eval()
    return net


def pack(x1=None, x2=None):
    """8-channel Direction-2 layout: [T1 slab | T2 slab | availT1 | availT2]."""
    ref = x1 if x1 is not None else x2
    z = torch.zeros_like(ref)
    o = torch.ones_like(ref[:, :1])
    a1 = o if x1 is not None else o * 0
    a2 = o if x2 is not None else o * 0
    return torch.cat([x1 if x1 is not None else z,
                      x2 if x2 is not None else z, a1, a2], 1)


@torch.no_grad()
def run(contrast_pair=("T1", "T2"), c1_frac=-0.5, n_sites=5, radii=(2, 3, 4, 6),
        cs=(0.12, 0.25), seed=5, cmodel="bothtoT1"):
    T.seed_all(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    mc = D.MultiContrastCohort("test")
    f1, f2 = operator("T1", "learned"), operator("T2", "learned")
    eA = enhancer(); eA.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_T1.pt")); eA.eval()
    eB, eC = d2_model("T2toT1"), d2_model(cmodel)

    rows = []
    for sid in sorted(set(mc.sid)):
        pos = np.where(mc.sid == sid)[0]
        sel = pos[np.linspace(len(pos)//5, 4*len(pos)//5, 8).astype(int)]
        for i0 in range(0, len(sel), 4):
            p = sel[i0:i0+4]
            _, hf = mc.batch(p)
            y1 = torch.from_numpy(hf[:, 0:1].astype(np.float32)).cuda()
            y2 = torch.from_numpy(hf[:, 1:2].astype(np.float32)).cuda()
            ok = site_pool(y1)
            # lesion-free controls for each route
            x1_0, x2_0 = f1(y1, gen), f2(y2, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pA0 = eA(x1_0).float().clamp(0, 1)
                pB0 = eB(pack(x2=x2_0)).float().clamp(0, 1)
                pC0 = eC(pack(x1_0, x2_0)).float().clamp(0, 1)
            jobs = []
            for s in range(len(p)):
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
                c2 = torch.tensor([t[4] for t in ch], device="cuda")[:, None, None, None]
                c1 = c2 * c1_frac
                b = blobs(cy, cx, rr) * (y1[si] > 0).float()
                y1l = (y1[si] + c1 * b).clamp(0, 1)
                y2l = (y2[si] + c2 * b).clamp(0, 1)
                d1 = y1l - y1[si]
                core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
                good = torch.tensor([bool(core[k].sum() >= 4 and ring[k].sum() >= 8)
                                     for k in range(len(ch))], device="cuda")
                tru = torch.stack([(d1[k][core[k]].mean() - d1[k][ring[k]].mean())
                                   if good[k] else torch.tensor(0., device="cuda")
                                   for k in range(len(ch))])
                x1l, x2l = f1(y1l, gen), f2(y2l, gen)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pA = eA(x1l).float().clamp(0, 1)
                    pB = eB(pack(x2=x2l)).float().clamp(0, 1)
                    pC = eC(pack(x1l, x2l)).float().clamp(0, 1)
                for nm, pl, p0 in (("A_T1in", pA, pA0), ("B_T2in", pB, pB0), ("C_both", pC, pC0)):
                    dp = pl - p0[si]
                    rec = torch.stack([(dp[k][core[k]].mean() - dp[k][ring[k]].mean())
                                       if good[k] else torch.tensor(0., device="cuda")
                                       for k in range(len(ch))])
                    for k in range(len(ch)):
                        if not good[k] or tru[k].abs() < 0.25 * c1[k].abs():
                            continue
                        rows.append(dict(route=nm, sid=sid, r=ch[k][3], c=ch[k][4],
                                         retention=float((rec[k] / tru[k]).clamp(-0.5, 1.5))))
        print(f"  {sid}: {len(rows)} measurements", flush=True)
    return rows


def report(rows, thr=0.25):
    routes = [("A_T1in", "A  64 mT T1 -> 3T T1"),
              ("B_T2in", "B  64 mT T2 -> 3T T1"),
              ("C_both", "C  both      -> 3T T1")]
    per = {}
    sids = sorted({x["sid"] for x in rows})
    print(f"\n== lesion survival in the synthesised 3T T1, by input route ==")
    print(f"  {'route':24s} {'mean retention':>15s} {'% erased':>10s}")
    for key, nm in routes:
        R = [x for x in rows if x["route"] == key]
        ret = np.array([np.mean([x["retention"] for x in R if x["sid"] == s]) for s in sids])
        er = np.array([np.mean([x["retention"] < thr for x in R if x["sid"] == s]) for s in sids])
        per[key] = (ret, er)
        print(f"  {nm:24s} {ret.mean():10.3f} ±{ret.std(ddof=1):.3f} {er.mean()*100:9.1f}%")
    print()
    for a, b in (("B_T2in", "A_T1in"), ("C_both", "A_T1in"), ("C_both", "B_T2in")):
        dr = per[a][0].mean() - per[b][0].mean()
        de = per[a][1].mean() - per[b][1].mean()
        print(f"  {a} vs {b}: retention {dr:+.3f} (p={paired_test(per[a][0], per[b][0]):.4f})   "
              f"erased {de*100:+.1f} pts (p={paired_test(per[a][1], per[b][1]):.4f})")
    print(f"\n  by lesion size (% erased):")
    print(f"  {'route':24s}" + "".join(f"{2*r:>8d} mm" for r in (2, 3, 4, 6)))
    for key, nm in routes:
        R = [x for x in rows if x["route"] == key]
        cells = []
        for r in (2, 3, 4, 6):
            v = [x["retention"] < thr for x in R if x["r"] == r]
            cells.append(f"{np.mean(v)*100:8.0f}%" if v else "       -")
        print(f"  {nm:24s}" + "".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1_frac", type=float, default=-0.5,
                    help="T1 lesion contrast as a fraction of the T2 one; negative = hypointense")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rows = run(c1_frac=a.c1_frac)
    json.dump(rows, open(f"{OUT}/d3pipe{a.tag}.json", "w"), indent=1, default=float)
    report(rows)
