"""Does the phantom result hold for REAL focal structure?

Every retention number so far comes from a Gaussian blob inserted into healthy-looking
anatomy. The obvious objection is that a real lesion is not a Gaussian blob.

The screen in d3real.py found 13 focal white-matter hyperintensities in 4 of the 50
subjects. They are NOT confirmed pathology - no radiologist has seen them, and several
sit close enough to a ventricle that partial volume is a live alternative. What they
certainly are is real focal structure with real texture, real edges and real contrast,
which is what the phantom is standing in for.

Retention needs a lesion-free counterfactual. For an inserted blob that is free: the
image before insertion. For a real feature it has to be constructed, by inpainting the
feature out with its surrounding white matter. The estimator is otherwise identical to
the phantom one, so the two numbers are comparable.

The inpaint defines the counterfactual, so it bounds what this can show - a measurement
of how much of the feature survives, only as good as the fill.
"""
import json, sys
import numpy as np
import scipy.ndimage as ndi
import torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, models as Mo, train as T
from exp.d3qa import enhancer
from exp.d3detect import operator, blobs
from exp.d3real import screen_slice, Z, ZA, MIN_A, MAX_A, EDGE, SCREEN_TAG

OUT = "/mnt/G/Projects/ULF-MICCAI/out"


def feature_mask(y, cy, cx):
    """Re-derive the exact cluster the screen found at this site."""
    m = y > 0
    bg = ndi.median_filter(y, size=15)
    r = (y - bg) * m
    s = 1.4826 * np.median(np.abs(r[m] - np.median(r[m])))
    z = r / max(s, 1e-6)
    cxm = float(np.nonzero(m)[1].mean()); xs = np.arange(y.shape[1])
    src = np.clip(np.rint(2 * cxm - xs).astype(int), 0, y.shape[1] - 1)
    za = (y - y[:, src]) / max(s, 1e-6)
    hi = np.quantile(y[m], 0.90)
    csf = ndi.binary_dilation(y > hi, np.ones((9, 9)))
    wm = (bg > np.quantile(y[m], 0.25)) & (bg < np.quantile(y[m], 0.70))
    deep = ndi.binary_erosion(m, np.ones((2 * EDGE + 1, 2 * EDGE + 1)))
    lab, n = ndi.label((z > Z) & (za > ZA) & deep & m[:, src] & wm & ~csf)
    k = lab[cy, cx]
    if k == 0:                      # centroid can fall just outside its own cluster
        sub = lab[max(0, cy-3):cy+4, max(0, cx-3):cx+4]
        v = sub[sub > 0]
        if not len(v):
            return None, bg
        k = np.bincount(v).argmax()
    return lab == k, bg


import os as _os
# The feather is not cosmetic: it spreads the real-minus-inpainted difference beyond the
# feature and so inflates any fitted extent. Exposed so the headline can be quoted at the
# conservative, feather-free setting.
DIL = int(_os.environ.get("REAL_DIL", 3))
SIG = float(_os.environ.get("REAL_SIG", 1.5))


def inpaint(y, fm, bg):
    """Replace the feature with its surrounding white matter."""
    m = ndi.binary_dilation(fm, np.ones((DIL, DIL))) if DIL > 1 else fm
    soft = ndi.gaussian_filter(m.astype(np.float32), SIG) if SIG > 0 else m.astype(np.float32)
    soft = np.clip(soft / max(soft.max(), 1e-6), 0, 1)
    return (y * (1 - soft) + bg * soft).astype(np.float32)


def fold_for(sid):
    """The cross-fit model that never saw this subject (None -> any model is held out)."""
    _, tr, _ = D.dev_split("T2")
    tr = sorted(tr)
    return tr.index(sid) % 4 if sid in tr else None


@torch.no_grad()
def main():
    T.seed_all(0)
    found = json.load(open(f"{OUT}/d3real_screen{SCREEN_TAG}.json"))
    nets = {}
    for f in range(4):
        n = enhancer(); n.load_state_dict(torch.load(f"{OUT}/d3qa_fold{f}_T2.pt")); n.eval()
        nets[f] = n
    det = Mo.UNet(6, 1, base=32, residual_from=None).cuda()
    det.load_state_dict(torch.load(f"{OUT}/d3detect_T2.pt")); det.eval()
    fwd = operator("T2", "learned")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)

    rows = []
    print("  real feature vs a Gaussian phantom of MATCHED size and contrast,")
    print("  same held-out enhancer, same site, same lesion-free baseline\n")
    print(f"  {'subject':11s} {'row':>5s} {'r_eq':>5s} {'contrast':>9s} "
          f"{'real':>7s} {'phantom':>8s} {'map':>7s}")
    for sid, d in found.items():
        coh = D.PairedCohort(d["split"], "T2")
        f = fold_for(sid)
        enh = nets[f if f is not None else 0]
        for h in d["top"]:
            y = coh.hf[h["row"], 0].astype(np.float32)
            fm, bg = feature_mask(y, h["cy"], h["cx"])
            if fm is None or fm.sum() < MIN_A:
                continue
            y0 = inpaint(y, fm, bg)
            yt = torch.from_numpy(y).cuda()[None, None]
            y0t = torch.from_numpy(y0).cuda()[None, None]
            r_eq = max(2.0, float(np.sqrt(fm.sum() / np.pi)))
            b = blobs(torch.tensor([h["cy"]], device="cuda"),
                      torch.tensor([h["cx"]], device="cuda"),
                      torch.tensor([r_eq], device="cuda")) * (y0t > 0).float()
            core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
            dy = yt - y0t
            tru = float(dy[core].mean() - dy[ring].mean())
            if abs(tru) < 0.02:
                continue
            # A Gaussian matched to the THRESHOLDED core is not matched to the feature:
            # real focal structure has a flat core and long soft shoulders, so its
            # effective extent is well beyond the suprathreshold radius. Fit the
            # effective radius to this feature's own radial profile and build a third
            # arm from it, to see whether profile shape explains the whole gap.
            dmap = np.sqrt((np.mgrid[0:256, 0:256][0] - h["cy"]) ** 2
                           + (np.mgrid[0:256, 0:256][1] - h["cx"]) ** 2)
            dres = (y - y0)
            pk = dres[dmap <= 1.5].mean()
            r_fit = r_eq
            if pk > 0.02:
                cand = np.arange(1.5, 9.01, 0.25)
                sel = dmap <= 9
                err = [np.mean((dres[sel] / pk - np.exp(-2 * dmap[sel] ** 2 / c ** 2)) ** 2)
                       for c in cand]
                r_fit = float(cand[int(np.argmin(err))])
            bf = blobs(torch.tensor([h["cy"]], device="cuda"),
                       torch.tensor([h["cx"]], device="cuda"),
                       torch.tensor([r_fit], device="cuda")) * (y0t > 0).float()
            corf, rinf = bf > 0.6, (bf > 0.05) & (bf < 0.25)

            # the inpainted slice is the common lesion-free baseline for all arms
            yph = (y0t + tru * b).clamp(0, 1)
            dyp = yph - y0t
            trup = float(dyp[core].mean() - dyp[ring].mean())
            trf = float((yt - y0t)[corf].mean() - (yt - y0t)[rinf].mean())
            yfit = (y0t + trf * bf).clamp(0, 1)
            dyf = yfit - y0t
            truf = float(dyf[corf].mean() - dyf[rinf].mean())
            xw, xo, xp = fwd(yt, gen), fwd(y0t, gen), fwd(yph, gen)
            xf = fwd(yfit, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pw = enh(xw).float().clamp(0, 1)
                po = enh(xo).float().clamp(0, 1)
                pp = enh(xp).float().clamp(0, 1)
                pf = enh(xf).float().clamp(0, 1)
            rr_ = lambda p: float((p - po)[core].mean() - (p - po)[ring].mean())
            rrf = lambda p: float((p - po)[corf].mean() - (p - po)[rinf].mean())
            ret = float(np.clip(rr_(pw) / tru, -0.5, 1.5))
            retp = float(np.clip(rr_(pp) / trup, -0.5, 1.5)) if abs(trup) > 1e-6 else float("nan")
            # the profile-matched arm is scored on ITS OWN core/ring, and the real
            # feature is re-scored the same way so the pair is comparable
            retf = float(np.clip(rrf(pf) / truf, -0.5, 1.5)) if abs(truf) > 1e-6 else float("nan")
            retw_f = float(np.clip(rrf(pw) / trf, -0.5, 1.5)) if abs(trf) > 0.01 else float("nan")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = det(torch.cat([xw, pw, torch.full_like(pw, r_eq / 6.0),
                                   torch.full_like(pw, tru / 0.25)], 1)).float()
            qv = float(q[0, 0, h["cy"], h["cx"]])
            # ask the map about the size the feature actually behaves as, not the
            # size its suprathreshold core measures
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q2 = det(torch.cat([xw, pw, torch.full_like(pw, r_fit / 6.0),
                                    torch.full_like(pw, trf / 0.25)], 1)).float()
            qv2 = float(q2[0, 0, h["cy"], h["cx"]])
            rows.append(dict(sid=sid, split=d["split"], row=h["row"], cy=h["cy"], cx=h["cx"], area=int(fm.sum()),
                             r_eq=r_eq, r_fit=r_fit, contrast=tru, retention=ret,
                             phantom=retp, prof=retf, real_at_fit=retw_f,
                             map=qv, map_fit=qv2, fold=f))
            print(f"  {sid:11s} {h['row']:5d} {r_eq:5.1f} {tru:9.3f} "
                  f"{ret:7.3f} {retp:8.3f} {qv:7.3f}")
    if not rows:
        print("\n  no measurable sites"); return
    objs = []
    for r in sorted(rows, key=lambda r: (r["sid"], r["row"])):
        for o in objs:
            if (o["sid"] == r["sid"] and abs(o["row"] - r["row"]) <= 2
                    and abs(o["cy"] - r["cy"]) <= 5 and abs(o["cx"] - r["cx"]) <= 5):
                o["R"].append(r["retention"]); o["P"].append(r["phantom"])
                o["row"] = r["row"]; break
        else:
            objs.append(dict(sid=r["sid"], split=r["split"], row=r["row"],
                             cy=r["cy"], cx=r["cx"], R=[r["retention"]], P=[r["phantom"]]))
    oR = np.array([np.mean(o["R"]) for o in objs])
    oP = np.array([np.nanmean(o["P"]) for o in objs])
    R = np.array([r["retention"] for r in rows]); P = np.array([r["phantom"] for r in rows])
    ok = ~np.isnan(P)
    print(f"\n  {len(R)} real focal features, {np.mean([r['r_eq'] for r in rows])*2:.1f} mm mean")
    print(f"    real feature    retention {R.mean():.3f} +-{R.std(ddof=1):.3f}   "
          f"{100*np.mean(R < 0.25):.0f}% erased")
    print(f"    matched phantom retention {P[ok].mean():.3f} +-{P[ok].std(ddof=1):.3f}   "
          f"{100*np.mean(P[ok] < 0.25):.0f}% erased")
    dd = R[ok] - P[ok]
    from scipy.stats import wilcoxon
    pv = wilcoxon(R[ok], P[ok]).pvalue if ok.sum() > 5 else float("nan")
    print(f"    paired difference {dd.mean():+.3f} (real - phantom), "
          f"{np.mean(dd > 0)*100:.0f}% of sites higher, wilcoxon p={pv:.4f}")
    F = np.array([r["prof"] for r in rows]); W = np.array([r["real_at_fit"] for r in rows])
    rf = np.array([r["r_fit"] for r in rows]); re_ = np.array([r["r_eq"] for r in rows])
    fk = ~np.isnan(F) & ~np.isnan(W)
    print(f"\n  effective radius fitted to each feature's own profile: "
          f"{rf.mean():.2f} px vs thresholded {re_.mean():.2f} px "
          f"(x{rf.mean()/max(re_.mean(),1e-6):.2f})")
    print(f"    PROFILE-matched phantom {F[fk].mean():.3f}   "
          f"real feature scored the same way {W[fk].mean():.3f}   "
          f"gap {W[fk].mean()-F[fk].mean():+.3f} (was {dd.mean():+.3f} against the core-matched one)")
    ook = ~np.isnan(oP)
    print(f"\n  grouped into {len(objs)} distinct 3D features in "
          f"{len({o['sid'] for o in objs})} subjects (adjacent slices are one object):")
    print(f"    real    {oR[ook].mean():.3f}   phantom {oP[ook].mean():.3f}   "
          f"higher at {int((oR[ook] > oP[ook]).sum())}/{int(ook.sum())}")
    if ook.sum() > 5:
        print(f"    wilcoxon over objects p={wilcoxon(oR[ook], oP[ook]).pvalue:.4f}")
    else:
        print(f"    too few objects for a rank test; sign test "
              f"p={2.0 ** -int(ook.sum()):.4f} if all one way")
    te = [r for r in rows if r["split"] == "test"]
    if te:
        print(f"\n  locked-test subjects only ({len(te)} sites): real "
              f"{np.mean([r['retention'] for r in te]):.3f}  "
              f"phantom {np.mean([r['phantom'] for r in te]):.3f}")
    json.dump(rows, open(f"{OUT}/d3realret{SCREEN_TAG}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
