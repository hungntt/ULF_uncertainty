"""Does cross-contrast synthesis invert REAL focal structure whose T1/T2 coupling is atypical?

The polarity results (d3polarity.py, d3lesiontype.py) use inserted lesions. This check
uses none. On the locked test subjects, every route receives the ACQUIRED 64 mT inputs,
and we look at real focal structure in the 3T reference: local extrema of a band-pass
(difference-of-Gaussians, lesion-scale) response of the 3T T1. Each feature is classed
by the sign of the 3T T2 response at the same place:

  inverse  - T1 and T2 responses have opposite signs (the population coupling)
  same     - both have the same sign (violates the coupling)

and we ask how often each route's synthesised 3T T1 response has the wrong sign. The
phantom result predicts: route A (T1 input only) is indifferent to the class; routes B
and C invert "same" features far more often than "inverse" ones. Features are matched
on |T1 response| across classes, so conspicuity cannot explain a gap.

Real anatomy is not a lesion, and "same"-sign features include vessels and partial-volume
structure. The check asks only whether the mechanism seen with phantoms also operates on
structure that was not inserted.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, train as T
from exp.d3qa import enhancer
from exp.d3pipe import d2_model, pack

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
C_MODELS = ("bothtoT1", "bothtoT1_ctl", "bothtoT1_aug", "bothtoT1_augw")


def gauss(x, s):
    k = int(3 * s) * 2 + 1
    t = torch.arange(k, device=x.device, dtype=torch.float32) - k // 2
    g = torch.exp(-t ** 2 / (2 * s * s)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))


def dog(x, s1=1.5, s2=4.0):
    return gauss(x, s1) - gauss(x, s2)


@torch.no_grad()
def run(q_feat=0.97, q_t2=0.60, min_sep=4):
    T.seed_all(0)
    mc = D.MultiContrastCohort("test")
    eA = enhancer(); eA.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_T1.pt")); eA.eval()
    nets = [("A_T1in", None), ("B_T2in", d2_model("T2toT1"))] + \
           [(f"C_{t}", d2_model(t)) for t in C_MODELS]
    feats = []
    for sid in sorted(set(mc.sid)):
        pos = np.where(mc.sid == sid)[0]
        pos = pos[len(pos) // 5: 4 * len(pos) // 5]
        for i0 in range(0, len(pos), 8):
            p = pos[i0:i0 + 8]
            sl = torch.from_numpy(mc.slab(p).astype(np.float32)).cuda()
            x1, x2 = sl[:, :3], sl[:, 3:]
            _, hf = mc.batch(p)
            y1 = torch.from_numpy(hf[:, 0:1].astype(np.float32)).cuda()
            y2 = torch.from_numpy(hf[:, 1:2].astype(np.float32)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs = {"A_T1in": eA(x1).float().clamp(0, 1)}
                for nm, e in nets[1:]:
                    outs[nm] = e(pack(x1 if nm.startswith("C") else None, x2)).float().clamp(0, 1)
            # interior of BOTH 3T brain masks, eroded so the brain edge is not a feature
            m = ((y1 > 0) & (y2 > 0)).float()
            m = (-F.max_pool2d(-m, 13, 1, 6)) > 0.5
            f1, f2 = dog(y1), dog(y2)
            fo = {k: dog(v) for k, v in outs.items()}
            a1 = f1.abs() * m
            if m.sum() < 100: continue
            thr = torch.quantile(f1.abs()[m], q_feat)
            t2thr = torch.quantile(f2.abs()[m], q_t2)
            peak = (a1 == F.max_pool2d(a1, 2 * min_sep + 1, 1, min_sep)) & (a1 > thr)
            for b, _, yy, xx in torch.nonzero(peak).tolist():
                v1, v2 = f1[b, 0, yy, xx].item(), f2[b, 0, yy, xx].item()
                if abs(v2) < t2thr: continue          # T2 silent: coupling undefined
                feats.append(dict(sid=sid, f1=v1, f2=v2,
                                  cls="same" if np.sign(v1) == np.sign(v2) else "inverse",
                                  **{k: fo[k][b, 0, yy, xx].item() for k in outs}))
        print(f"  {sid}: {len(feats)} features", flush=True)
    return feats


def summarise(feats, routes, n_bins=6, n_boot=10000):
    """Wrong-sign rate per class, matched on |T1 response| by stratified reweighting."""
    a = np.array([abs(f["f1"]) for f in feats])
    cls = np.array([f["cls"] for f in feats])
    sid = np.array([f["sid"] for f in feats])
    edges = np.quantile(a, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-9
    b = np.clip(np.digitize(a, edges) - 1, 0, n_bins - 1)
    # weight "inverse" features so their |f1| distribution matches the "same" class
    w = np.ones(len(feats))
    for k in range(n_bins):
        ns, ni = np.sum((b == k) & (cls == "same")), np.sum((b == k) & (cls == "inverse"))
        w[(b == k) & (cls == "inverse")] = ns / ni if ni else 0
    res = {"n_same": int(np.sum(cls == "same")), "n_inverse": int(np.sum(cls == "inverse"))}
    sids = sorted(set(sid))
    rng = np.random.default_rng(0)
    for r in routes:
        wrong = np.array([np.sign(f[r]) != np.sign(f["f1"]) for f in feats], float)
        def rate(sel_sids):
            out = {}
            for c in ("same", "inverse"):
                mk = np.isin(sid, sel_sids) & (cls == c)
                out[c] = np.sum(wrong[mk] * w[mk]) / max(np.sum(w[mk]), 1e-9)
            return out
        pt = rate(sids)
        bs = [rate(rng.choice(sids, len(sids))) for _ in range(n_boot)]
        res[r] = {c: dict(rate=pt[c], ci=[float(np.percentile([x[c] for x in bs], 2.5)),
                                          float(np.percentile([x[c] for x in bs], 97.5))])
                  for c in ("same", "inverse")}
        d = np.array([x["same"] - x["inverse"] for x in bs])
        res[r]["diff_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return res


if __name__ == "__main__":
    feats = run()
    json.dump(feats, open(f"{OUT}/d3realcoupling_feats.json", "w"), default=float)
    routes = ["A_T1in", "B_T2in"] + [f"C_{t}" for t in C_MODELS]
    res = summarise(feats, routes)
    json.dump(res, open(f"{OUT}/d3realcoupling.json", "w"), indent=1, default=float)
    print(f"\nreal focal features: {res['n_same']} same-sign, {res['n_inverse']} inverse "
          f"(inverse reweighted to the same |T1| distribution)\n")
    print(f"  {'route':18s}{'wrong sign: same':>26s}{'wrong sign: inverse':>26s}{'diff 95% CI':>20s}")
    for r in routes:
        s, i = res[r]["same"], res[r]["inverse"]
        print(f"  {r:18s}{s['rate']*100:12.1f}% [{s['ci'][0]*100:4.1f},{s['ci'][1]*100:4.1f}]"
              f"{i['rate']*100:12.1f}% [{i['ci'][0]*100:4.1f},{i['ci'][1]*100:4.1f}]"
              f"   [{res[r]['diff_ci'][0]*100:+.1f},{res[r]['diff_ci'][1]*100:+.1f}]")
