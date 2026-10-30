"""What makes a lesion survive? An interpretable account of the detectability map.

The map predicts erasure at AUROC 0.92, but a black box that says "do not trust here"
is a weaker deliverable than one that says why. Regressing retention on local image
descriptors turns the map into a statement a physicist or radiologist can check:

  local gradient magnitude   is the site near structure the model is already drawing?
  local variance             homogeneous interior against textured cortex
  distance to brain edge     peripheral sites behave differently from deep ones
  local intensity            tissue class, crudely
  64 mT local SNR proxy      how much measurement noise is at this site
  lesion radius / contrast   the stimulus itself

Fitted on the test lesions with a gradient-boosted tree, reported as permutation
importance plus the partial direction of each effect.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F_
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, models as Mo, train as T
from exp.d3qa import enhancer
from exp.d3detect import operator

OUT = "/mnt/G/Projects/ULF-MICCAI/out"


CACHE = "/mnt/G/Projects/ULF-MICCAI/datasets/npy"


def features(contrast="T2"):
    f = f"{OUT}/d3mech_feat_{contrast}_v2.npz"
    import os
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        return z["X"], z["Y"], z["G"], list(z["names"])
    X, Y, G, names = _features(contrast)
    np.savez(f, X=X, Y=Y, G=G, names=np.array(names, dtype=object))
    return X, Y, G, names


@torch.no_grad()
def _features(contrast="T2"):
    """Site-local descriptors. The eval JSON drops the coordinates, so the cached
    label file - which stores (row, cy, cx, r, c, retention) - is the source."""
    recs = [tuple(r) for r in np.load(f"{CACHE}/detect_test_{contrast}.npy", allow_pickle=True)]
    test = D.PairedCohort("test", contrast)
    enh = enhancer(); enh.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_{contrast}.pt")); enh.eval()
    fwd = operator(contrast, "learned")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    by_row = {}
    for row, cy, cx, r, c, ret in recs:
        by_row.setdefault(int(row), []).append((cy, cx, r, c, ret))
    det = Mo.UNet(6, 1, base=32, residual_from=None).cuda()
    det.load_state_dict(torch.load(f"{OUT}/d3detect_{contrast}.pt")); det.eval()
    names = ["radius", "|contrast|", "local gradient", "local variance",
             "depth from edge", "local intensity", "noise proxy", "model change",
             "map score"]
    X, Y, G = [], [], []
    sub_of = {}
    for sid in test.subjects:
        for r_ in test.subject_rows(sid):
            sub_of[int(r_)] = sid
    rows = sorted(by_row)
    for i in range(0, len(rows), 8):
        rr = rows[i:i + 8]
        y0 = torch.from_numpy(test.hf[np.array(rr)].astype(np.float32)).cuda()
        x0 = fwd(y0, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p0 = enh(x0).float().clamp(0, 1)
        c0 = x0[:, 1:2]
        gx = torch.zeros_like(c0); gy = torch.zeros_like(c0)
        gx[:, :, :, 1:] = (c0[:, :, :, 1:] - c0[:, :, :, :-1]).abs()
        gy[:, :, 1:] = (c0[:, :, 1:] - c0[:, :, :-1]).abs()
        maps = dict(grad=F_.avg_pool2d(gx + gy, 9, 1, 4))
        mu = F_.avg_pool2d(c0, 9, 1, 4)
        maps["var"] = (F_.avg_pool2d(c0 * c0, 9, 1, 4) - mu * mu).clamp(min=0)
        m = (y0 > 0).float()
        d = m.clone()
        for _ in range(6):                       # cheap distance-from-edge proxy
            d = d + (-F_.max_pool2d(-d, 5, 1, 2))
        maps["edge"] = d / 7.0
        maps["mu"] = mu
        maps["noise"] = F_.avg_pool2d((c0 - F_.avg_pool2d(c0, 5, 1, 2)).abs(), 9, 1, 4)
        maps["delta"] = F_.avg_pool2d((p0 - c0).abs(), 9, 1, 4)
        # the map is conditioned on the stimulus, so query it once per (r, c) setting
        qs = {}
        for (rr_, cc_) in sorted({(a[2], round(float(a[3]), 4))
                                  for row in rr for a in by_row[row]}):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qs[(rr_, cc_)] = det(torch.cat(
                    [x0, p0, torch.full_like(p0, rr_ / 6.0),
                     torch.full_like(p0, cc_ / 0.25)], 1)).float()
        for k, row in enumerate(rr):
            for (cy, cx, r, c, ret) in by_row[row]:
                g = lambda key: float(maps[key][k, 0, cy, cx])
                X.append([r, abs(c), g("grad"), g("var"), g("edge"),
                          g("mu"), g("noise"), g("delta"),
                          float(qs[(r, round(float(c), 4))][k, 0, cy, cx])])
                Y.append(ret); G.append(sub_of[row])
    return np.array(X), np.array(Y), np.array(G), names


def main(contrast="T2"):
    T.seed_all(0)
    X, Y, G, names = features(contrast)
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    # split by SUBJECT: sites from one slice are near-duplicates, and a random
    # split over sites would let the model memorise the slice instead of the rule.
    subs = sorted(set(G)); held = set(subs[-3:])
    tr = np.array([g not in held for g in G]); te = ~tr
    xa, ya, xb, yb = X[tr], Y[tr], X[te], Y[te]
    print(f"  held-out subjects: {', '.join(sorted(held))}  ({te.sum()} of {len(Y)} sites)")
    m = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(xa, ya)
    print(f"[{contrast}] retention ~ local descriptors   R^2 = {m.score(xb, yb):.3f}  (n={len(Y)})\n")
    pi = permutation_importance(m, xb, yb, n_repeats=10, random_state=0)
    order = np.argsort(-pi.importances_mean)
    print(f"  {'feature':20s} {'importance':>11s} {'corr with retention':>21s}")
    for i in order:
        r = np.corrcoef(X[:, i], Y)[0, 1]
        print(f"  {names[i]:20s} {pi.importances_mean[i]:11.4f} {r:+21.3f}")
    print(f"\n[{contrast}] WITHIN each radius: can hand-crafted descriptors stand in for the map?")
    print("  held-out R^2. descriptors = the 7 local statistics; map = the CNN score alone.\n")
    DESC = list(range(1, 8)); MAP = [8]
    print(f"  {'lesion':>8s} {'n':>6s} {'mean ret':>9s} {'descriptors':>12s} {'map':>8s} {'both':>8s}")
    strat = {}
    for r in sorted(set(X[:, 0])):
        sel = X[:, 0] == r
        if sel.sum() < 200:
            continue
        ys, gs = Y[sel], G[sel]
        ta = np.array([g not in held for g in gs]); tb = ~ta
        sc = {}
        for nm, cols in (("desc", DESC), ("map", MAP), ("both", DESC + MAP)):
            xs = X[sel][:, cols]
            ms = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(xs[ta], ys[ta])
            sc[nm] = float(ms.score(xs[tb], ys[tb]))
        print(f"  {int(r)*2:>5d} mm {int(sel.sum()):>6d} {ys.mean():>9.3f} "
              f"{sc['desc']:>12.3f} {sc['map']:>8.3f} {sc['both']:>8.3f}")
        strat[int(r)] = dict(n=int(sel.sum()), mean_ret=float(ys.mean()), **sc)
    json.dump({"stratified": strat, "r2": float(m.score(xb, yb)),
               "importance": {names[i]: float(pi.importances_mean[i]) for i in order},
               "corr": {names[i]: float(np.corrcoef(X[:, i], Y)[0, 1]) for i in order}},
              open(f"{OUT}/d3mech_{contrast}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "T2")
