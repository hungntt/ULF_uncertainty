"""Subject-swap counterfactual: does this voxel depend on *whose* scan went in?

The input-perturbation probe failed because a small local perturbation mostly
measures the image gradient, so it behaves like an edge detector. The question
Finding A actually poses is different: the ULF->3T correction is subject-specific
(cross-subject residual cosine 0.09), so a trustworthy voxel must change when the
subject changes. Feeding a *different* subject's 64 mT scan at the same axial
position is the direct probe, and between-subject anatomy differs wherever there
is real structure, so it is not edge-confounded.

    u_swap(v) = mean_B | yhat(x_A)(v) - yhat(x_B)(v) |,  B != A, matched z

Low u_swap = the model renders the same anatomy regardless of the measurement.
"""
import argparse, json, sys, zlib
import numpy as np
import torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, train as T
import exp.d3v2 as d3
from exp.d3v2_analyze import load, auroc

OUT = "/mnt/G/Projects/ULF-MICCAI/out"


@torch.no_grad()
def predict_all(net, sch, cohort, sids, K, ns, bs=16):
    """Posterior mean, dispersion and ground truth per subject, kept as full slices
    so that slices from different subjects can be compared voxel-for-voxel."""
    store = {}
    for sid in sids:
        idx = cohort.subject_rows(sid)
        P, U, Y, X, Z = [], [], [], [], []
        n = len(idx)
        for i in range(0, n, bs):
            j = idx[i:i + bs]
            x = torch.from_numpy(cohort.ulf[j].astype(np.float32)).cuda()
            y = torch.from_numpy(cohort.hf[j].astype(np.float32)).cuda()
            dr = d3.sample(net, sch, x, K=K, n_steps=16, seed=0, ns=ns)
            P.append(dr.mean(0).cpu()); U.append(dr.std(0).cpu())
            Y.append(y.cpu()); X.append(x.cpu())
            Z.append(torch.from_numpy((np.arange(i, i + len(j)) / n).astype(np.float32)))
        store[sid] = dict(p=torch.cat(P), u=torch.cat(U), y=torch.cat(Y),
                          x=torch.cat(X), z=torch.cat(Z))
        print(f"  {sid}: {len(store[sid]['p'])} slices", flush=True)
    return store


@torch.no_grad()
def swap_channel(net, sch, store, sids, K, ns, n_other=3, bs=8):
    """For each slice of A, run A's own model output against the outputs produced from
    other subjects' 64 mT scans at the matched relative depth."""
    rng = np.random.default_rng(0)
    for sid in sids:
        s = store[sid]
        acc = torch.zeros_like(s["p"])
        others = [o for o in sids if o != sid]
        for rep in range(n_other):
            o = others[rng.integers(len(others))]
            so = store[o]
            # match on relative depth
            k = torch.cdist(s["z"][:, None], so["z"][:, None]).argmin(1)
            xo = so["x"][k]
            po = []
            for i in range(0, len(xo), bs):
                dr = d3.sample(net, sch, xo[i:i + bs].cuda(), K=max(4, K // 8),
                               n_steps=16, seed=0, ns=ns)
                po.append(dr.mean(0).cpu())
            acc += (s["p"] - torch.cat(po)).abs()
        s["swap"] = acc / n_other
        print(f"  swap {sid}: mean {s['swap'].mean():.5f}", flush=True)


def harvest(store, sids, budget=300_000):
    out = {}
    for sid in sids:
        s = store[sid]
        m = ((s["x"][:, 1:2] > 0) | (s["y"] > 0))
        arrs = [s[k][m].numpy().astype(np.float32) for k in ("p", "u", "swap", "y")]
        if len(arrs[0]) > budget:
            k = np.sort(np.random.default_rng(zlib.crc32(sid.encode())).choice(
                len(arrs[0]), budget, replace=False))
            arrs = [a[k] for a in arrs]
        out[sid] = tuple(arrs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--tag", default="")
    ap.add_argument("--K", type=int, default=16)
    a = ap.parse_args()
    T.seed_all(0)
    net, sch, ns = load(a.contrast, a.tag)
    test = D.PairedCohort("test", a.contrast)
    sids = test.subjects
    print(f"[{a.contrast}{a.tag}] predicting {len(sids)} test subjects")
    store = predict_all(net, sch, test, sids, a.K, ns)
    print("computing subject-swap counterfactual")
    swap_channel(net, sch, store, sids, a.K, ns)
    te = harvest(store, sids)

    rows = []
    for sid, (p, u, sw, y) in te.items():
        e = np.abs(p - y)
        hi = (e > np.quantile(e, 0.9)).astype(np.int8)
        lo = u <= np.median(u)
        smed = np.median(sw[lo])
        rows.append(dict(sid=sid, auroc_disp=auroc(u, hi), auroc_swap=auroc(-sw, hi),
                         err_lodisp_loswap=float(e[lo & (sw <= smed)].mean()),
                         err_lodisp_hiswap=float(e[lo & (sw > smed)].mean()),
                         ratio=float(e[lo & (sw <= smed)].mean() / max(e[lo & (sw > smed)].mean(), 1e-9)),
                         swap_mean=float(sw.mean())))
    g = M.aggregate(rows)
    print(f"\n== subject-swap counterfactual, {a.contrast}{a.tag} ==")
    print(f"  mean |yhat(A) - yhat(B)| = {g['swap_mean'][0]:.5f}  "
          f"(a model ignoring its input would give ~0)")
    print(f"  AUROC, dispersion            {g['auroc_disp'][0]:.3f}")
    print(f"  AUROC, LOW swap = risky      {g['auroc_swap'][0]:.3f}")
    print(f"\n  inside the LOW-dispersion half:")
    print(f"    mean |err|, low swap (prior-driven)  {g['err_lodisp_loswap'][0]:.5f}")
    print(f"    mean |err|, high swap (input-driven) {g['err_lodisp_hiswap'][0]:.5f}")
    print(f"    ratio {g['ratio'][0]:.3f}±{g['ratio'][1]:.3f}  "
          f"(>1 confirms prior-driven voxels are worse at equal dispersion)")
    json.dump({"rows": rows, "agg": g}, open(f"{OUT}/d3v2_swap_{a.contrast}{a.tag}.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
