"""Score the QA map against every unsupervised alternative, and attach a
false-negative guarantee.

The question a reader actually asks is not "how uncertain is this voxel" but
"which parts of this image should I not rely on". So the headline metric is:

    at a guaranteed miss rate alpha, what fraction of the brain must be flagged?

Conformal risk control gives the guarantee. Flag every voxel whose QA score exceeds
lambda; the risk is the fraction of *unflagged* brain voxels whose error exceeds tau.
That risk falls as lambda falls, so the CRC bound (n*Rbar + B)/(n+1) <= alpha picks
the largest lambda that is still safe, and the flagged fraction is the price.
"""
import argparse, json, os, sys, zlib
import numpy as np
import torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T
from exp.d3qa import enhancer

OUT = "/mnt/G/Projects/ULF-MICCAI/out"


@torch.no_grad()
def gather(contrast, sids, cohort, enh, qa, folds=None, budget=300_000):
    """Per subject: error, QA score, and the unsupervised baselines."""
    out = {}
    for sid in sids:
        idx = cohort.subject_rows(sid)
        E, Q, R, G, N = [], [], [], [], []
        for i in range(0, len(idx), 32):
            j = idx[i:i + 32]
            x = torch.from_numpy(cohort.ulf[j].astype(np.float32)).cuda()
            y = torch.from_numpy(cohort.hf[j].astype(np.float32)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = enh(x).float().clamp(0, 1)
                q = qa(torch.cat([x, p], 1)).float()
            m = ((x[:, 1:2] > 0) | (y > 0))
            E.append((p - y).abs()[m]); Q.append(q[m])
            R.append((p - x[:, 1:2]).abs()[m])          # how much the model changed it
            gx = torch.zeros_like(x[:, 1:2]); gy = torch.zeros_like(gx)
            gx[:, :, :, 1:] = (x[:, 1:2, :, 1:] - x[:, 1:2, :, :-1]).abs()
            gy[:, :, 1:] = (x[:, 1:2, 1:] - x[:, 1:2, :-1]).abs()
            G.append((gx + gy)[m])                      # 64 mT edge magnitude
            if folds:                                   # deep-ensemble disagreement
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ps = torch.stack([f(x).float().clamp(0, 1) for f in folds])
                N.append(ps.std(0)[m])
        a = [torch.cat(v).cpu().numpy().astype(np.float32) for v in (E, Q, R, G)]
        if folds: a.append(torch.cat(N).cpu().numpy().astype(np.float32))
        if len(a[0]) > budget:
            k = np.sort(np.random.default_rng(zlib.crc32(sid.encode())).choice(
                len(a[0]), budget, replace=False))
            a = [v[k] for v in a]
        d = dict(err=a[0], qa=a[1], resid=a[2], edge=a[3])
        if folds: d["ens"] = a[4]
        out[sid] = d
    return out


def ranking(data, tau):
    from sklearn.metrics import roc_auc_score, average_precision_score
    rows = []
    for sid, d in data.items():
        lab = (d["err"] > tau).astype(np.int8)
        r = {"sid": sid, "bad_frac": float(lab.mean())}
        if 0 < lab.sum() < len(lab):
            for k in [c for c in ("qa", "ens", "resid", "edge") if c in d]:
                r[f"auroc_{k}"] = float(roc_auc_score(lab, d[k]))
                r[f"ap_{k}"] = float(average_precision_score(lab, d[k]))
        rows.append(r)
    return rows


def crc_flag(cal, te, key, tau, alpha):
    """Largest lambda whose CRC bound on the unflagged-miss rate is <= alpha.

    The risk is bounded by 1, so the bound (n*Rbar + 1)/(n+1) <= alpha is unreachable
    unless alpha > 1/(n+1). With 8 calibration subjects that floor is 0.111, and any
    alpha at or below it silently drives the threshold to its minimum and flags the
    whole brain. Return the floor rather than a degenerate answer."""
    n = len(cal)
    floor = 1.0 / (n + 1)
    if alpha <= floor:
        return None, None, floor
    allq = np.concatenate([d[key] for d in cal.values()])
    grid = np.quantile(allq, np.linspace(0.999, 0.0, 200))     # high -> low threshold
    chosen = grid[-1]
    for lam in grid:
        R = [float(((d["err"] > tau) & (d[key] <= lam)).sum() / max((d[key] <= lam).sum(), 1))
             for d in cal.values()]
        if (n * np.mean(R) + 1) / (n + 1) <= alpha:
            chosen = lam; break
    rows = []
    for sid, d in te.items():
        un = d[key] <= chosen
        rows.append(dict(sid=sid,
                         miss_rate=float(((d["err"] > tau) & un).sum() / max(un.sum(), 1)),
                         flagged_frac=float((~un).mean()),
                         caught=float(((d["err"] > tau) & ~un).sum() / max((d["err"] > tau).sum(), 1))))
    return float(chosen), rows, floor


def tradeoff(cal, te, key, tau, fracs=(0.05, 0.10, 0.20, 0.30, 0.50)):
    """Threshold-free view: pick the threshold on calibration to flag a target
    fraction, then measure what it costs and catches on the locked test subjects."""
    allq = np.concatenate([d[key] for d in cal.values()])
    rows = []
    for f in fracs:
        lam = float(np.quantile(allq, 1 - f))
        per = []
        for d in te.values():
            un = d[key] <= lam; bad = d["err"] > tau
            per.append((float((~un).mean()),
                        float((bad & un).sum() / max(un.sum(), 1)),
                        float((bad & ~un).sum() / max(bad.sum(), 1))))
        a = np.array(per)
        rows.append(dict(target=f, flagged=float(a[:, 0].mean()),
                         miss=float(a[:, 1].mean()), catch=float(a[:, 2].mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--tau", type=float, default=None, help="error threshold; default = p90 of calibration error")
    a = ap.parse_args()
    T.seed_all(0)
    coh, tr, val = D.dev_split(a.contrast)
    test = D.PairedCohort("test", a.contrast)
    enh = enhancer(); enh.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_{a.contrast}.pt")); enh.eval()
    qa = Mo.UNet(4, 1, base=32, residual_from=None).cuda()
    qa.load_state_dict(torch.load(f"{OUT}/d3qa_{a.contrast}.pt")); qa.eval()

    folds = []
    for f in range(4):
        fp = f"{OUT}/d3qa_fold{f}_{a.contrast}.pt"
        if os.path.exists(fp):
            n = enhancer(); n.load_state_dict(torch.load(fp)); n.eval(); folds.append(n)
    print(f"  deep-ensemble baseline: {len(folds)} fold models")
    cal = gather(a.contrast, val, coh, enh, qa, folds)
    te = gather(a.contrast, test.subjects, test, enh, qa, folds)
    tau = a.tau or float(np.quantile(np.concatenate([d["err"] for d in cal.values()]), 0.90))
    res = {"contrast": a.contrast, "alpha": a.alpha, "tau": tau}

    rk = ranking(te, tau); res["ranking"] = {"rows": rk, "agg": M.aggregate(rk)}
    g = res["ranking"]["agg"]
    print(f"[{a.contrast}] error threshold tau = {tau:.4f} (p90 of calibration error); "
          f"{g['bad_frac'][0]*100:.1f}% of test voxels exceed it\n")
    print(f"== how well does each map find the badly synthesised voxels? ==")
    for k, nm in [("qa", "learned QA map (ours)"), ("ens", "deep ensemble (4 models)"),
                  ("resid", "|change from 64 mT|"), ("edge", "64 mT edge magnitude")]:
        if f"auroc_{k}" not in g: continue
        print(f"  {nm:24s} AUROC {g['auroc_'+k][0]:.3f}±{g['auroc_'+k][1]:.3f}  "
              f"AP {g['ap_'+k][0]:.3f}")

    names = [("qa", "learned QA map (ours)"), ("ens", "deep ensemble (4 models)"),
             ("resid", "|change from 64 mT|"), ("edge", "64 mT edge magnitude")]
    avail = [(k, nm) for k, nm in names if k in list(te.values())[0]]

    print(f"\n== flag/catch trade-off (threshold set on calibration, measured on test) ==")
    print(f"  {'score':24s}" + "".join(f"{int(f*100):>7d}%" for f in (5, 10, 20, 30, 50)))
    res["tradeoff"] = {}
    for k, nm in avail:
        rows = tradeoff(cal, te, k, tau); res["tradeoff"][k] = rows
        print(f"  {nm:24s}" + "".join(f"{r['catch']*100:7.1f}" for r in rows)
              + "   <- % of bad voxels caught")
    print(f"  {'(flagged fraction)':24s}" + "".join(
        f"{res['tradeoff'][avail[0][0]][i]['flagged']*100:7.1f}" for i in range(5)))

    print(f"\n== conformal risk control, guaranteed miss rate ==")
    res["crc"] = {}
    for alpha in sorted({a.alpha, 0.15, 0.20}):
        for k, nm in avail:
            lam, rows, floor = crc_flag(cal, te, k, tau, alpha)
            if lam is None:
                if k == avail[0][0]:
                    print(f"  alpha={alpha}: INFEASIBLE - {len(cal)} calibration subjects "
                          f"put the floor at {floor:.3f}")
                continue
            ag = M.aggregate(rows)
            res["crc"][f"{alpha}_{k}"] = {"lambda": lam, "agg": ag, "rows": rows}
            print(f"  alpha={alpha} {nm:24s} flag {ag['flagged_frac'][0]*100:5.1f}%  "
                  f"-> miss {ag['miss_rate'][0]:.4f}  catches {ag['caught'][0]*100:4.1f}%")
    json.dump(res, open(f"{OUT}/d3qa_{a.contrast}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
