"""Does the detectability map beat the strong baselines - including the trivial one?

Retention depends heavily on lesion radius, so a map that has only learned
"bigger lesions survive" would look impressive and be useless. Radius-alone is
therefore the control that matters, alongside the two real competitors:

  ensemble SD      what the literature would reach for, free from cross-fitting
  intensity QA map the supervised map trained on |error| rather than detectability
  radius alone     the trivial predictor the map must beat to be learning anything spatial

Scored on held-out lesions in the locked 10 test subjects, and additionally *within*
each radius, where the radius baseline is constant by construction and only spatial
information can help.
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T
from exp.d3qa import enhancer
from exp.d3detect import operator, make_labels

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
TAG = [""]


@torch.no_grad()
def score_sites(contrast, recs, cohort, enh, fwd, folds=4, bs=4):
    """At every labelled site, read off each candidate score."""
    det = Mo.UNet(6, 1, base=32, residual_from=None).cuda()
    det.load_state_dict(torch.load(f"{OUT}/d3detect_{contrast}{TAG[0]}.pt")); det.eval()
    qa = Mo.UNet(4, 1, base=32, residual_from=None).cuda()
    qa.load_state_dict(torch.load(f"{OUT}/d3qa_{contrast}.pt")); qa.eval()
    ens = []
    for f in range(folds):
        n = enhancer(); n.load_state_dict(torch.load(f"{OUT}/d3qa_fold{f}_{contrast}.pt"))
        n.eval(); ens.append(n)

    by_row = {}
    for row, cy, cx, r, c, ret in recs:
        by_row.setdefault(row, []).append((cy, cx, r, c, ret))
    rows = sorted(by_row)
    gen = torch.Generator(device="cuda"); gen.manual_seed(1)
    out = []
    for i in range(0, len(rows), bs):
        rr = rows[i:i + bs]
        y0 = torch.from_numpy(cohort.hf[np.array(rr)].astype(np.float32)).cuda()
        x0 = fwd(y0, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p0 = enh(x0).float().clamp(0, 1)
            q = qa(torch.cat([x0, p0], 1)).float()
            e = torch.stack([m(x0).float().clamp(0, 1) for m in ens]).std(0)
        for k, row in enumerate(rr):
            for (cy, cx, rad, c, ret) in by_row[row]:
                rmap = torch.full_like(p0[k:k+1], rad / 6.0)
                cmap = torch.full_like(p0[k:k+1], c / 0.25)   # signed
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    d = det(torch.cat([x0[k:k+1], p0[k:k+1], rmap, cmap], 1)).float()
                sl = (slice(max(0, cy-rad), min(256, cy+rad+1)),
                      slice(max(0, cx-rad), min(256, cx+rad+1)))
                out.append(dict(row=row, r=rad, c=c, retention=ret,
                                detect=float(d[0, 0][sl].mean()),
                                qa=float(q[k, 0][sl].mean()),
                                ens=float(e[k, 0][sl].mean())))
    return out


def report(rec, contrast, thr=0.25):
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    ret = np.array([x["retention"] for x in rec])
    rad = np.array([x["r"] for x in rec], float)
    erased = (ret < thr).astype(np.int8)
    scores = {"detectability map (ours)": np.array([x["detect"] for x in rec]),
              "intensity QA map": -np.array([x["qa"] for x in rec]),
              "ensemble SD": -np.array([x["ens"] for x in rec]),
              "lesion radius alone": rad}
    print(f"\n[{contrast}] {len(rec)} held-out lesions; "
          f"{erased.mean()*100:.0f}% erased (retention < {thr})\n")
    print(f"  {'score':28s} {'AUROC(erased)':>14s} {'spearman(retention)':>20s}")
    for nm, s in scores.items():
        print(f"  {nm:28s} {roc_auc_score(erased, -s):14.3f} "
              f"{spearmanr(s, ret).statistic:20.3f}")
    print(f"\n  within each radius (radius baseline is constant, so only spatial info helps):")
    print(f"  {'radius':>8s} {'n':>6s} " + " ".join(f"{k.split()[0]:>12s}" for k in scores if "radius" not in k))
    for q in sorted(set(rad.tolist())):
        m = rad == q
        if m.sum() < 30 or len(set(erased[m].tolist())) < 2:
            continue
        cells = []
        for nm, s in scores.items():
            if "radius" in nm: continue
            cells.append(f"{roc_auc_score(erased[m], -s[m]):12.3f}")
        print(f"  {int(q)*2:6d} mm {int(m.sum()):6d} " + " ".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--op", default="learned")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    T.seed_all(1)
    TAG[0] = a.tag
    test = D.PairedCohort("test", a.contrast)
    enh = enhancer(); enh.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_{a.contrast}.pt")); enh.eval()
    fwd = operator(a.contrast, a.op)
    print(f"[{a.contrast}] generating held-out lesion labels on the locked test set")
    lab = f"/tmp/claude-17055/-mnt-G/5ec3b7fc-b916-4b70-887e-dbe4d6709b39/scratchpad/data/detect_test_{a.contrast}.npy"
    if os.path.exists(lab):
        recs = [tuple(r) for r in np.load(lab, allow_pickle=True)]
        print(f"  reusing {len(recs)} cached test labels")
    else:
        recs = make_labels(a.contrast, "test", test.subjects, test, enh, fwd, n_sites=5, seed=1)
        np.save(lab, np.array(recs, dtype=object), allow_pickle=True)
    torch.cuda.empty_cache()
    rec = score_sites(a.contrast, recs, test, enh, fwd)
    json.dump(rec, open(f"{OUT}/d3detect_{a.contrast}{a.tag}.json", "w"), indent=1, default=float)
    report(rec, a.contrast)
