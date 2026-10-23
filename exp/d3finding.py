"""Finding-aware conformal risk control, on hierarchical units.

The report's central negative result: the voxel-error guarantee behaves correctly by
its own definition while a 4 mm lesion loses 83% of its contrast. Conformal risk
control is not tied to pixel intensity - it controls any bounded, monotone loss - so
the fix is to re-pose the loss around missed findings:

    R_i(lam) = among lesions this map declared safe in subject i,
               the fraction that were actually erased

Flag where the predicted detectability d < lam. Raising lam flags more and declares
fewer safe, so R is non-increasing and CRC applies. The cost of the guarantee is the
fraction of brain flagged, and - because a reader acts on regions, not voxels - the
number of connected flagged regions per slice.

Exchangeability note: lesions and regions cluster within a subject, so the
exchangeable unit is the SUBJECT. Risk is computed per subject and averaged over
subjects, which keeps the CRC floor at 1/(n+1) = 0.111 for 8 calibration subjects.
Treating lesions as iid would give a much finer alpha and an invalid guarantee.
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T
from exp.d3qa import enhancer
from exp.d3detect import operator, make_labels
from exp.d3detect_eval import score_sites

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
CACHE = "/tmp/claude-17055/-mnt-G/5ec3b7fc-b916-4b70-887e-dbe4d6709b39/scratchpad/data"


def gather(contrast, split, sids, cohort, enh, fwd, seed):
    """Lesion labels + every candidate score at each site, cached."""
    f = f"{CACHE}/finding_{split}_{contrast}.json"
    if os.path.exists(f):
        rec = json.load(open(f)); print(f"  reusing {len(rec)} cached {split} lesions")
        return rec
    recs = make_labels(contrast, split, sids, cohort, enh, fwd, n_sites=5, seed=seed)
    rec = score_sites(contrast, recs, cohort, enh, fwd)
    # attach the subject each lesion belongs to, for subject-level risk
    row2sid = {}
    for s in sids:
        for r in cohort.subject_rows(s):
            row2sid[int(r)] = s
    for x in rec:
        x["sid"] = row2sid[x["row"]]
    json.dump(rec, open(f, "w"), indent=1, default=float)
    return rec


def risk(rec, key, lam, thr, sign):
    """Per-subject: among lesions declared safe, the fraction actually erased."""
    out = {}
    for x in rec:
        out.setdefault(x["sid"], []).append(x)
    R = []
    for sid, xs in out.items():
        safe = [x for x in xs if sign * x[key] > sign * lam]
        if not safe:
            R.append(0.0); continue
        R.append(float(np.mean([x["retention"] < thr for x in safe])))
    return np.array(R)


def crc(cal, te, key, thr, alpha, sign, n_grid=160):
    """Smallest flagging effort whose CRC bound on the missed-finding rate holds."""
    v = np.array([x[key] for x in cal])
    grid = np.quantile(v, np.linspace(0.0, 1.0, n_grid))
    if sign < 0: grid = grid[::-1]
    n = len({x["sid"] for x in cal})
    chosen, ok = (grid[-1], False)
    for lam in grid:
        R = risk(cal, key, lam, thr, sign)
        if (n * R.mean() + 1) / (n + 1) <= alpha:
            chosen, ok = lam, True; break
    Rt = risk(te, key, chosen, thr, sign)
    flagged = np.mean([sign * x[key] <= sign * chosen for x in te])
    caught = np.mean([x["retention"] < thr and sign * x[key] <= sign * chosen for x in te]) / \
             max(np.mean([x["retention"] < thr for x in te]), 1e-9)
    return dict(lam=float(chosen), feasible=bool(ok), floor=1.0 / (n + 1),
                test_risk=float(Rt.mean()), test_risk_max=float(Rt.max()),
                lesion_flagged=float(flagged), erased_caught=float(caught))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--op", default="learned")
    ap.add_argument("--thr", type=float, default=0.25, help="retention below this = erased")
    a = ap.parse_args()
    T.seed_all(2)
    coh, tr, val = D.dev_split(a.contrast)
    test = D.PairedCohort("test", a.contrast)
    enh = enhancer(); enh.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_{a.contrast}.pt")); enh.eval()
    fwd = operator(a.contrast, a.op)

    print(f"[{a.contrast}] calibration lesions ({len(val)} dev-val subjects)")
    cal = gather(a.contrast, "cal", val, coh, enh, fwd, seed=7)
    print(f"[{a.contrast}] test lesions ({len(test.subjects)} locked subjects)")
    te = gather(a.contrast, "test", test.subjects, test, enh, fwd, seed=1)

    base = np.mean([x["retention"] < a.thr for x in te])
    nsub = len({x["sid"] for x in cal})
    print(f"\n  {len(cal)} calibration / {len(te)} test lesions; "
          f"{base*100:.0f}% of test lesions erased (retention < {a.thr})")
    print(f"  CRC floor with {nsub} calibration subjects: alpha > {1/(nsub+1):.3f}")

    # higher detectability = safer; higher intensity-error score = the map calls it risky
    cands = [("detectability map (ours)", "detect", +1),
             ("intensity QA map", "qa", -1),
             ("ensemble SD", "ens", -1)]
    res = {"contrast": a.contrast, "thr": a.thr, "base_erased": float(base), "crc": {}}
    for alpha in (0.15, 0.20, 0.30):
        print(f"\n== guaranteed missed-finding rate <= {alpha} ==")
        print(f"  {'score':26s} {'lesions flagged':>16s} {'realised miss':>14s} {'erased caught':>14s}")
        for nm, key, sign in cands:
            r = crc(cal, te, key, a.thr, alpha, sign)
            res["crc"].setdefault(str(alpha), {})[key] = r
            tag = "" if r["feasible"] else "  (INFEASIBLE)"
            print(f"  {nm:26s} {r['lesion_flagged']*100:15.1f}% {r['test_risk']:14.3f} "
                  f"{r['erased_caught']*100:13.1f}%{tag}")
    json.dump(res, open(f"{OUT}/d3finding_{a.contrast}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
