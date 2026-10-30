"""Summarise d3external.py on the paper's protocol: bright lesions (c = 0.12, 0.25), as in the
cached POCEMR test labels behind the reported map AUROC. Dark lesions are reported separately.

    python exp/d3external_summary.py   -> out/d3external_summary.json
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
MAPS = ["d3detect_T2", "d3detect_T2_s1", "d3detect_T2_s2", "d3detect_T2_s3"]
SPLITS = ["test", "ext_on_hfc", "ext_on_hfe", "ext_v5"]


def boot(rows, fn, n=2000, seed=0):
    sids = np.array([x["sid"] for x in rows]); us = sorted(set(sids))
    ix = {u: np.where(sids == u)[0] for u in us}
    rng = np.random.default_rng(seed); reps = []
    for _ in range(n):
        idx = np.concatenate([ix[u] for u in rng.choice(us, len(us))])
        try:
            reps.append(fn([rows[i] for i in idx]))
        except ValueError:
            pass
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def auc(rows, key, sign=-1):
    er = np.array([x["retention"] < 0.25 for x in rows], int)
    return roc_auc_score(er, sign * np.array([x[key] for x in rows], float))


def map_auc(rows):
    return float(np.mean([auc(rows, m) for m in MAPS]))


def within(rows, key_fn):
    out = {}
    for r in (2, 3, 4, 6):
        R = [x for x in rows if x["r"] == r]
        er = [x["retention"] < 0.25 for x in R]
        if len(R) >= 30 and 0 < sum(er) < len(R):
            out[f"{2 * r} mm"] = key_fn(R)
    return out


def main():
    res = {}
    for split in SPLITS:
        f = f"{OUT}/d3external_rows_{split}.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        res[split] = {}
        for pol, sel in (("bright", lambda x: x["c"] > 0), ("dark", lambda x: x["c"] < 0)):
            r = {}
            for arm in ("sim", "real"):
                R = [x for x in d[arm] if sel(x)]
                a = [auc(R, m) for m in MAPS]
                r[arm] = dict(n=len(R), erased=float(np.mean([x["retention"] < 0.25 for x in R])),
                              ret=float(np.mean([x["retention"] for x in R])),
                              map=[float(np.mean(a)), float(np.std(a, ddof=1))],
                              map_ci=boot(R, map_auc),
                              qa=float(auc(R, "qa", +1)), ens=float(auc(R, "ens", +1)),
                              radius=float(auc(R, "r")),
                              within=within(R, map_auc))
                if arm == "real":
                    r[arm]["realin_map"] = float(auc(R, "d3detect_T2_realin"))
            # matched sites: same (row, cy, cx, r, c) in both arms
            k = lambda x: (x["row"], x["cy"], x["cx"], x["r"], x["c"])
            S = {k(x): x["retention"] for x in d["sim"] if sel(x)}
            Q = {k(x): x["retention"] for x in d["real"] if sel(x)}
            com = sorted(set(S) & set(Q))
            a = np.array([S[c] for c in com]); b = np.array([Q[c] for c in com])
            r["closed_loop"] = dict(n=len(com), spearman=float(spearmanr(a, b).statistic),
                                    ret_sim=float(a.mean()), ret_real=float(b.mean()))
            res[split][pol] = r
        b = res[split]["bright"]
        print(f"{split:11s} bright: n={b['sim']['n']}  closed loop rho {b['closed_loop']['spearman']:+.3f} "
              f"ret {b['closed_loop']['ret_sim']:.3f}/{b['closed_loop']['ret_real']:.3f} | "
              f"map sim {b['sim']['map'][0]:.3f} [{b['sim']['map_ci'][0]:.3f},{b['sim']['map_ci'][1]:.3f}] "
              f"real {b['real']['map'][0]:.3f}  QA {b['sim']['qa']:.3f} ens {b['sim']['ens']:.3f} "
              f"radius {b['sim']['radius']:.3f}  within {[round(v, 2) for v in b['sim']['within'].values()]}")
        dk = res[split]["dark"]
        print(f"{'':11s} dark:   map sim {dk['sim']['map'][0]:.3f}  real {dk['real']['map'][0]:.3f}  "
              f"realin-map (real) {dk['real']['realin_map']:.3f}  erased {dk['sim']['erased']*100:.0f}%")
    json.dump(res, open(f"{OUT}/d3external_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
