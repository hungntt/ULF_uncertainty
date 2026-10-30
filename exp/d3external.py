"""External cohorts: does the closed loop hold on other 64 mT scanners, and does the map transfer?

For each external cohort from prep_external.py, with the map's home enhancer
(U-Net absolute, base_unet_noresid_T2) and identical lesion sites in both arms:

  sim   the enhancer receives the learned operator's rendering of the lesioned 3 T
        (the closed loop used for every retention figure);
  real  the enhancer receives the ACQUIRED 64 mT slab of that scanner plus the operator's
        local response to the lesion (d3detreal.lesion_input), so the scanner's own noise,
        texture and contrast are kept and only the lesion is synthetic.

Reported: mean retention and % erased in each arm and their Spearman over matched sites
(the POCEMR value is +0.942); AUROC for erasure of the lesion-retention map (four seeds,
trained on POCEMR only), of the intensity-error QA map, of the deep-ensemble s.d. and of
lesion radius alone, pooled and within each radius; and, on the real arm, the map queried
on the acquired slab. Subject-bootstrap 95% intervals throughout.

    python exp/d3external.py ext_on_hfc ext_on_hfe ext_v5
"""
import json, sys
import numpy as np
import torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, models as Mo, train as T
from exp.d3qa import enhancer
from exp.d3detect import operator
from exp.d3detreal import make_labels

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
MAPS = ["d3detect_T2", "d3detect_T2_s1", "d3detect_T2_s2", "d3detect_T2_s3"]


@torch.no_grad()
def score(recs, cohort, enh, fwd, mode, maps, bs=4):
    """Score every labelled site with each map and baseline, on the input of this arm."""
    dets = []
    for m in maps:
        d = Mo.UNet(6, 1, base=32, residual_from=None).cuda()
        d.load_state_dict(torch.load(f"{OUT}/{m}.pt")); d.eval(); dets.append((m, d))
    qa = Mo.UNet(4, 1, base=32, residual_from=None).cuda()
    qa.load_state_dict(torch.load(f"{OUT}/d3qa_T2.pt")); qa.eval()
    ens = []
    for f in range(4):
        n = enhancer(); n.load_state_dict(torch.load(f"{OUT}/d3qa_fold{f}_T2.pt")); n.eval()
        ens.append(n)
    by_row = {}
    for row, cy, cx, r, c, ret in recs:
        by_row.setdefault(row, []).append((cy, cx, r, c, ret))
    rows = sorted(by_row)
    gen = torch.Generator(device="cuda"); gen.manual_seed(1)
    out = []
    for i in range(0, len(rows), bs):
        rr = np.array(rows[i:i + bs])
        y0 = torch.from_numpy(cohort.hf[rr].astype(np.float32)).cuda()
        x0 = (torch.from_numpy(cohort.ulf[rr].astype(np.float32)).cuda() if mode == "real"
              else fwd(y0, gen))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p0 = enh(x0).float().clamp(0, 1)
            q = qa(torch.cat([x0, p0], 1)).float()
            e = torch.stack([m(x0).float().clamp(0, 1) for m in ens]).std(0)
        for k, row in enumerate(rr):
            for (cy, cx, rad, c, ret) in by_row[int(row)]:
                rmap = torch.full_like(p0[k:k + 1], rad / 6.0)
                cmap = torch.full_like(p0[k:k + 1], c / 0.25)
                sl = (slice(max(0, cy - rad), min(256, cy + rad + 1)),
                      slice(max(0, cx - rad), min(256, cx + rad + 1)))
                rec = dict(row=int(row), sid=sid_of(cohort, int(row)), cy=cy, cx=cx, r=rad,
                           c=c, retention=ret, qa=float(q[k, 0][sl].mean()),
                           ens=float(e[k, 0][sl].mean()))
                for nm, d in dets:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        v = d(torch.cat([x0[k:k + 1], p0[k:k + 1], rmap, cmap], 1)).float()
                    rec[nm] = float(v[0, 0][sl].mean())
                out.append(rec)
    return out


def sid_of(cohort, row):
    for s in cohort.man:
        if s["slice_start"] <= row < s["slice_start"] + s["slice_count"]:
            return s["subject_id"]


def auroc_ci(erased, s, sids, n=2000, seed=0):
    from sklearn.metrics import roc_auc_score
    pt = roc_auc_score(erased, -s)
    rng = np.random.default_rng(seed)
    us = sorted(set(sids)); ix = {u: np.where(sids == u)[0] for u in us}
    reps = []
    for _ in range(n):
        idx = np.concatenate([ix[u] for u in rng.choice(us, len(us))])
        if len(set(erased[idx].tolist())) == 2:
            reps.append(roc_auc_score(erased[idx], -s[idx]))
    return float(pt), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def summarise(rec, maps, thr=0.25):
    ret = np.array([x["retention"] for x in rec]); er = (ret < thr).astype(int)
    rad = np.array([x["r"] for x in rec], float); sids = np.array([x["sid"] for x in rec])
    res = {"n": len(rec), "erased": float(er.mean()), "ret": float(ret.mean())}
    scores = {m: np.array([x[m] for x in rec]) for m in maps}
    scores.update({"qa": -np.array([x["qa"] for x in rec]), "ens": -np.array([x["ens"] for x in rec]),
                   "radius": rad})
    for k, s in scores.items():
        res[k] = auroc_ci(er, s, sids)
    from sklearn.metrics import roc_auc_score
    res["within"] = {}
    for q in sorted(set(rad.tolist())):
        m = rad == q
        if m.sum() >= 30 and len(set(er[m].tolist())) == 2:
            res["within"][f"{int(q) * 2} mm"] = {k: float(roc_auc_score(er[m], -scores[k][m]))
                                                 for k in maps + ["qa", "ens"]}
    a = np.array([res[m][0] for m in maps])
    res["map_mean"], res["map_sd"] = float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return res


def main(splits):
    from scipy.stats import spearmanr
    T.seed_all(1)
    enh = enhancer(); enh.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_T2.pt")); enh.eval()
    fwd = operator("T2", "learned")
    allres = {}
    for split in splits:
        coh = D.PairedCohort(split, "T2")
        print(f"\n== {split}: {len(coh.subjects)} subjects ==", flush=True)
        lab = {m: make_labels(coh, coh.subjects, enh, fwd, n_sites=5, seed=1, mode=m)
               for m in ("sim", "real")}
        key = lambda r: tuple(r[:5])
        sim = {key(r): r[5] for r in lab["sim"]}; real = {key(r): r[5] for r in lab["real"]}
        common = sorted(set(sim) & set(real))
        a = np.array([sim[k] for k in common]); b = np.array([real[k] for k in common])
        res = {"closed_loop": dict(n=len(common), ret_sim=float(a.mean()), ret_real=float(b.mean()),
                                   erased_sim=float((a < 0.25).mean()),
                                   erased_real=float((b < 0.25).mean()),
                                   spearman=float(spearmanr(a, b).statistic))}
        print(f"  closed loop: {len(common)} matched sites, retention sim {a.mean():.3f} / real "
              f"{b.mean():.3f}, erased {np.mean(a < .25)*100:.0f}% / {np.mean(b < .25)*100:.0f}%, "
              f"Spearman {res['closed_loop']['spearman']:+.3f}", flush=True)
        rs = score(lab["sim"], coh, enh, fwd, "sim", MAPS)
        res["sim"] = summarise(rs, MAPS)
        rr = score(lab["real"], coh, enh, fwd, "real", MAPS + ["d3detect_T2_realin"])
        res["real"] = summarise(rr, MAPS + ["d3detect_T2_realin"])
        json.dump(dict(sim=rs, real=rr), open(f"{OUT}/d3external_rows_{split}.json", "w"))
        for arm in ("sim", "real"):
            r = res[arm]
            print(f"  [{arm}] n={r['n']} erased {r['erased']*100:.0f}%  map AUROC "
                  f"{r['map_mean']:.3f}+-{r['map_sd']:.3f}  QA {r['qa'][0]:.3f}  ens {r['ens'][0]:.3f}  "
                  f"radius {r['radius'][0]:.3f}" +
                  (f"  realin-map {r['d3detect_T2_realin'][0]:.3f}" if arm == "real" else ""), flush=True)
            print("     within size:", {k: round(np.mean([v[m] for m in MAPS]), 3)
                                       for k, v in r["within"].items()})
        allres[split] = res
        json.dump(allres, open(f"{OUT}/d3external.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["ext_on_hfc", "ext_on_hfe", "ext_v5"])
