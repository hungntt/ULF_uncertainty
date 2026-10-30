"""Second calibration pass: is the scale-level trust score weak because the
evidence is weak, or because of how it was built?

Two defects in the first pass. (1) The MIND disagreement feature was computed
once at full resolution and pooled, so it carried almost the same value at every
scale (0.1801 / 0.1805 / 0.1809) and could not express per-band structure.
(2) The score is a per-slice scalar, while this project's supervised QA map
(AUROC 0.820) works per voxel. Both are fixed here and scored against the same
out-of-fold teacher, so the comparison is like for like.

Out-of-fold teacher predictions are cached, so refitting costs no GPU time.
Runs capped to a small slice of the card: the main ladder owns the GPU and must
not be the process that dies if this one asks for too much.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, eco as E, metrics as M, models as Mo, train as T

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
CACHE = "/mnt/G/Projects/ULF-MICCAI/datasets/npy"
NSC, SCALES = 3, ["fine", "med", "coarse"]
import importlib.util
_s = importlib.util.spec_from_file_location("ecomt", "/mnt/G/Projects/ULF-MICCAI/exp/ecomt.py")
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
aurc, spearman, h1_table = _m.aurc, _m.spearman, _m.h1_table


def feats_v2(p1, p2, ulf_c, win=9):
    return E.features_sp(p1, p2, ulf_c, NSC, win)


def band_err(pred, hf, win=9):
    return [E.loc((a - b).abs(), win) for a, b in zip(E.lap_pyr(pred, NSC), E.lap_pyr(hf, NSC))]


def cache_oof(a):
    """Inner-fold teacher predictions under two weak views, for every training
    subject, plus the operational teacher's predictions on dev-val and test."""
    coh, tr, val = D.dev_split(a.contrast)
    tr, test = sorted(tr), D.PairedCohort("test", a.contrast)
    rng = np.random.default_rng(0)
    P1, P2 = np.zeros_like(coh.hf), np.zeros_like(coh.hf)

    def predict(ema, cohort, sids, p1, p2):
        for sid in sids:
            idx = cohort.subject_rows(sid)
            for i in range(0, len(idx), 8):
                j = idx[i:i + 8]
                u = torch.from_numpy(cohort.ulf[j].astype(np.float32)).cuda()
                p1[j] = ema(E.weak(u, rng)).cpu().numpy().astype(p1.dtype)
                p2[j] = ema(E.weak(u, rng)).cpu().numpy().astype(p2.dtype)

    for f in range(a.folds):
        hold = tr[f::a.folds]
        fit = [s for s in tr if s not in hold]
        print(f"[fold {f}] warm-up on {len(fit)}, predict {len(hold)}", flush=True)
        ema, net = _m._warmup(coh, fit, a.steps, seed=300 + f, bs=a.bs)
        predict(ema, coh, hold, P1, P2)
        del net, ema; torch.cuda.empty_cache()
    np.save(f"{CACHE}/eco2_oof1_{a.contrast}.npy", P1)
    np.save(f"{CACHE}/eco2_oof2_{a.contrast}.npy", P2)

    print("[operational teacher] warm-up on all 32", flush=True)
    ema, net = _m._warmup(coh, tr, a.steps, seed=300 + a.folds, bs=a.bs)
    V1, V2 = np.zeros_like(coh.hf), np.zeros_like(coh.hf)
    predict(ema, coh, val, V1, V2)
    np.save(f"{CACHE}/eco2_dev1_{a.contrast}.npy", V1)
    np.save(f"{CACHE}/eco2_dev2_{a.contrast}.npy", V2)
    S1, S2 = np.zeros_like(test.hf), np.zeros_like(test.hf)
    predict(ema, test, test.subjects, S1, S2)
    np.save(f"{CACHE}/eco2_test1_{a.contrast}.npy", S1)
    np.save(f"{CACHE}/eco2_test2_{a.contrast}.npy", S2)
    torch.save(net.state_dict(), f"{OUT}/eco2_warm_{a.contrast}.pt")
    print("cached", flush=True)


def gather(coh, sids, p1, p2, vox_per_slice=0, seed=0):
    """Scale-level scalars, and optionally a voxel subsample, for a subject set."""
    rng = np.random.default_rng(seed)
    SX, SY, SS, VX, VY, VS = [], [], [], [], [], []
    for sid in sids:
        idx = coh.subject_rows(sid)
        for i in range(0, len(idx), 8):
            j = idx[i:i + 8]
            u = torch.from_numpy(coh.ulf[j].astype(np.float32)).cuda()
            h = torch.from_numpy(coh.hf[j].astype(np.float32)).cuda()
            a1 = torch.from_numpy(p1[j].astype(np.float32)).cuda()
            a2 = torch.from_numpy(p2[j].astype(np.float32)).cuda()
            c = u[:, 1:2]
            fe, be = feats_v2(a1, a2, c), band_err(a1, h)
            m = E.pyr_pool(((c > 0) | (h > 0)).float(), NSC)
            sx, sy = [], []
            for l in range(NSC):
                n = m[l].sum((1, 2, 3)).clamp(min=1)
                sx.append(torch.stack([(fe[l][k] * m[l]).sum((1, 2, 3)) / n for k in E.FEATS], 1))
                sy.append((be[l] * m[l]).sum((1, 2, 3)) / n)
            SX.append(torch.stack(sx, 1).cpu().numpy()); SY.append(torch.stack(sy, 1).cpu().numpy())
            SS += [sid] * len(j)
            if vox_per_slice:
                for l in range(NSC):
                    fl = torch.cat([fe[l][k] for k in E.FEATS], 1)
                    ok = (m[l] > 0.5)[:, 0]
                    for b in range(len(j)):
                        w = ok[b].nonzero()
                        if not len(w):
                            continue
                        p = w[torch.from_numpy(rng.integers(0, len(w), vox_per_slice)).cuda()]
                        VX.append(fl[b, :, p[:, 0], p[:, 1]].T.cpu().numpy())
                        VY.append(be[l][b, 0, p[:, 0], p[:, 1]].cpu().numpy())
                        VS.append(np.full(vox_per_slice, f"{sid}|{l}"))
    out = [np.concatenate(SX), np.concatenate(SY), np.array(SS)]
    if vox_per_slice:
        out += [np.concatenate(VX), np.concatenate(VY), np.concatenate(VS)]
    return out


def auroc(score, y):
    o = np.argsort(score)
    r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    n1 = y.sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def fit(a):
    coh, tr, val = D.dev_split(a.contrast)
    tr, test = sorted(tr), D.PairedCohort("test", a.contrast)
    ld = lambda n: np.load(f"{CACHE}/eco2_{n}_{a.contrast}.npy")
    res = {}

    X, Y, S, VX, VY, VS = gather(coh, tr, ld("oof1"), ld("oof2"), a.vox)
    print(f"out-of-fold: {len(X)} slices, {len(VX)} voxels", flush=True)
    risk = E.Risk.fit(X, Y, alpha=a.ridge)
    res["risk"] = risk.state()
    res["weights"] = {SCALES[l]: dict(zip(E.FEATS, risk.w[l].tolist())) for l in range(NSC)}

    # voxel-level ridge: one model per scale, same three features
    vox, vst = {}, {"mu": [], "sd": [], "w": [], "b": []}
    for l in range(NSC):
        m = np.array([s.endswith(f"|{l}") for s in VS])
        vx, vy = VX[m][:, None, :], VY[m][:, None]
        vox[l] = E.Risk.fit(vx, vy, alpha=a.ridge)
        for k in vst:
            vst[k].append(np.asarray(getattr(vox[l], k))[0].tolist()
                          if np.asarray(getattr(vox[l], k)).ndim > 1
                          else float(np.asarray(getattr(vox[l], k))[0]))
    res["risk_vox"] = vst
    rv = E.Risk.load(vst)

    # tolerance: the same rule as the scalar gates, mean weight 0.5 on the
    # unlabelled pool, so arms differ in where trust sits and not how much.
    net = Mo.UNet(3, 1, base=32, residual_from=None).cuda()
    net.load_state_dict(torch.load(f"{OUT}/eco2_warm_{a.contrast}.pt"))
    ema = E.EMA(net)
    unp = D.unpaired_ulf(a.contrast)
    rng2 = np.random.default_rng(7)
    pk = np.sort(rng2.choice(len(unp), min(512, len(unp)), replace=False))
    acc = [[] for _ in range(NSC)]
    with torch.no_grad():
        for i in range(0, len(pk), 8):
            xu = torch.from_numpy(unp[pk[i:i + 8]].astype(np.float32)).cuda()
            c = xu[:, 1:2]
            mp = E.pyr_pool((c > 0).float(), NSC)
            rm = rv.apply_sp(E.features_sp(ema(E.weak(xu, rng2)), ema(E.weak(xu, rng2)), c, NSC))
            for l in range(NSC):
                acc[l].append(rm[l][mp[l] > 0.5].cpu().numpy())
    tau_vox = []
    for l in range(NSC):
        v = np.concatenate(acc[l])
        lo, hi = 1e-10, float(np.abs(v).max()) * 100 + 1e-6
        for _ in range(80):
            mid = (lo + hi) / 2
            if np.clip(1 - v / mid, 0, 1).mean() < 0.5:
                lo = mid
            else:
                hi = mid
        tau_vox.append(float((lo + hi) / 2))
    res["tau_vox"] = tau_vox
    print("tau_vox", [round(t, 5) for t in tau_vox], flush=True)
    del net, ema; torch.cuda.empty_cache()

    for nm, (c, ids, k) in {"dev_val": (coh, val, "dev"), "test": (test, test.subjects, "test")}.items():
        x, y, s, vx, vy, vs = gather(c, ids, ld(k + "1"), ld(k + "2"), a.vox)
        res.setdefault("h1", {})[nm] = h1_table(x, y, s, risk)
        rows = []
        for l in range(NSC):
            m = np.array([t.endswith(f"|{l}") for t in vs])
            xx, yy = vx[m], vy[m]
            hot = yy >= np.percentile(yy, 90)
            sc = {"r_hat (voxel)": vox[l](xx[:, None, :])[:, 0],
                  "u": xx[:, 0], "d": xx[:, 1], "s": xx[:, 2]}
            for q, v in sc.items():
                rows.append({"scale": SCALES[l], "score": q, "auroc": auroc(v, hot),
                             "spearman": spearman(v, yy)})
        res.setdefault("voxel", {})[nm] = rows
    json.dump(res, open(f"{OUT}/eco_calib2_{a.contrast}.json", "w"), indent=1, default=float)

    print(f"\n== scale-level H1 (v2 features), {a.contrast} ==")
    for nm in res["h1"]:
        print(f"-- {nm} --")
        for r in res["h1"][nm]:
            print(f"  {r['scale']:7s} {r['score']:24s} rho {r['spearman']:+.3f} "
                  f"[{r['spearman_lo']:+.3f},{r['spearman_hi']:+.3f}]  AURC {r['aurc']:.3f}")
    print(f"\n== voxel-level: AUROC for the worst-10% error voxels ==")
    for nm in res["voxel"]:
        print(f"-- {nm} --")
        for r in res["voxel"][nm]:
            print(f"  {r['scale']:7s} {r['score']:14s} AUROC {r['auroc']:.3f}  rho {r['spearman']:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cache", "fit"])
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--vox", type=int, default=200)
    ap.add_argument("--memfrac", type=float, default=0.14)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.memfrac)
    (cache_oof if a.stage == "cache" else fit)(a)
