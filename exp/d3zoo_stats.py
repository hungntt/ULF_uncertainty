"""Rank correlations between global scores and lesion retention, with hierarchical bootstrap CIs.

Unit of analysis: the ENHANCER (15: six architectures averaged over five seeds each, plus
nine single-run enhancers). Two intervals are reported for each rho, both 95% percentile
intervals over 10^4 replicates:

  primary  ("enh"): resample the enhancers, then the seeds within each multi-seed
           architecture, then the ten test subjects (the same subject draw for every model
           and metric). This reflects that only 15 enhancers were sampled.
  measurement ("subj"): resample seeds and subjects only, enhancers fixed. This shows how
           much of the uncertainty is measurement noise rather than the choice of enhancers.

Run-level (n = 39 trained models) and within-architecture (seed-level) correlations are
secondary. The point estimate uses all enhancers, subjects and seeds.
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr, wilcoxon

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
B = 10000
SPLIT = os.environ.get("ULF_SPLIT", "test")
SFX = "" if SPLIT == "test" else f"_{SPLIT}"


def load():
    z = json.load(open(f"{OUT}/d3zoo{SFX}_T2.json"))
    try:
        ss = json.load(open(f"{OUT}/d3zoo_synthseg{SFX}_T2.json"))
        for nm in z:
            if nm in ss["runs"]:
                z[nm]["synthseg"] = ss["runs"][nm]["dice"]
                z[nm]["vol_err"] = ss["runs"][nm]["vol_err"]
    except FileNotFoundError:
        pass
    return z


def matrix(z, key):
    """-> runs, enhancer of each run, (n_runs, n_subj) array aligned on subject id."""
    runs = sorted(z)
    sids = z[runs[0]]["sids"]
    X = np.array([[z[r][key][z[r]["sids"].index(s)] for s in sids] for r in runs], float)
    return runs, [z[r]["enhancer"] for r in runs], X


def rho(a, b):
    return float(spearmanr(a, b).statistic)


def enh_means(enh, X, subj_idx, seed_pick):
    names = sorted(set(enh))
    out = []
    for e in names:
        rows = [i for i, n in enumerate(enh) if n == e]
        rows = [rows[k] for k in seed_pick[e]] if e in seed_pick else rows
        out.append(X[np.ix_(rows, subj_idx)].mean())
    return names, np.array(out)


def boot_rho(z, k1, k2, level="enhancer", seed=0):
    runs, enh, X1 = matrix(z, k1)
    _, _, X2 = matrix(z, k2)
    ns = X1.shape[1]
    rng = np.random.default_rng(seed)
    multi = {e: sum(n == e for n in enh) for e in set(enh) if sum(n == e for n in enh) > 1}
    allpick = {e: list(range(m)) for e, m in multi.items()}
    if level == "enhancer":
        _, a = enh_means(enh, X1, np.arange(ns), allpick)
        _, b = enh_means(enh, X2, np.arange(ns), allpick)
    else:
        a, b = X1.mean(1), X2.mean(1)
    pt = rho(a, b)
    reps, reps_e = [], []
    for _ in range(B):
        si = rng.integers(ns, size=ns)
        if level == "enhancer":
            pick = {e: list(rng.integers(m, size=m)) for e, m in multi.items()}
            _, a = enh_means(enh, X1, si, pick); _, b = enh_means(enh, X2, si, pick)
            ei = rng.integers(len(a), size=len(a))           # resample the enhancers too
            reps_e.append(rho(a[ei], b[ei]))
        else:
            a, b = X1[:, si].mean(1), X2[:, si].mean(1)
        reps.append(rho(a, b))
    reps = np.array(reps)
    out = dict(rho=pt, lo_subj=float(np.nanpercentile(reps, 2.5)),
               hi_subj=float(np.nanpercentile(reps, 97.5)), n=len(a))
    if reps_e:
        reps_e = np.array(reps_e)
        out.update(lo=float(np.nanpercentile(reps_e, 2.5)), hi=float(np.nanpercentile(reps_e, 97.5)),
                   p_pos=float(np.nanmean(reps_e > 0)))
    else:
        out.update(lo=out["lo_subj"], hi=out["hi_subj"])
    return out


def within_arch(z, k1, k2, seed=0):
    """Mean over architectures of the seed-level Spearman (n = 5 seeds each)."""
    runs, enh, X1 = matrix(z, k1)
    _, _, X2 = matrix(z, k2)
    archs = [e for e in sorted(set(enh)) if sum(n == e for n in enh) > 1]
    def stat(si):
        v = []
        for e in archs:
            rows = [i for i, n in enumerate(enh) if n == e]
            v.append(rho(X1[rows][:, si].mean(1), X2[rows][:, si].mean(1)))
        return float(np.nanmean(v)), v
    ns = X1.shape[1]
    pt, per = stat(np.arange(ns))
    rng = np.random.default_rng(seed)
    reps = [stat(rng.integers(ns, size=ns))[0] for _ in range(2000)]
    return dict(rho=pt, lo=float(np.nanpercentile(reps, 2.5)), hi=float(np.nanpercentile(reps, 97.5)),
                per_arch=dict(zip(archs, per)))


def seed_table(z):
    runs, enh, S = matrix(z, "ssim")
    _, _, R = matrix(z, "ret")
    _, _, E = matrix(z, "erased")
    out = {}
    for e in sorted(set(enh)):
        rows = [i for i, n in enumerate(enh) if n == e]
        if len(rows) < 2: continue
        s, r, er = S[rows].mean(1), R[rows].mean(1), E[rows].mean(1)
        out[e] = dict(n=len(rows), ssim=[float(s.mean()), float(s.std(ddof=1))],
                      ret=[float(r.mean()), float(r.std(ddof=1))],
                      erased=[float(er.mean()), float(er.std(ddof=1))],
                      ssim_range=[float(s.min()), float(s.max())],
                      ret_range=[float(r.min()), float(r.max())],
                      e4=float(np.mean([z[runs[i]]["erased4"] for i in rows])))
    return out


def selection(z, keys):
    """Which enhancer each score would choose, and where that choice ranks on retention."""
    runs, enh, R = matrix(z, "ret")
    names = sorted(set(enh))
    rmean = {e: R[[i for i, n in enumerate(enh) if n == e]].mean() for e in names}
    order = sorted(names, key=lambda e: -rmean[e])
    out = {}
    for k in keys:
        _, _, X = matrix(z, k)
        xm = {e: X[[i for i, n in enumerate(enh) if n == e]].mean() for e in names}
        pick = max(names, key=lambda e: xm[e])
        out[k] = dict(pick=pick, ret=float(rmean[pick]), ret_rank=order.index(pick) + 1)
    out["ret"] = dict(pick=order[0], ret=float(rmean[order[0]]), ret_rank=1)
    return out


def paired(z, a, b, key="ret"):
    runs, enh, X = matrix(z, key)
    xa = X[[i for i, n in enumerate(enh) if n == a]].mean(0)
    xb = X[[i for i, n in enumerate(enh) if n == b]].mean(0)
    return dict(diff=float(xa.mean() - xb.mean()), p=float(wilcoxon(xa, xb).pvalue))


def main():
    z = load()
    keys = ["ssim", "psnr", "tissue"] + (["synthseg"] if "synthseg" in next(iter(z.values())) else [])
    res = {"n_runs": len(z), "n_enhancers": len({v["enhancer"] for v in z.values()})}
    print(f"{res['n_runs']} trained models, {res['n_enhancers']} enhancers\n")
    print(f"  {'score':10s} {'vs':8s} {'enhancer rho [CI enh] [CI subj]':>44s} {'run-level rho [CI]':>26s}"
          f" {'within-arch rho [CI]':>26s}")
    for k in keys:
        for t in ("ret", "erased"):
            e = boot_rho(z, k, t, "enhancer"); r = boot_rho(z, k, t, "run"); w = within_arch(z, k, t)
            res[f"{k}~{t}"] = dict(enhancer=e, run=r, within=w)
            f = lambda d: f"{d['rho']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"
            fe = f"{f(e)} [{e['lo_subj']:+.2f}, {e['hi_subj']:+.2f}]"
            print(f"  {k:10s} {t:8s} {fe:>44s} {f(r):>26s} {f(w):>26s}")
    for k in keys[1:]:
        e = boot_rho(z, "ssim", k, "enhancer")
        res[f"ssim~{k}"] = dict(enhancer=e)
        print(f"  ssim vs {k:10s} enhancer rho {e['rho']:+.2f} [{e['lo']:+.2f}, {e['hi']:+.2f}]")
    res["selection"] = selection(z, keys)
    print("\n  choosing the enhancer by each score (retention rank of the pick, 1 = best of "
          f"{res['n_enhancers']}):")
    for k, d in res["selection"].items():
        print(f"    {k:10s} picks {d['pick']:28s} retention {d['ret']:.3f}, rank {d['ret_rank']}")
    res["seeds"] = seed_table(z)
    print("\n  seed variability (5 seeds, 12000 steps each)")
    print(f"  {'architecture':20s} {'SSIM mean+-sd':>16s} {'retention mean+-sd':>20s} {'ret range':>16s} {'4 mm erased':>12s}")
    for e, d in res["seeds"].items():
        print(f"  {e:20s} {d['ssim'][0]:8.3f}+-{d['ssim'][1]:.3f} {d['ret'][0]:12.3f}+-{d['ret'][1]:.3f}"
              f"   [{d['ret_range'][0]:.3f},{d['ret_range'][1]:.3f}] {d['e4']*100:10.0f}%")
    res["pix2pix_vs_unet"] = paired(z, "cGAN / pix2pix", "U-Net L1")
    print(f"\n  pix2pix vs U-Net L1, seeds averaged, paired over subjects: "
          f"{res['pix2pix_vs_unet']['diff']:+.3f} (p={res['pix2pix_vs_unet']['p']:.4f})")
    json.dump(res, open(f"{OUT}/d3zoo_stats{SFX}_T2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
