"""Two-axis trust maps: posterior dispersion x evidence sensitivity, validated against
voxel-level error and then conformalised.

Yeow et al. close by noting their uncertainty map "is interpretable in only one
direction: high standard deviations identify poorly constrained regions, but low
values do not guarantee resolved anatomy". Posterior dispersion cannot separate

    low SD because the measurement determines the anatomy      (trustworthy)
    low SD because the prior determines it and the measurement is irrelevant
                                                               (confidently wrong)

A second axis does separate them: perturb the 64 mT input and see whether the output
moves. Finding A says the ULF->3T correction is subject-specific (cross-subject
cosine 0.09), so a voxel that ignores its own measurement is being written by the
prior. This cohort has voxel-level paired ground truth, so the claim is testable
rather than only plausible.
"""
import argparse, json, sys, zlib
import numpy as np
import torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T
import exp.d3v2 as d3

OUT = "/mnt/G/Projects/ULF-MICCAI/out"


def load(contrast, tag=""):
    ck = torch.load(f"{OUT}/d3v2_diff_{contrast}{tag}.pt")
    net = d3.build(contrast, ck["base"]); net.load_state_dict(ck["sd"]); net.eval()
    ns = ck.get("sb_scale")                      # absent in the first checkpoint
    return net, d3.Schedule(gamma=ck["gamma"]), (ns if ck.get("norm") else None)


def noise_sigma(ulf):
    """Robust noise estimate from the finest Haar subband: MAD / 0.6745.

    Restricted to brain, because the volumes are skull-stripped and a median over
    the whole slice would just return the background zero.
    """
    w = d3.to_wav(ulf[:, 1:2])
    hh, ll = w[:, 3], w[:, 0]
    brain = ll > 0.02 * 2.0                      # Haar LL scales the image by 2
    if brain.sum() < 64:
        return 0.01
    return (hh[brain].abs().median() / 0.6745).clamp(min=1e-4).item()


@torch.no_grad()
def channels(net, sch, ulf, K=32, J=8, Ks=4, sig=None, seed=0, ns=None):
    """-> posterior mean, dispersion, and evidence-sensitivity, all (B,1,256,256)."""
    draws = d3.sample(net, sch, ulf, K=K, n_steps=16, seed=seed, ns=ns)
    mean, disp = draws.mean(0), draws.std(0)

    # sensitivity: re-sample with the SAME diffusion seeds under input perturbations,
    # so the only thing varying between the J estimates is the measurement itself.
    sig = sig if sig is not None else noise_sigma(ulf)
    g = torch.Generator(device="cuda"); g.manual_seed(seed + 977)
    ests = []
    supp = (ulf > 0).float()
    for j in range(J):
        pert = (ulf + sig * torch.randn(ulf.shape, device="cuda", generator=g) * supp).clamp(0, 1)
        ests.append(d3.sample(net, sch, pert, K=Ks, n_steps=16, seed=seed, ns=ns).mean(0))
    sens = torch.stack(ests).std(0)
    return mean, disp, sens, sig


@torch.no_grad()
def collect(net, sch, cohort, sids, K=32, J=8, bs=16, budget=300_000, tag="", ns=None):
    out = {}
    for sid in sids:
        idx = cohort.subject_rows(sid)
        P, U, S, Y = [], [], [], []
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            x = torch.from_numpy(cohort.ulf[j].astype(np.float32)).cuda()
            y = torch.from_numpy(cohort.hf[j].astype(np.float32)).cuda()
            mean, disp, sens, _ = channels(net, sch, x, K=K, J=J, ns=ns)
            m = ((x[:, 1:2] > 0) | (y > 0))
            P.append(mean[m]); U.append(disp[m]); S.append(sens[m]); Y.append(y[m])
        arrs = [torch.cat(v).cpu().numpy().astype(np.float32) for v in (P, U, S, Y)]
        if budget and len(arrs[0]) > budget:
            k = np.sort(np.random.default_rng(zlib.crc32(sid.encode())).choice(
                len(arrs[0]), budget, replace=False))
            arrs = [a[k] for a in arrs]
        out[sid] = tuple(arrs)
        print(f"  {tag}{sid}: {len(arrs[0])} voxels  |err| {np.abs(arrs[0]-arrs[3]).mean():.4f}  "
              f"disp {arrs[1].mean():.4f}  sens {arrs[2].mean():.4f}", flush=True)
    return out


def auroc(score, label):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(label, score)) if 0 < label.sum() < len(label) else float("nan")


def validate(te):
    """Does each channel predict real error, and does the second axis rescue the
    low-dispersion regime?"""
    from scipy.stats import spearmanr
    rows = []
    for sid, (p, u, s, y) in te.items():
        e = np.abs(p - y)
        hi = (e > np.quantile(e, 0.9)).astype(np.int8)
        r = dict(sid=sid,
                 auroc_disp=auroc(u, hi), auroc_sens=auroc(s, hi),
                 auroc_comb=auroc(u / np.maximum(s, 1e-6), hi),
                 rho_disp=float(spearmanr(u, e).statistic),
                 rho_sens=float(spearmanr(s, e).statistic))
        # the 2x2: inside the low-dispersion half, split on sensitivity
        lo = u <= np.median(u)
        smed = np.median(s[lo])
        r["err_lodisp_losens"] = float(e[lo & (s <= smed)].mean())
        r["err_lodisp_hisens"] = float(e[lo & (s > smed)].mean())
        r["ratio_losens_over_hisens"] = r["err_lodisp_losens"] / max(r["err_lodisp_hisens"], 1e-9)
        rows.append(r)
    return rows


def conformal(cal, te, alpha=0.1, mode="disp"):
    """Normalised split conformal: |y - p| <= Q * s(x). mode picks the scaling."""
    def scale(u, s):
        if mode == "const": return np.ones_like(u)
        if mode == "disp": return u + 1e-4
        if mode == "sens": return s + 1e-4
        if mode == "disp_over_sens": return (u + 1e-4) / np.sqrt(s + 1e-4)
        if mode == "disp_plus_sens": return u + s + 1e-4
        raise ValueError(mode)
    E = np.concatenate([np.abs(p - y) / scale(u, s) for p, u, s, y in cal.values()])
    n = len(E); q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    Q = float(np.quantile(E, q))
    rows = []
    for sid, (p, u, s, y) in te.items():
        w = Q * scale(u, s)
        rows.append(dict(sid=sid, coverage=float((np.abs(p - y) <= w).mean()),
                         width=float((2 * w).mean())))
    return Q, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--J", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    T.seed_all(0)
    net, sch, ns = load(a.contrast, a.tag)
    coh, tr, val = D.dev_split(a.contrast)
    test = D.PairedCohort("test", a.contrast)
    print(f"[{a.contrast}] collecting calibration ({len(val)} subj)")
    cal = collect(net, sch, coh, val, K=a.K, J=a.J, tag="cal ", ns=ns)
    print(f"[{a.contrast}] collecting test ({len(test.subjects)} subj)")
    te = collect(net, sch, test, test.subjects, K=a.K, J=a.J, tag="test ", ns=ns)

    res = {"contrast": a.contrast, "K": a.K, "J": a.J, "alpha": a.alpha}
    pm = [dict(sid=s, mae=float(np.abs(p - y).mean())) for s, (p, u, sn, y) in te.items()]
    res["point_mae"] = M.aggregate(pm)

    v = validate(te); res["validate"] = {"rows": v, "agg": M.aggregate(v)}
    g = res["validate"]["agg"]
    print(f"\n== does uncertainty predict error? (AUROC on top-10% error voxels) ==")
    print(f"  posterior dispersion   AUROC {g['auroc_disp'][0]:.3f}±{g['auroc_disp'][1]:.3f}  "
          f"spearman {g['rho_disp'][0]:.3f}")
    print(f"  evidence sensitivity   AUROC {g['auroc_sens'][0]:.3f}±{g['auroc_sens'][1]:.3f}  "
          f"spearman {g['rho_sens'][0]:.3f}")
    print(f"  dispersion / sensitivity AUROC {g['auroc_comb'][0]:.3f}±{g['auroc_comb'][1]:.3f}")
    print(f"\n== the regime posterior SD cannot see: inside the LOW-dispersion half ==")
    print(f"  mean |err|, low sensitivity  {g['err_lodisp_losens'][0]:.5f}")
    print(f"  mean |err|, high sensitivity {g['err_lodisp_hisens'][0]:.5f}")
    print(f"  ratio {g['ratio_losens_over_hisens'][0]:.3f}±{g['ratio_losens_over_hisens'][1]:.3f} "
          f"(>1 means low-sensitivity voxels are worse at equal dispersion)")

    res["conformal"] = {}
    print(f"\n== conformal intervals at alpha={a.alpha} (calibrated on {len(val)} subjects) ==")
    for mode in ["const", "disp", "sens", "disp_plus_sens", "disp_over_sens"]:
        Q, rows = conformal(cal, te, a.alpha, mode)
        ag = M.aggregate(rows)
        res["conformal"][mode] = {"Q": Q, "agg": ag, "rows": rows}
        print(f"  {mode:16s} coverage {ag['coverage'][0]:.4f}±{ag['coverage'][1]:.4f}  "
              f"mean width {ag['width'][0]:.4f}")
    json.dump(res, open(f"{OUT}/d3v2_{a.contrast}{a.tag}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
