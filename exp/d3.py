"""Direction 3 — distribution-free risk control for ULF enhancement.

An enhanced 64 mT image is only clinically usable if the reader knows where to
trust it. We attach per-voxel prediction intervals with a finite-sample
guarantee and ask three questions that matter in the ULF regime specifically:

  * does heteroscedastic (CQR) calibration buy tighter intervals than a constant
    width at the same coverage?
  * does the interval width actually localise error (i.e. is it a usable
    hallucination map), or is it just an edge detector?
  * how many *paired subjects* does the calibration set need? Low-field cohorts
    have tens of subjects, not thousands, and the exchangeable unit is a
    subject, not a voxel.
"""
import argparse, json, sys, zlib
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T

OUT = "/mnt/G/Projects/ULF-MICCAI/out"


def pinball(p, y, taus, mask):
    """p: (B,K,H,W) quantile predictions; y: (B,1,H,W)."""
    d = y - p
    L = torch.stack([torch.maximum(t * d[:, k:k + 1], (t - 1) * d[:, k:k + 1])
                     for k, t in enumerate(taus)], 1).squeeze(2)
    return L.mean() + (L * mask).sum() / mask.sum().clamp(min=1) / len(taus)


def train_quantile(contrast, alpha, steps, base=32, bs=16, mode="cqr"):
    T.seed_all(0)
    coh, tr, val = D.dev_split(contrast)
    taus = [alpha / 2, 0.5, 1 - alpha / 2]
    K = 3 if mode == "cqr" else 2          # gaussian: [mean, log-sigma]
    net = Mo.UNet(3, K, base=base, residual_from=[1] * K).cuda()
    rows = coh.rows_for(tr)
    rng = np.random.default_rng(0)
    opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=steps, pct_start=0.05)
    for it in range(steps):
        p_ = np.sort(rng.choice(len(rows), bs, replace=False))
        r = rows[p_]
        x = torch.from_numpy(coh.ulf[r].astype(np.float32)).cuda()
        y = torch.from_numpy(coh.hf[r].astype(np.float32)).cuda()
        if rng.random() < 0.5: x, y = torch.flip(x, [-1]), torch.flip(y, [-1])
        mask = (y > 0).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = net(x).float()
            if mode == "cqr":
                loss = pinball(p, y, taus, mask)
            else:
                mu, ls = p[:, :1], p[:, 1:2].clamp(-7, 2)
                nll = 0.5 * (((y - mu) ** 2) * torch.exp(-2 * ls) + 2 * ls)
                loss = nll.mean() + (nll * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
        if (it + 1) % 2000 == 0:
            print(f"  step {it+1:6d} loss {loss.item():.5f}")
    torch.save(net.state_dict(), f"{OUT}/d3_{mode}_{contrast}.pt")
    return net


@torch.no_grad()
def collect(net, cohort, sids, mode, alpha, budget=300_000):
    """Per-subject arrays of (lo, hi, point, y) inside the brain mask.

    Subsampled to `budget` voxels per subject: a subject contributes ~1M brain
    voxels, which is far more than coverage estimation needs and makes the
    repeated-split study below tractable."""
    from scipy.stats import norm
    net.eval(); out = {}
    z = float(norm.ppf(1 - alpha / 2))
    for sid in sids:
        idx = cohort.subject_rows(sid)
        L, H, P, Y = [], [], [], []
        for i in range(0, len(idx), 32):
            j = idx[i:i + 32]
            x = torch.from_numpy(cohort.ulf[j].astype(np.float32)).cuda()
            y = torch.from_numpy(cohort.hf[j].astype(np.float32)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = net(x).float()
            m = ((x[:, 1:2] > 0) | (y > 0))
            if mode == "cqr":
                lo, pt, hi = p[:, 0:1], p[:, 1:2], p[:, 2:3]
                lo, hi = torch.minimum(lo, hi), torch.maximum(lo, hi)
            else:
                pt = p[:, 0:1]; s = torch.exp(p[:, 1:2].clamp(-7, 2))
                lo, hi = pt - z * s, pt + z * s
            L.append(lo[m]); H.append(hi[m]); P.append(pt[m]); Y.append(y[m])
        arrs = [torch.cat(v).cpu().numpy().astype(np.float32) for v in (L, H, P, Y)]
        if budget and len(arrs[0]) > budget:
            k = np.random.default_rng(zlib.crc32(sid.encode())).choice(len(arrs[0]), budget, replace=False)
            arrs = [a[np.sort(k)] for a in arrs]
        out[sid] = tuple(arrs)
    net.train(); return out


def cqr_calibrate(cal, alpha, mode="hetero"):
    """Split-conformal inflation Q from the calibration subjects' pooled voxels."""
    E = []
    for lo, hi, pt, y in cal.values():
        E.append(np.maximum(lo - y, y - hi) if mode == "hetero" else np.abs(y - pt))
    E = np.concatenate(E)
    n = len(E)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(E, q))


def eval_intervals(te, Q, alpha, mode="hetero"):
    rows = []
    for sid, (lo, hi, pt, y) in te.items():
        l, h = (lo - Q, hi + Q) if mode == "hetero" else (pt - Q, pt + Q)
        cov = float(((y >= l) & (y <= h)).mean())
        rows.append(dict(sid=sid, coverage=cov, width=float((h - l).mean()),
                         median_width=float(np.median(h - l))))
    return rows


def crc_lambda(cal, alpha, grid):
    """Conformal risk control with the subject as the exchangeable unit.

    lambda_hat = inf{ lambda : (n * Rbar(lambda) + B) / (n + 1) <= alpha }, B = 1.
    Because the risk is bounded by B = 1, no lambda can satisfy this unless
    alpha > 1/(n+1): a cohort of n paired subjects cannot certify a risk level
    finer than 1/(n+1), however good the model is. With the 8 calibration
    subjects a 50-subject ULF study can spare, the floor is 0.111."""
    n = len(cal)
    feasible = alpha > 1.0 / (n + 1)
    for lam in grid:
        R = [float(((y < lo - lam) | (y > hi + lam)).mean()) for lo, hi, pt, y in cal.values()]
        if (n * np.mean(R) + 1) / (n + 1) <= alpha:
            return float(lam), True, 1.0 / (n + 1)
    return float(grid[-1]), feasible, 1.0 / (n + 1)


def width_quality(te):
    """Does interval width localise error? Spearman rho and AUROC for |err| > 90th pct."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    rs, au = [], []
    rng = np.random.default_rng(0)
    for sid, (lo, hi, pt, y) in te.items():
        w = hi - lo; e = np.abs(y - pt)
        k = rng.choice(len(w), min(200000, len(w)), replace=False)
        rs.append(float(spearmanr(w[k], e[k]).statistic))
        thr = np.quantile(e, 0.9)
        lab = (e[k] > thr).astype(np.int8)
        au.append(float(roc_auc_score(lab, w[k])) if lab.any() and not lab.all() else np.nan)
    return float(np.mean(rs)), float(np.nanmean(au))


class _PointHead(torch.nn.Module):
    """Exposes just the median (CQR) or mean (Gaussian) channel, so the point
    prediction can be scored with exactly the same metrics as the baselines."""

    def __init__(self, net, ch):
        super().__init__(); self.net, self.ch = net, ch

    def forward(self, x):
        return self.net(x)[:, self.ch:self.ch + 1]

    def eval(self): self.net.eval(); return self
    def train(self, m=True): self.net.train(m); return self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--mode", default="cqr", choices=["cqr", "gauss"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--base", type=int, default=32)
    a = ap.parse_args()

    net = train_quantile(a.contrast, a.alpha, a.steps, base=a.base, mode=a.mode)
    coh, tr, val = D.dev_split(a.contrast)
    test = D.PairedCohort("test", a.contrast)
    cal = collect(net, coh, val, a.mode, a.alpha)          # 8 held-out training subjects
    te = collect(net, test, test.subjects, a.mode, a.alpha)  # locked 10

    res = {"contrast": a.contrast, "mode": a.mode, "alpha": a.alpha}

    # point-prediction quality of the median / mean head, scored exactly like the baselines
    head = _PointHead(net, 1 if a.mode == "cqr" else 0)
    in_fn = lambda rows: (test.ulf[rows].astype(np.float32),
                          test.ulf[rows][:, 1:2].astype(np.float32),
                          test.hf[rows].astype(np.float32))
    prows = T.evaluate(head, test, test.subjects, in_fn)
    res["point"] = {"agg": M.aggregate(prows), "rows": prows}
    pg = res["point"]["agg"]
    print(f"[point] masked PSNR {pg['psnr'][0]:.3f}±{pg['psnr'][1]:.3f}  "
          f"SSIM {pg['ssim'][0]:.4f}±{pg['ssim'][1]:.4f}  cosLL {pg['cos_LL'][0]:.3f}")

    for nm in (["hetero", "const"] if a.mode == "cqr" else ["hetero"]):
        Q = cqr_calibrate(cal, a.alpha, nm)
        rows = eval_intervals(te, Q, a.alpha, nm)
        res[nm] = {"Q": Q, "agg": M.aggregate(rows), "rows": rows}
        g = res[nm]["agg"]
        print(f"[{a.mode}/{nm}] Q={Q:.4f}  test coverage {g['coverage'][0]:.4f}+-{g['coverage'][1]:.4f} "
              f"(target {1-a.alpha:.2f})  mean width {g['width'][0]:.4f}")

    if a.mode == "cqr":
        res["crc"] = {}
        for al in sorted({a.alpha, round(1.0 / (len(cal) + 1) + 0.02, 3), 0.2}):
            lam, ok, floor = crc_lambda(cal, al, np.linspace(0, 0.6, 241))
            R = [float(((y < lo - lam) | (y > hi + lam)).mean()) for lo, hi, pt, y in te.values()]
            res["crc"][str(al)] = {"lambda": lam, "feasible": ok, "alpha_floor": floor,
                                   "test_risk_mean": float(np.mean(R)),
                                   "test_risk_max": float(np.max(R)), "per_subject": R}
            print(f"[CRC alpha={al}] {'OK' if ok else 'INFEASIBLE (floor %.3f)' % floor}  "
                  f"lambda={lam:.4f}  mean per-subject miscoverage {np.mean(R):.4f}  "
                  f"worst subject {np.max(R):.4f}")

        rho, auc = width_quality(te)
        res["width_quality"] = {"spearman": rho, "auroc_top10pct_error": auc}
        print(f"[width] spearman(width,|err|)={rho:.3f}  AUROC(top-10% error)={auc:.3f}")

        # how many calibration subjects are enough? pool the 18 non-training subjects,
        # resample the split, and look at the spread of realised coverage.
        pool = {**cal, **te}
        keys = sorted(pool); rng = np.random.default_rng(0)
        ss = {}
        for k in [2, 4, 6, 8, 10, 12]:
            covs, split_means = [], []
            for _ in range(200):
                perm = rng.permutation(keys)
                c = {s: pool[s] for s in perm[:k]}
                t = {s: pool[s] for s in perm[k:]}
                Qk = cqr_calibrate(c, a.alpha, "hetero")
                cv = [r["coverage"] for r in eval_intervals(t, Qk, a.alpha, "hetero")]
                covs += cv; split_means.append(float(np.mean(cv)))
            ss[k] = dict(mean=float(np.mean(covs)), sd=float(np.std(covs)),
                         p05=float(np.quantile(covs, 0.05)),
                         frac_below=float(np.mean(np.array(covs) < 1 - a.alpha)),
                         sd_split_mean=float(np.std(split_means)))
            print(f"  n_cal={k:2d}: per-subject coverage {ss[k]['mean']:.4f} +- {ss[k]['sd']:.4f}  "
                  f"5th pct {ss[k]['p05']:.4f}  P(cov<{1-a.alpha:.2f})={ss[k]['frac_below']:.3f}  "
                  f"| SD of split mean (calibration noise only) {ss[k]['sd_split_mean']:.4f}")
        res["calibration_size"] = ss

    json.dump(res, open(f"{OUT}/d3_{a.mode}_{a.contrast}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
