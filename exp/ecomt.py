"""ECO-MT: evidence-calibrated, observability-aware Mean Teacher for 64 mT -> 3T.

The claim under test is narrow. Mean Teacher on unpaired ULF can only propagate
what the paired teacher already knows, and Finding A says the cross-subject part
of the correction is almost nothing (LL cosine 0.09). So the question is not
"does unlabelled data add information" - it cannot - but "does gating the
teacher's pseudo-target by out-of-fold error evidence and by what the 64 mT scan
actually supports make the consistency term a better regulariser than trusting
the teacher everywhere".

Three stages:
  calib   inner 3-fold cross-fitted warm-ups -> out-of-fold teacher errors ->
          the risk calibrator r_hat_l(u,d,s), the observability map o_l, the band
          normalisers and the tolerances tau_l. Also answers H1 on held-out
          subjects, which needs no semi-supervised run at all.
  train   the baseline ladder B2..ECO-MT, every arm identical to the paired-only
          control except for the term being tested.
  report  the locked 10-subject table with subject-level Wilcoxon and bootstrap.

B1 (the paired-only control) is base_unet_noresid_{contrast}: the same backbone,
sampler seed, optimiser, schedule, step count and model selection as every arm
here, with the semi-supervised term deleted. It is the stronger of the two
supervised U-Nets in this study, which is the one a gain has to clear.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, eco as E, metrics as M, models as Mo, train as T

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
NSC = 3                                   # pyramid levels: fine / medium / coarse
SCALES = ["fine", "med", "coarse"]


def enhancer(base=32):
    return Mo.UNet(3, 1, base=base, residual_from=None).cuda()


def make_inputs(cohort):
    def in_fn(rows):
        return (cohort.ulf[rows].astype(np.float32),
                cohort.ulf[rows][:, 1:2].astype(np.float32),
                cohort.hf[rows].astype(np.float32))
    return in_fn


# ------------------------------------------------------------------ stage: calib

@torch.no_grad()
def _evidence(teacher, ulf, rng, two_views=True):
    """Teacher prediction and the (u,d,s) evidence for one batch of ULF slabs."""
    x1 = E.weak(ulf, rng)
    p1 = teacher(x1)
    p2 = teacher(E.weak(ulf, rng)) if two_views else None
    c = ulf[:, 1:2]
    mask = (c > 0).float()
    return p1, p2, E.features(p1, p2, c, mask, NSC), mask


@torch.no_grad()
def _true_err(pred, hf, ulf_c):
    """e_l: mean |P_l(pred) - P_l(3T)| inside the brain, per scale."""
    m = ((ulf_c > 0) | (hf > 0)).float()
    pm = E.pyr_pool(m, NSC)
    pp, ph = E.lap_pyr(pred, NSC), E.lap_pyr(hf, NSC)
    return torch.stack([((pp[l] - ph[l]).abs() * pm[l]).sum((1, 2, 3)) /
                        pm[l].sum((1, 2, 3)).clamp(min=1) for l in range(NSC)], 1)


def _gather(teacher, cohort, sids, rng, bs=16):
    """Per-slice evidence, true error and subject id for a set of subjects."""
    X, Y, S = [], [], []
    for sid in sids:
        idx = cohort.subject_rows(sid)
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            u = torch.from_numpy(cohort.ulf[j].astype(np.float32)).cuda()
            h = torch.from_numpy(cohort.hf[j].astype(np.float32)).cuda()
            p, _, f, _ = _evidence(teacher, u, rng)
            X.append(np.stack([[f[l][k].cpu().numpy() for k in E.FEATS] for l in range(NSC)], 1).T)
            Y.append(_true_err(p, h, u[:, 1:2]).cpu().numpy())
            S += [sid] * len(j)
    return np.concatenate(X), np.concatenate(Y), np.array(S)


def _warmup(coh, fit_ids, steps, seed, bs=16):
    """Supervised warm-up with an EMA teacher; returns the teacher."""
    T.seed_all(seed)
    net = enhancer()
    ema = E.EMA(net, warm=1000)
    smp = T.Sampler(coh.ulf, coh.hf, coh.rows_for(fit_ids), bs=bs, seed=seed)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=steps,
                                              pct_start=max(0.05, 3.0 / steps))
    for it in range(steps):
        x, y = smp()
        m = (y > 0).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = net(x)
            l1 = (p - y).abs()
            loss = l1.mean() + (l1 * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sch.step(); ema.update(net, it)
    return ema, net


def aurc(score, err, grid=None):
    """Risk-coverage area: accept the lowest-score fraction c, measure mean true
    error. Normalised by the mean error, so 1.0 is an uninformative score."""
    grid = grid if grid is not None else np.arange(0.05, 1.001, 0.05)
    o = np.argsort(score)
    e = err[o]
    r = [e[:max(1, int(round(c * len(e))))].mean() for c in grid]
    return float(np.mean(r) / err.mean())


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def h1_table(X, Y, S, risk, boot=2000, seed=0):
    """H1: does r_hat rank true out-of-fold teacher error better than the
    variance-only confidence? Subject-level bootstrap, because slices within one
    person are not independent."""
    rh = risk(X)
    rng = np.random.default_rng(seed)
    subs = np.unique(S)
    rows = []
    for l in range(NSC):
        cand = {"r_hat": rh[:, l], "u (teacher variance)": X[:, l, 0],
                "d (MIND disagreement)": X[:, l, 1], "s (ULF band energy)": X[:, l, 2]}
        for nm, sc in cand.items():
            bs_sp, bs_ar = [], []
            for _ in range(boot):
                pick = rng.choice(subs, len(subs), replace=True)
                m = np.concatenate([np.where(S == p)[0] for p in pick])
                bs_sp.append(spearman(sc[m], Y[m, l]))
                bs_ar.append(aurc(sc[m], Y[m, l]))
            rows.append({"scale": SCALES[l], "score": nm,
                         "spearman": spearman(sc, Y[:, l]),
                         "spearman_lo": float(np.percentile(bs_sp, 2.5)),
                         "spearman_hi": float(np.percentile(bs_sp, 97.5)),
                         "aurc": aurc(sc, Y[:, l]),
                         "aurc_lo": float(np.percentile(bs_ar, 2.5)),
                         "aurc_hi": float(np.percentile(bs_ar, 97.5))})
    return rows


def _cfg(a):
    return {k: v for k, v in vars(a).items() if k not in ("fn", "cmd")}


def cmd_calib(a):
    coh, tr, val = D.dev_split(a.contrast)
    tr, test = sorted(tr), D.PairedCohort("test", a.contrast)
    rng = np.random.default_rng(0)
    print(f"[calib {a.contrast}] {len(tr)} train / {len(val)} dev-val / "
          f"{len(test.subjects)} locked test", flush=True)

    # --- inner 3-fold: every training subject gets a teacher that never saw it
    X, Y, S = [], [], []
    for f in range(a.folds):
        hold = tr[f::a.folds]
        fit = [s for s in tr if s not in hold]
        print(f"  [fold {f}] warm-up on {len(fit)}, evidence on {len(hold)}", flush=True)
        ema, net = _warmup(coh, fit, a.steps, seed=300 + f, bs=a.bs)
        x, y, s = _gather(ema, coh, hold, rng)
        X.append(x); Y.append(y); S.append(s)
        del net, ema; torch.cuda.empty_cache()
    X, Y, S = np.concatenate(X), np.concatenate(Y), np.concatenate(S)
    print(f"  out-of-fold evidence: {len(X)} slices, {len(np.unique(S))} subjects", flush=True)

    risk = E.Risk.fit(X, Y, alpha=a.ridge)
    tau_oof = (2.0 * np.median(Y, 0)).tolist()             # w = 0.5 at the median OOF error
    tau_u_oof = (2.0 * np.median(X[:, :, 0], 0)).tolist()

    # --- observability and band normalisers, from the training pairs only
    r = coh.rows_for(tr)
    r = r[np.linspace(0, len(r) - 1, min(len(r), 1600)).astype(int)]

    def chunks():
        for i in range(0, len(r), 32):
            j = r[i:i + 32]
            yield (torch.from_numpy(coh.ulf[j][:, 1:2].astype(np.float32)).cuda(),
                   torch.from_numpy(coh.hf[j].astype(np.float32)).cuda())

    obs = E.Observability.fit(chunks(), NSC, win=a.win)
    acc, cnt = np.zeros(NSC), np.zeros(NSC)
    for u, h in chunks():
        m = ((u > 0) | (h > 0)).float()
        pm, ph = E.pyr_pool(m, NSC), E.lap_pyr(h, NSC)
        for l in range(NSC):
            acc[l] += float((ph[l].abs() * pm[l]).sum()); cnt[l] += float(pm[l].sum())
    nrm = (acc / np.maximum(cnt, 1)).tolist()
    torch.cuda.empty_cache()
    print(f"  band normalisers {['%.4f' % v for v in nrm]}", flush=True)

    # --- the teacher the ladder actually starts from
    print("  operational teacher: warm-up on all 32", flush=True)
    ema, net = _warmup(coh, tr, a.steps, seed=300 + a.folds, bs=a.bs)
    torch.save(net.state_dict(), f"{OUT}/eco_warm_{a.contrast}.pt")

    # --- tolerances, matched on the unpaired pool.
    # L_semi is a normalised weighted mean, so scaling w uniformly changes
    # nothing; only the shape of w matters. Fixing every scalar gate to the same
    # mean weight on the unlabelled data therefore costs no generality and stops
    # the comparison turning into "whose tolerance happened to be better set".
    # The out-of-fold tolerance is kept for the record: on unpaired input the raw
    # teacher variance sits *above* its paired-data median, so the literal
    # variance gate would silence the fine band almost completely.
    unp = D.unpaired_ulf(a.contrast)
    pk = np.sort(np.random.default_rng(7).choice(len(unp), min(512, len(unp)), replace=False))
    U, R = [], []
    for i in range(0, len(pk), 16):
        xu = torch.from_numpy(unp[pk[i:i + 16]].astype(np.float32)).cuda()
        _, _, f, _ = _evidence(ema, xu, rng)
        U.append(torch.stack([f[l]["u"] for l in range(NSC)], 1).cpu().numpy())
        R.append(risk.torch_apply(f).cpu().numpy())
    U, R = np.concatenate(U), np.concatenate(R)

    def solve(sc, target=0.5):
        out = []
        for l in range(sc.shape[1]):
            lo, hi = 1e-10, float(np.abs(sc[:, l]).max()) * 100 + 1e-6
            for _ in range(80):
                mid = (lo + hi) / 2
                if np.clip(1 - sc[:, l] / mid, 0, 1).mean() < target:
                    lo = mid
                else:
                    hi = mid
            out.append(float((lo + hi) / 2))
        return out

    tau, tau_u = solve(R), solve(U)
    print(f"  tau (risk)     unpaired-matched {['%.4f' % v for v in tau]}"
          f"   out-of-fold {['%.4f' % v for v in tau_oof]}", flush=True)
    print(f"  tau (variance) unpaired-matched {['%.6f' % v for v in tau_u]}"
          f"   out-of-fold {['%.6f' % v for v in tau_u_oof]}", flush=True)

    # --- H1 on subjects no fold model and no calibrator ever saw
    h1 = {}
    for nm, (c, ids) in {"dev_val": (coh, val), "test": (test, test.subjects)}.items():
        x, y, s = _gather(ema, c, ids, rng)
        h1[nm] = h1_table(x, y, s, risk)
        print(f"  H1 [{nm}]  " + "  ".join(
            f"{SCALES[l]}: AURC r_hat {[r_['aurc'] for r_ in h1[nm] if r_['scale']==SCALES[l] and r_['score']=='r_hat'][0]:.3f}"
            f" vs u {[r_['aurc'] for r_ in h1[nm] if r_['scale']==SCALES[l] and r_['score'].startswith('u ')][0]:.3f}"
            for l in range(NSC)), flush=True)
    del net, ema; torch.cuda.empty_cache()

    json.dump({"cfg": _cfg(a), "risk": risk.state(), "obs": obs.state(), "tau": tau,
               "tau_u": tau_u, "tau_oof": tau_oof, "tau_u_oof": tau_u_oof,
               "nrm": nrm, "oof_err_median": np.median(Y, 0).tolist(),
               "oof_n": int(len(X)), "h1": h1},
              open(f"{OUT}/eco_calib_{a.contrast}.json", "w"), indent=1, default=float)
    print(f"\nwrote {OUT}/eco_calib_{a.contrast}.json", flush=True)


# ------------------------------------------------------------------ stage: train

class UnpairedSampler:
    def __init__(self, arr, bs=8, seed=0):
        self.a, self.bs = arr, bs
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        r = np.sort(self.rng.choice(len(self.a), self.bs, replace=False))
        return torch.from_numpy(self.a[r].astype(np.float32)).cuda()


VOX = ("b5vox", "ecomtvox")


def gate(mode, feats, obs, risk, tau, tau_u, ulf_c, rng, riskv=None, tauv=None,
         p1=None, p2=None):
    """(scalar per sample+scale, spatial map per scale). Never differentiable."""
    B = ulf_c.shape[0]
    one = torch.ones(B, NSC, device=ulf_c.device)
    sp = [torch.ones_like(p) for p in E.pyr_pool(torch.ones_like(ulf_c), NSC)]
    if mode in ("b5", "ecomt"):
        rh = risk.torch_apply(feats, ulf_c.device)
        one = (1 - rh / torch.tensor(tau, device=ulf_c.device)).clamp(0, 1)
    elif mode == "b3":
        u = torch.stack([feats[l]["u"] for l in range(NSC)], 1)
        one = (1 - u / torch.tensor(tau_u, device=ulf_c.device)).clamp(0, 1)
    elif mode == "b2rand":
        one = torch.from_numpy(rng.random((B, NSC)).astype(np.float32)).cuda()
    if mode in VOX:
        rm = riskv.apply_sp(E.features_sp(p1, p2, ulf_c, NSC))
        tv = [float(t) for t in tauv]
        sp = [(1 - rm[l] / tv[l]).clamp(0, 1) for l in range(NSC)]
    if mode in ("b6", "ecomt", "ecomtvox"):
        o = obs(ulf_c)
        sp = [sp[l] * o[l] for l in range(NSC)]
    return one.detach(), [s.detach() for s in sp]


def cmd_train(a):
    cal = json.load(open(f"{OUT}/eco_calib_{a.contrast}.json"))
    risk, obs = E.Risk.load(cal["risk"]), E.Observability.load(cal["obs"])
    tau, tau_u, nrm = cal["tau"], cal["tau_u"], cal["nrm"]
    riskv = tauv = None
    if a.mode in VOX:
        c2 = json.load(open(f"{OUT}/eco_calib2_{a.contrast}.json"))
        riskv, tauv = E.Risk.load(c2["risk_vox"]), c2["tau_vox"]

    coh, tr, val = D.dev_split(a.contrast)
    test = D.PairedCohort("test", a.contrast)
    in_tr, in_te = make_inputs(coh), make_inputs(test)
    unp = D.unpaired_ulf(a.contrast)
    print(f"[{a.mode} {a.contrast}] {len(tr)} paired train / {len(val)} dev-val / "
          f"{len(test.subjects)} test / {len(unp)} unpaired ULF slices", flush=True)

    if a.mode == "b1":
        a.lam = 0.0                       # the paired-only control, same code path
    T.seed_all(a.seed)
    net = enhancer(a.base)
    ema = E.EMA(net, warm=1000)
    smp = T.Sampler(coh.ulf, coh.hf, coh.rows_for(tr), bs=a.bs, seed=a.seed)
    usmp = UnpairedSampler(unp, bs=a.ubs, seed=a.seed + 1)
    rng = np.random.default_rng(a.seed + 2)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                              pct_start=max(0.05, 3.0 / a.steps))
    nrm_t = torch.tensor(nrm, device="cuda")

    def val_fn(model):
        r = M.aggregate(T.evaluate(model, coh, val, in_tr))
        return {"score": r["ssim"][0], "ssim": r["ssim"][0], "psnr": r["psnr"][0],
                "cosLL": r["cos_LL"][0]}

    best, hist, gstat, t0 = None, [], [], time.time()
    for it in range(a.steps):
        lam = a.lam * min(1.0, max(0.0, (it - a.warm) / max(1, a.ramp))) ** 2
        x, y = smp()
        m = (y > 0).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = net(x)
            l1 = (p - y).abs()
            loss = l1.mean() + (l1 * m).sum() / m.sum().clamp(min=1)

        if lam > 0:
            xu = usmp()
            c = xu[:, 1:2]
            need_u = a.mode in ("b3", "b5", "ecomt") + VOX
            with torch.no_grad():
                tgt, tgt2, feats, umask = _evidence(ema, xu, rng, two_views=need_u)
            g1, gsp = gate(a.mode, feats, obs, risk, tau, tau_u, c, rng,
                           riskv, tauv, tgt, tgt2)
            xs, th = E.strong(xu, rng)
            valid = (E.warp(torch.ones_like(c), th) > 0.99).float()
            tgt_w = E.warp(tgt, th)
            mk = E.warp(umask, th) * valid
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ps = net(xs)
            pp, pt = E.lap_pyr(ps.float(), NSC), E.lap_pyr(tgt_w, NSC)
            pmk, psp = E.pyr_pool(mk, NSC), [E.warp(s, th) for s in gsp]
            # per-band weighted mean, then equal weight across bands: a fine band
            # has 16x the pixels of the coarse one, so a single pooled mean would
            # make the coarse gate irrelevant and the per-scale trust idea moot
            per, wmean = [], []
            for l in range(NSC):
                w = pmk[l] * psp[l] * g1[:, l].view(-1, 1, 1, 1)
                ws = w.sum()
                per.append(((pp[l] - pt[l]).abs() * w).sum() / nrm_t[l] / ws.clamp(min=1e-6))
                wmean.append(float(ws / pmk[l].sum().clamp(min=1)))
            lsemi = torch.stack(per).mean()
            loss = loss + lam * lsemi
            if a.mode == "b4":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ds = (E.mind(F.avg_pool2d(ps.float(), 2)) -
                          E.mind(F.avg_pool2d(E.warp(c, th), 2))).abs()
                loss = loss + lam * a.lstruct * (ds * F.avg_pool2d(mk, 2)).sum() / \
                    F.avg_pool2d(mk, 2).sum().clamp(min=1) / ds.shape[1]
            if not gstat or it % 200 == 0:
                gstat.append([it, float(lsemi)] + wmean +
                             [float(g1[:, l].mean()) for l in range(NSC)])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sch.step(); ema.update(net, it)

        if (it + 1) % a.val_every == 0:
            v = val_fn(net)
            hist.append((it + 1, v))
            if best is None or v["score"] > best[1]["score"]:
                best = (it + 1, v, {k: t.detach().clone() for k, t in net.state_dict().items()})
            print(f"  step {it+1:6d}  loss {loss.item():.4f}  lam {lam:.3f}  "
                  + "  ".join(f"{k} {vv:.4f}" for k, vv in v.items())
                  + f"  [{time.time()-t0:.0f}s]", flush=True)

    if best is not None:
        net.load_state_dict(best[2])
    rows = T.evaluate(net, test, test.subjects, in_te)
    agg = M.aggregate(rows)
    trow = T.evaluate(ema.t, test, test.subjects, in_te)
    tag = a.tag or a.mode
    json.dump({"cfg": _cfg(a), "info": {"best_step": best[0] if best else a.steps, "secs": time.time() - t0},
               "dev": best[1] if best else None, "test": agg, "test_rows": rows,
               "teacher_test": M.aggregate(trow), "teacher_rows": trow,
               "gate": gstat, "hist": [(i, v) for i, v in hist]},
              open(f"{OUT}/eco_{tag}_{a.contrast}.json", "w"), indent=1, default=float)
    torch.save(net.state_dict(), f"{OUT}/eco_{tag}_{a.contrast}.pt")
    print(f"\n== TEST [{tag} {a.contrast}] best_step {best[0] if best else a.steps} ==", flush=True)
    for k in ["psnr", "ssim", "cos_LL", "r_LL", "cos_HH", "r_HH", "hfpow"]:
        print(f"  {k:10s} {agg[k][0]:.4f} +- {agg[k][1]:.4f}")
    print(f"  teacher    ssim {M.aggregate(trow)['ssim'][0]:.4f}  psnr {M.aggregate(trow)['psnr'][0]:.4f}")


# ------------------------------------------------------------------ stage: pick / report

def cmd_pick(a):
    """The lambda with the best dev-val masked SSIM. Printed bare so a shell can
    read it; the locked test subjects play no part in this choice."""
    import glob
    best = None
    for f in glob.glob(f"{OUT}/eco_b2_lam*_{a.contrast}.json"):
        d = json.load(open(f))
        v = d["dev"]["score"]
        if best is None or v > best[1]:
            best = (d["cfg"]["lam"], v)
    print(best[0] if best else 1.0)


def _delta(rows_a, rows_b, key, boot=10000, seed=0):
    """Paired subject-level difference a - b, with a bootstrap CI and Wilcoxon."""
    ia = {r["sid"]: r for r in rows_a}
    ib = {r["sid"]: r for r in rows_b}
    sid = sorted(set(ia) & set(ib))
    d = np.array([ia[k][key] - ib[k][key] for k in sid])
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (boot, len(d)))].mean(1)
    return (float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)),
            M.paired_test([ia[k][key] for k in sid], [ib[k][key] for k in sid]))


LADDER = [("B0  do nothing", "base_identity"), ("B1  paired-only U-Net", "eco_b1"),
          ("B1' published baseline", "base_unet_noresid"),
          ("B2  + vanilla Mean Teacher", "eco_b2"), ("B3  + teacher-variance gate", "eco_b3"),
          ("B4  + MIND structural term", "eco_b4"), ("B5  + calibrated risk gate", "eco_b5"),
          ("B6  + observability gate", "eco_b6"), ("ECO-MT  both gates (scalar risk)", "eco_ecomt"),
          ("B5v + voxel risk gate", "eco_b5vox"),
          ("ECO-MT-v  both gates (voxel risk)", "eco_ecomtvox"),
          ("ctrl  random gate", "eco_b2rand")]


def cmd_report(a):
    got = {}
    for nm, stem in LADDER:
        f = f"{OUT}/{stem}_{a.contrast}.json"
        if os.path.exists(f):
            got[nm] = json.load(open(f))
    b1 = got.get("B1  paired-only U-Net")
    b2 = got.get("B2  + vanilla Mean Teacher")
    keys = ["ssim", "psnr", "cos_LL", "cos_HH", "r_HH", "hfpow"]
    print(f"\n## Locked 10-subject test, {a.contrast}, brain-masked\n")
    print("| arm | " + " | ".join(keys) + " | dSSIM vs B1 [95% CI] | p | dSSIM vs B2 | p |")
    print("|" + "---|" * (len(keys) + 5))
    for nm, _ in LADDER:
        d = got.get(nm)
        if d is None:
            continue
        g = d["test"]
        cells = [f"{g[k][0]:.4f}" for k in keys]
        r = d["test_rows"]
        c1 = c2 = "- | -"
        if b1 and d is not b1:
            m, lo, hi, p = _delta(r, b1["test_rows"], "ssim")
            c1 = f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] | {p:.4f}"
        if b2 and d is not b2 and d is not b1:
            m, lo, hi, p = _delta(r, b2["test_rows"], "ssim")
            c2 = f"{m:+.4f} | {p:.4f}"
        print(f"| {nm} | " + " | ".join(cells) + f" | {c1} | {c2} |")

    print("\n## Prespecified anatomy-fidelity endpoint: cos_LL, non-inferiority margin -0.010\n")
    for nm, _ in LADDER:
        d = got.get(nm)
        if d is None or not b1 or d is b1 or nm.startswith("B0"):
            continue
        m, lo, hi, p = _delta(d["test_rows"], b1["test_rows"], "cos_LL")
        print(f"  {nm:30s} dcos_LL {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"{'PASS' if lo > -0.010 else 'FAIL'}")

    print("\n## Student vs its own EMA teacher (masked SSIM)\n")
    for nm, _ in LADDER:
        d = got.get(nm)
        if d is None or "teacher_test" not in d:
            continue
        print(f"  {nm:30s} student {d['test']['ssim'][0]:.4f}   teacher {d['teacher_test']['ssim'][0]:.4f}"
              f"   best_step {d['info']['best_step']}")

    print("\n## Mean gate weight on the unpaired pool (last recorded step)\n")
    print(f"  {'arm':30s} {'L_semi':>8s}   effective w per scale (fine/med/coarse)")
    for nm, _ in LADDER:
        d = got.get(nm)
        if d is None or not d.get("gate"):
            continue
        r = d["gate"][-1]
        print(f"  {nm:30s} {r[1]:8.4f}   " + "  ".join(f"{v:.3f}" for v in r[2:5]))

    cf = f"{OUT}/eco_calib_{a.contrast}.json"
    if os.path.exists(cf):
        h1 = json.load(open(cf))["h1"]
        for split in h1:
            print(f"\n## H1, {split}: does r_hat rank true teacher error better than variance alone?\n")
            print("| scale | score | Spearman [95% CI] | AURC [95% CI] |")
            print("|---|---|---|---|")
            for r in h1[split]:
                print(f"| {r['scale']} | {r['score']} | {r['spearman']:.3f} "
                      f"[{r['spearman_lo']:.3f}, {r['spearman_hi']:.3f}] | {r['aurc']:.3f} "
                      f"[{r['aurc_lo']:.3f}, {r['aurc_hi']:.3f}] |")


# ------------------------------------------------------------------ main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calib")
    c.add_argument("--contrast", default="T2")
    c.add_argument("--folds", type=int, default=3)
    c.add_argument("--steps", type=int, default=6000)
    c.add_argument("--bs", type=int, default=16)
    c.add_argument("--ridge", type=float, default=1.0)
    c.add_argument("--win", type=int, default=9)
    c.set_defaults(fn=cmd_calib)

    t = sub.add_parser("train")
    t.add_argument("--contrast", default="T2")
    t.add_argument("--mode", default="b2",
                   choices=["b1", "b2", "b3", "b4", "b5", "b6", "ecomt",
                            "b2rand", "b5vox", "ecomtvox"])
    t.add_argument("--steps", type=int, default=12000)
    t.add_argument("--bs", type=int, default=16)
    t.add_argument("--ubs", type=int, default=8)
    t.add_argument("--base", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--lam", type=float, default=1.0)
    t.add_argument("--warm", type=int, default=1000)
    t.add_argument("--ramp", type=int, default=500)
    t.add_argument("--val_every", type=int, default=500)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lstruct", type=float, default=1.0)
    t.add_argument("--tag", default="")
    t.set_defaults(fn=cmd_train)

    k = sub.add_parser("pick")
    k.add_argument("--contrast", default="T2")
    k.set_defaults(fn=cmd_pick)

    r = sub.add_parser("report")
    r.add_argument("--contrast", default="T2")
    r.set_defaults(fn=cmd_report)

    a = ap.parse_args()
    a.fn(a)
