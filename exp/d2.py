"""Direction 2 — asymmetric cross-contrast fusion for ULF enhancement.

At 64 mT the two routine contrasts do not degrade equally. T1 loses GM/WM
separation (T1 relaxation times converge at low field), while T2 keeps tissue
contrast and loses resolution. The two ULF images are therefore two differently
corrupted views of one anatomy, and the useful information flows mostly one way.

This script measures that asymmetry under a controlled protocol (identical slice
subset, identical budget) and trains one availability-conditioned network that
serves T1-only, T2-only and both-contrast deployment.
"""
import argparse, json, sys
import numpy as np, torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
# input layout: [T1 slab(3) | T2 slab(3) | avail_T1 | avail_T2]
CENTRES = [1, 4]


class MCSampler:
    """Batches of co-registered T1/T2 slabs with optional modality dropout."""

    def __init__(self, mc, rows, bs=16, drop=False, seed=0, use=("T1", "T2"),
                 out_sel=(0, 1)):
        self.mc, self.rows, self.bs, self.drop, self.use = mc, rows, bs, drop, use
        self.out_sel = list(out_sel)      # target channels this head predicts
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        p = np.sort(self.rng.choice(len(self.rows), self.bs, replace=False))
        pos = self.rows[p]
        x = torch.from_numpy(self.mc.slab(pos).astype(np.float32)).cuda()
        _, hf = self.mc.batch(pos)
        y = torch.from_numpy(hf[:, self.out_sel].astype(np.float32)).cuda()
        a = torch.ones(len(pos), 2, 256, 256, device="cuda")
        if self.drop:
            r = self.rng.random(len(pos))
            a[torch.from_numpy(r < 1 / 3).cuda(), 0] = 0        # T1 missing
            a[torch.from_numpy((r >= 1 / 3) & (r < 2 / 3)).cuda(), 1] = 0   # T2 missing
        else:
            if "T1" not in self.use: a[:, 0] = 0
            if "T2" not in self.use: a[:, 1] = 0
        x[:, 0:3] *= a[:, 0:1]
        x[:, 3:6] *= a[:, 1:2]
        if self.rng.random() < 0.5:
            x, y, a = torch.flip(x, [-1]), torch.flip(y, [-1]), torch.flip(a, [-1])
        return torch.cat([x, a], 1), y


def mc_inputs(mc, avail=(1, 1)):
    def in_fn(pos):
        x = mc.slab(pos).astype(np.float32)
        ulf, hf = mc.batch(pos)
        a = np.ones((len(pos), 2, 256, 256), np.float32)
        a[:, 0] *= avail[0]; a[:, 1] *= avail[1]
        x[:, 0:3] *= a[:, 0:1]; x[:, 3:6] *= a[:, 1:2]
        return np.concatenate([x, a], 1), ulf.astype(np.float32), hf.astype(np.float32)
    return in_fn


class _SubjView:
    """Gives ulfkit.train.evaluate a subject_rows() over the intersected index."""
    def __init__(self, mc): self.mc = mc
    def subject_rows(self, sid): return np.where(self.mc.sid == sid)[0]


def run(a):
    T.seed_all(0)
    mc_tr = D.MultiContrastCohort("train")
    mc_te = D.MultiContrastCohort("test")
    tr_ids = [s for s in sorted(set(mc_tr.sid)) if s not in D.DEV_VAL_IDS]
    val_ids = [s for s in sorted(set(mc_tr.sid)) if s in D.DEV_VAL_IDS]
    print(f"multi-contrast index: {len(mc_tr.keys)} train slices / {len(mc_te.keys)} test slices")
    print(f"  {len(tr_ids)} train subj, {len(val_ids)} val subj, {len(set(mc_te.sid))} test subj")

    use = tuple(a.use.split(","))
    out_ch = 2 if a.out == "both" else 1
    out_sel = {"both": [0, 1], "T1": [0], "T2": [1]}[a.out]
    net = Mo.UNet(8, out_ch, base=a.base,
                  residual_from=[CENTRES[i] for i in out_sel]).cuda()

    smp = MCSampler(mc_tr, mc_tr.rows_for(tr_ids), bs=a.bs, drop=a.drop, seed=0,
                    use=use, out_sel=out_sel)
    view_tr, view_te = _SubjView(mc_tr), _SubjView(mc_te)

    def sub(fn):
        def g(pos):
            x, u, h = fn(pos)
            return x, u[:, out_sel], h[:, out_sel]
        return g

    def val_fn(model):
        g = M.aggregate(T.evaluate(model, view_tr, val_ids,
                                   sub(mc_inputs(mc_tr, (int("T1" in use), int("T2" in use)))),
                                   out_ch=out_ch))
        k = "ssim" if out_ch == 1 else "c0_ssim"
        score = g[k][0] if out_ch == 1 else (g["c0_ssim"][0] + g["c1_ssim"][0]) / 2
        return {"score": score, **{kk: g[kk][0] for kk in g if kk.endswith("ssim") or kk.endswith("psnr")}}

    info = T.train_regression(net, smp, a.steps, lr=2e-4, log=print, val_fn=val_fn, val_every=1000)

    results = {}
    # a model trained with one contrast visible must be tested the same way;
    # only the dropout model is meaningfully evaluated in all three regimes.
    regimes = [tuple(use)] if not a.drop else [("T1", "T2"), ("T1",), ("T2",)]
    for reg in regimes:
        rows = T.evaluate(net, view_te, sorted(set(mc_te.sid)),
                          sub(mc_inputs(mc_te, (int("T1" in reg), int("T2" in reg)))), out_ch=out_ch)
        results["+".join(reg)] = {"agg": M.aggregate(rows), "rows": rows}
        g = results["+".join(reg)]["agg"]
        print(f"\n== D2 TEST [in={'+'.join(reg)} out={a.out}] ==")
        for c, nm in enumerate(out_sel):
            pre = "" if out_ch == 1 else f"c{c}_"
            print(f"  {['T1','T2'][nm]}: PSNR {g[pre+'psnr'][0]:.3f}  SSIM {g[pre+'ssim'][0]:.4f}  "
                  f"cosLL {g[pre+'cos_LL'][0]:.3f}  rHH {g[pre+'r_HH'][0]:.3f}  hfpow {g[pre+'hfpow'][0]:.3f}")
    tag = a.tag or f"{a.use.replace(',','')}_to_{a.out}" + ("_drop" if a.drop else "")
    json.dump({"cfg": vars(a), "dev": info["best"], "results": results},
              open(f"{OUT}/d2_{tag}.json", "w"), indent=1, default=float)
    torch.save(net.state_dict(), f"{OUT}/d2_{tag}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--use", default="T1,T2")      # which ULF contrasts are visible
    ap.add_argument("--out", default="both", choices=["both", "T1", "T2"])
    ap.add_argument("--drop", action="store_true")  # train with modality dropout
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--tag", default=None)
    run(ap.parse_args())
