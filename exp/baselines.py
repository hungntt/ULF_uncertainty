"""Direction-0 benchmark: identity, U-Net, pix2pix, LoHiResGAN on single contrast."""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T

OUT = "/mnt/G/Projects/ULF-MICCAI/out"


def make_inputs(cohort):
    def in_fn(rows):
        return (cohort.ulf[rows].astype(np.float32),
                cohort.ulf[rows][:, 1:2].astype(np.float32),
                cohort.hf[rows].astype(np.float32))
    return in_fn


class MismatchSampler:
    """ULF from one subject, 3T from a *different* subject at the same relative
    depth. Keeps the marginals, the anatomy class and the axial position, and
    destroys only the subject correspondence. If the ULF->3T correction were
    mostly a population-level effect, this would cost little."""

    def __init__(self, cohort, sids, bs=16, seed=0):
        self.c, self.bs = cohort, bs
        self.rng = np.random.default_rng(seed)
        self.by_sub, self.z = {}, {}
        for s in cohort.man:
            if s["subject_id"] not in sids: continue
            r = np.arange(s["slice_start"], s["slice_start"] + s["slice_count"])
            self.by_sub[s["subject_id"]] = r
            self.z[s["subject_id"]] = np.arange(len(r)) / len(r)
        self.sids = sorted(self.by_sub)

    def __call__(self):
        xi, yi = [], []
        for _ in range(self.bs):
            a, b = self.rng.choice(len(self.sids), 2, replace=False)
            a, b = self.sids[a], self.sids[b]
            k = self.rng.integers(len(self.by_sub[a]))
            xi.append(self.by_sub[a][k])
            j = int(np.argmin(np.abs(self.z[b] - self.z[a][k])))
            yi.append(self.by_sub[b][j])
        x = torch.from_numpy(self.c.ulf[np.array(xi)].astype(np.float32)).cuda()
        y = torch.from_numpy(self.c.hf[np.array(yi)].astype(np.float32)).cuda()
        if self.rng.random() < 0.5: x, y = torch.flip(x, [-1]), torch.flip(y, [-1])
        return x, y


def identity_rows(cohort, sids):
    out = []
    for sid in sids:
        idx = cohort.subject_rows(sid)
        u = torch.from_numpy(cohort.ulf[idx][:, 1:2].astype(np.float32)).cuda()
        h = torch.from_numpy(cohort.hf[idx].astype(np.float32)).cuda()
        r = {"sid": sid}
        r.update(M.volume_metrics(u, h, u))
        r.update(M.band_metrics(u, h, u))
        r["hfpow"] = M.hf_power_error(u, h)
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--method", default="unet")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    T.seed_all(a.seed)

    coh, tr_ids, val_ids = D.dev_split(a.contrast)
    test = D.PairedCohort("test", a.contrast)
    in_tr, in_te = make_inputs(coh), make_inputs(test)
    print(f"[{a.contrast}] train {len(tr_ids)} subj / val {len(val_ids)} subj / test {len(test.subjects)} subj")

    if a.method == "identity":
        res = {"dev": M.aggregate(identity_rows(coh, val_ids)),
               "test": M.aggregate(identity_rows(test, test.subjects)),
               "test_rows": identity_rows(test, test.subjects)}
        json.dump(res, open(f"{OUT}/base_identity_{a.contrast}.json", "w"), indent=1, default=float)
        print(json.dumps(res["test"], indent=1, default=float)); return

    disc, l_ssim, l_l1, l_adv, lr = None, 0.0, 1.0, 0.0, a.lr
    if a.method == "unet":
        net = Mo.UNet(3, 1, base=a.base, residual_from=(1, 2)).cuda()
    elif a.method == "unet_ssim":
        net = Mo.UNet(3, 1, base=a.base, residual_from=(1, 2)).cuda(); l_ssim = 0.2
    elif a.method == "unet_noresid":
        # same trunk, but predicts the absolute image like LoHiResGAN does, so the
        # residual parameterisation can be priced separately from the backbone.
        net = Mo.UNet(3, 1, base=a.base, residual_from=None).cuda()
    elif a.method == "unet_mismatched":
        net = Mo.UNet(3, 1, base=a.base, residual_from=(1, 2)).cuda()
    elif a.method == "pix2pix":
        net = Mo.UNet(3, 1, base=a.base, residual_from=(1, 2)).cuda()
        disc = Mo.PatchGAN(4).cuda(); l_adv = 0.01
    elif a.method == "lohiresgan":
        net = Mo.LoHiResGen(3, 1).cuda(); disc = Mo.PatchGAN(4).cuda()
        # paper objective is L_adv + 100*L1 + 1*SSIM; divided through by 100
        l_l1, l_ssim, l_adv = 1.0, 0.01, 0.01
    else:
        raise SystemExit("unknown method")

    # seed the SAMPLER too, not just the global RNG: with a hardcoded sampler seed a
    # "different seed" changes only the weight init and leaves batch order identical.
    sampler = (MismatchSampler(coh, tr_ids, bs=a.bs, seed=a.seed) if a.method == "unet_mismatched"
               else T.Sampler(coh.ulf, coh.hf, coh.rows_for(tr_ids), bs=a.bs, seed=a.seed))

    def val_fn(model):
        r = M.aggregate(T.evaluate(model, coh, val_ids, in_tr))
        return {"score": r["ssim"][0], "ssim": r["ssim"][0], "psnr": r["psnr"][0],
                "cosLL": r["cos_LL"][0], "rLL": r["r_LL"][0]}

    info = T.train_regression(net, sampler, a.steps, lr=lr, l_ssim=l_ssim, l_l1=l_l1,
                              disc=disc, l_adv=l_adv, log=print, val_fn=val_fn, val_every=1000)
    rows = T.evaluate(net, test, test.subjects, in_te)
    agg = M.aggregate(rows)
    tag = a.tag or a.method
    json.dump({"cfg": vars(a), "info": {k: v for k, v in info.items() if k != "hist"},
               "dev": info["best"], "test": agg, "test_rows": rows},
              open(f"{OUT}/base_{tag}_{a.contrast}.json", "w"), indent=1, default=float)
    torch.save(net.state_dict(), f"{OUT}/base_{tag}_{a.contrast}.pt")
    print(f"\n== TEST [{tag} {a.contrast}] ==")
    for k in ["psnr", "ssim", "psnr_full", "ssim_full", "nrmse", "cos_LL", "r_LL", "cos_HH", "r_HH", "hfpow"]:
        print(f"  {k:10s} {agg[k][0]:.4f} +- {agg[k][1]:.4f}")


if __name__ == "__main__":
    main()
