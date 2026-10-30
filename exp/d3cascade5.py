"""Does refinement help the *best* model, with both confounds removed?

Three earlier cascade runs disagreed because two things varied at once:
  * whether stage-2 saw residuals from the same-strength model it would correct
  * how many subjects stage-2 was fitted on (32 works, 8 overfits and harms)

The clean design needs held-out predictions from *32-subject* models across enough
subjects to fit a corrector. Rotate 5 folds over all 40 training subjects: each fold
trains on 32 and predicts the 8 it held out, so every subject gets a prediction from
a 32-subject model that never saw it. Stage-2 then fits on all 40.
"""
import json, sys
import numpy as np
import torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T
from ulfkit.metrics import paired_test

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
CACHE = "/mnt/G/Projects/ULF-MICCAI/datasets/npy"


def rotate(contrast, steps=9000, bs=16):
    coh = D.PairedCohort("train", contrast)
    ids = sorted(coh.subjects)                      # all 40
    oof = np.zeros_like(coh.hf)
    for f in range(5):
        hold = ids[f::5]
        fit = [s for s in ids if s not in hold]
        print(f"[rot {f}] fit {len(fit)} subj -> predict {len(hold)}", flush=True)
        T.seed_all(200 + f)
        net = Mo.UNet(3, 1, base=32, residual_from=None).cuda()
        T.train_regression(net, T.Sampler(coh.ulf, coh.hf, coh.rows_for(fit), bs=bs, seed=f),
                           steps, lr=2e-4, log=None)
        net.eval()
        with torch.no_grad():
            for sid in hold:
                idx = coh.subject_rows(sid)
                for i in range(0, len(idx), 16):
                    j = idx[i:i + 16]
                    x = torch.from_numpy(coh.ulf[j].astype(np.float32)).cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        oof[j] = net(x).float().clamp(0, 1).cpu().numpy().astype(oof.dtype)
        del net; torch.cuda.empty_cache()
    np.save(f"{CACHE}/cas5_oof_{contrast}.npy", oof)
    print("rotation done: every subject now has a 32-subject-model prediction", flush=True)


def fit_stage2(contrast, steps=9000, bs=16):
    T.seed_all(0)
    coh = D.PairedCohort("train", contrast)
    oof = np.load(f"{CACHE}/cas5_oof_{contrast}.npy")
    rows = np.arange(len(coh.hf))
    net = Mo.UNet(4, 1, base=32, residual_from=None).cuda()
    opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-4)
    lrs = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    for it in range(steps):
        j = rows[np.sort(rng.choice(len(rows), bs, replace=False))]
        x = torch.from_numpy(coh.ulf[j].astype(np.float32)).cuda()
        p = torch.from_numpy(oof[j].astype(np.float32)).cuda()
        y = torch.from_numpy(coh.hf[j].astype(np.float32)).cuda()
        if rng.random() < 0.5: x, p, y = (torch.flip(v, [-1]) for v in (x, p, y))
        m = (y > 0).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d = net(torch.cat([x, p], 1)).float()
        r = y - p
        loss = (d - r).abs().mean() + ((d - r).abs() * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); lrs.step()
        if (it + 1) % 3000 == 0:
            print(f"  step {it+1}  L1 {loss.item():.6f}", flush=True)
    torch.save(net.state_dict(), f"{OUT}/d3cascade5_{contrast}.pt")


@torch.no_grad()
def evaluate(contrast):
    test = D.PairedCohort("test", contrast)
    s1 = Mo.UNet(3, 1, base=32, residual_from=None).cuda()
    s1.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_{contrast}.pt")); s1.eval()
    s2 = Mo.UNet(4, 1, base=32, residual_from=None).cuda()
    s2.load_state_dict(torch.load(f"{OUT}/d3cascade5_{contrast}.pt")); s2.eval()
    rows = []
    for sid in test.subjects:
        idx = test.subject_rows(sid); P1, P2, Y, U = [], [], [], []
        for i in range(0, len(idx), 16):
            j = idx[i:i + 16]
            x = torch.from_numpy(test.ulf[j].astype(np.float32)).cuda()
            y = torch.from_numpy(test.hf[j].astype(np.float32)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p1 = s1(x).float().clamp(0, 1)
                p2 = (p1 + s2(torch.cat([x, p1], 1)).float()).clamp(0, 1)
            P1.append(p1); P2.append(p2); Y.append(y); U.append(x[:, 1:2])
        p1, p2, y, u = (torch.cat(v) for v in (P1, P2, Y, U))
        r = {"sid": sid}
        for nm, p in (("s1", p1), ("s2", p2)):
            mm = M.volume_metrics(p, y, u)
            r[f"{nm}_psnr"], r[f"{nm}_ssim"], r[f"{nm}_mae"] = mm["psnr"], mm["ssim"], mm["mae"]
        rows.append(r)
    g = M.aggregate(rows)
    print(f"\n== refinement of the best model, both confounds removed ({contrast}) ==")
    print(f"  stage 1       PSNR {g['s1_psnr'][0]:.3f}  SSIM {g['s1_ssim'][0]:.4f}  MAE {g['s1_mae'][0]:.5f}")
    print(f"  stage 1 + 2   PSNR {g['s2_psnr'][0]:.3f}  SSIM {g['s2_ssim'][0]:.4f}  MAE {g['s2_mae'][0]:.5f}")
    print(f"  delta         PSNR {g['s2_psnr'][0]-g['s1_psnr'][0]:+.3f}  "
          f"SSIM {g['s2_ssim'][0]-g['s1_ssim'][0]:+.4f}")
    for k in ("psnr", "ssim"):
        print(f"  Wilcoxon {k}: p = {paired_test([r[f's2_{k}'] for r in rows], [r[f's1_{k}'] for r in rows]):.4f}")
    json.dump({"rows": rows, "agg": g}, open(f"{OUT}/d3cascade5_{contrast}.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "T2"
    rotate(c); fit_stage2(c); evaluate(c)
