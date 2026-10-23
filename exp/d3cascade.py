"""Is the flagged error reducible from the data already in hand?

The QA map predicts error *magnitude* at AUROC 0.82, yet the enhancer still makes
that error. If the error were predictable in sign and value the enhancer could just
subtract it, so the open question is whether the predictable part is only "where and
how much" (r) and not "which way" (cosine) - the Finding A split, one level up.

A stage-2 network is the direct test: give it the 64 mT slab and the stage-1 output
and ask it to regress the *signed* residual y - yhat. Trained on out-of-fold stage-1
predictions, so it never sees a prediction the stage-1 model made on its own training
subjects.

  error falls  -> refinement is real and an iterative loop has something to converge to
  error flat   -> the error is irreducible and no amount of re-synthesis will move it
"""
import argparse, json, sys
import numpy as np
import torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
CACHE = "/tmp/claude-17055/-mnt-G/5ec3b7fc-b916-4b70-887e-dbe4d6709b39/scratchpad/data"


def train(contrast, steps=9000, bs=16, base=32):
    T.seed_all(0)
    coh, tr, val = D.dev_split(contrast)
    oof = np.load(f"{CACHE}/qa_oof_{contrast}.npy")
    rows = coh.rows_for(sorted(tr))
    net = Mo.UNet(4, 1, base=base, residual_from=None).cuda()
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
    torch.save(net.state_dict(), f"{OUT}/d3cascade_{contrast}.pt")
    return net


@torch.no_grad()
def evaluate(contrast):
    test = D.PairedCohort("test", contrast)
    s1 = Mo.UNet(3, 1, base=32, residual_from=None).cuda()
    s1.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_{contrast}.pt")); s1.eval()
    s2 = Mo.UNet(4, 1, base=32, residual_from=None).cuda()
    s2.load_state_dict(torch.load(f"{OUT}/d3cascade_{contrast}.pt")); s2.eval()
    qa = Mo.UNet(4, 1, base=32, residual_from=None).cuda()
    qa.load_state_dict(torch.load(f"{OUT}/d3qa_{contrast}.pt")); qa.eval()

    rows = []
    for sid in test.subjects:
        idx = test.subject_rows(sid)
        P1, P2, Y, U, Q = [], [], [], [], []
        for i in range(0, len(idx), 24):
            j = idx[i:i + 24]
            x = torch.from_numpy(test.ulf[j].astype(np.float32)).cuda()
            y = torch.from_numpy(test.hf[j].astype(np.float32)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p1 = s1(x).float().clamp(0, 1)
                p2 = (p1 + s2(torch.cat([x, p1], 1)).float()).clamp(0, 1)
                q = qa(torch.cat([x, p1], 1)).float()
            P1.append(p1); P2.append(p2); Y.append(y); U.append(x[:, 1:2]); Q.append(q)
        p1, p2, y, u, q = (torch.cat(v) for v in (P1, P2, Y, U, Q))
        r = {"sid": sid}
        for nm, p in (("s1", p1), ("s2", p2)):
            mm = M.volume_metrics(p, y, u)
            r[f"{nm}_psnr"], r[f"{nm}_ssim"] = mm["psnr"], mm["ssim"]
            r[f"{nm}_mae"] = mm["mae"]
        # did it help specifically where the QA map said it would be bad?
        m = ((u > 0) | (y > 0))
        qv, e1, e2 = q[m], (p1 - y).abs()[m], (p2 - y).abs()[m]
        hi = qv > torch.quantile(qv.float(), 0.90)
        r["mae_flagged_s1"] = float(e1[hi].mean()); r["mae_flagged_s2"] = float(e2[hi].mean())
        r["mae_rest_s1"] = float(e1[~hi].mean()); r["mae_rest_s2"] = float(e2[~hi].mean())
        rows.append(r)
    g = M.aggregate(rows)
    print(f"\n== stage-2 cascade, {contrast} (locked 10-subject test) ==")
    print(f"  stage 1        PSNR {g['s1_psnr'][0]:.3f}  SSIM {g['s1_ssim'][0]:.4f}  MAE {g['s1_mae'][0]:.5f}")
    print(f"  stage 1 + 2    PSNR {g['s2_psnr'][0]:.3f}  SSIM {g['s2_ssim'][0]:.4f}  MAE {g['s2_mae'][0]:.5f}")
    print(f"  delta          PSNR {g['s2_psnr'][0]-g['s1_psnr'][0]:+.3f}  "
          f"SSIM {g['s2_ssim'][0]-g['s1_ssim'][0]:+.4f}  MAE {g['s2_mae'][0]-g['s1_mae'][0]:+.5f}")
    print(f"\n  in the worst-10% QA-flagged voxels:  {g['mae_flagged_s1'][0]:.5f} -> "
          f"{g['mae_flagged_s2'][0]:.5f}  ({100*(g['mae_flagged_s2'][0]/g['mae_flagged_s1'][0]-1):+.1f}%)")
    print(f"  everywhere else:                     {g['mae_rest_s1'][0]:.5f} -> "
          f"{g['mae_rest_s2'][0]:.5f}  ({100*(g['mae_rest_s2'][0]/g['mae_rest_s1'][0]-1):+.1f}%)")
    from ulfkit.metrics import paired_test
    for k in ("psnr", "ssim"):
        a = [r[f"s2_{k}"] for r in rows]; b = [r[f"s1_{k}"] for r in rows]
        print(f"  Wilcoxon {k}: p = {paired_test(a, b):.4f}")
    json.dump({"rows": rows, "agg": g}, open(f"{OUT}/d3cascade_{contrast}.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--steps", type=int, default=9000)
    a = ap.parse_args()
    train(a.contrast, a.steps); evaluate(a.contrast)
