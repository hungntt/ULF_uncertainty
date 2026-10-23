"""A quality-assurance map: which regions of a synthesised 3T image should a reader
not trust?

This is a prediction task, not an interpretability argument. Posterior dispersion,
MC-dropout and input sensitivity all *guess* at error from model-internal signals.
A paired cohort lets the error be learned directly, which is the one thing a
synthesis-only or domain-randomised pipeline cannot do.

Leakage: the enhancer's residuals on its own training subjects are far smaller than
on unseen ones, so a QA map fitted to them under-predicts error at test time. The
enhancer is therefore cross-fitted over subject-disjoint folds and the QA model only
ever sees out-of-fold predictions.
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
CACHE = "/tmp/claude-17055/-mnt-G/5ec3b7fc-b916-4b70-887e-dbe4d6709b39/scratchpad/data"


def enhancer(base=32):
    return Mo.UNet(3, 1, base=base, residual_from=None).cuda()   # absolute output wins


def crossfit(contrast, folds=4, steps=9000, bs=16):
    """Out-of-fold enhancer predictions for every training subject."""
    coh, tr, val = D.dev_split(contrast)
    tr = sorted(tr)
    pred = np.zeros_like(coh.hf)
    for f in range(folds):
        hold = tr[f::folds]
        fit = [s for s in tr if s not in hold]
        print(f"\n[fold {f}] fit {len(fit)} subj, predict {len(hold)}", flush=True)
        T.seed_all(100 + f)
        net = enhancer()
        smp = T.Sampler(coh.ulf, coh.hf, coh.rows_for(fit), bs=bs, seed=f)
        T.train_regression(net, smp, steps, lr=2e-4, l_ssim=0.0, log=None)
        net.eval()
        with torch.no_grad():
            for sid in hold:
                idx = coh.subject_rows(sid)
                for i in range(0, len(idx), 32):
                    j = idx[i:i + 32]
                    x = torch.from_numpy(coh.ulf[j].astype(np.float32)).cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p = net(x).float().clamp(0, 1)
                    pred[j] = p.cpu().numpy().astype(pred.dtype)
        # keep the fold model: on the locked test subjects none of the four has seen
        # the data, so their disagreement is a free deep-ensemble uncertainty baseline
        torch.save(net.state_dict(), f"{OUT}/d3qa_fold{f}_{contrast}.pt")
        del net; torch.cuda.empty_cache()
    np.save(f"{CACHE}/qa_oof_{contrast}.npy", pred)
    print(f"\nout-of-fold predictions written for {len(tr)} subjects")


def pinball(p, t, tau=0.9):
    d = t - p
    return torch.maximum(tau * d, (tau - 1) * d).mean()


def train_qa(contrast, steps=9000, bs=16, base=32, tau=0.9):
    """Predict the tau-quantile of |error| from (64 mT slab, synthesised 3T)."""
    T.seed_all(0)
    coh, tr, val = D.dev_split(contrast)
    oof = np.load(f"{CACHE}/qa_oof_{contrast}.npy")
    rows = coh.rows_for(sorted(tr))
    qa = Mo.UNet(4, 1, base=base, residual_from=None).cuda()
    opt = torch.optim.AdamW(qa.parameters(), 2e-4, weight_decay=1e-4)
    lrs = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    for it in range(steps):
        j = rows[np.sort(rng.choice(len(rows), bs, replace=False))]
        x = torch.from_numpy(coh.ulf[j].astype(np.float32)).cuda()
        p = torch.from_numpy(oof[j].astype(np.float32)).cuda()
        y = torch.from_numpy(coh.hf[j].astype(np.float32)).cuda()
        if rng.random() < 0.5:
            x, p, y = (torch.flip(v, [-1]) for v in (x, p, y))
        err = (p - y).abs()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = qa(torch.cat([x, p], 1)).float()
        loss = pinball(q, err, tau)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(qa.parameters(), 1.0); opt.step(); lrs.step()
        if (it + 1) % 2000 == 0:
            print(f"  step {it+1:6d}  pinball {loss.item():.6f}", flush=True)
    torch.save(qa.state_dict(), f"{OUT}/d3qa_{contrast}.pt")
    return qa


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["crossfit", "qa"])
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--steps", type=int, default=9000)
    a = ap.parse_args()
    if a.stage == "crossfit": crossfit(a.contrast, steps=a.steps)
    else: train_qa(a.contrast, steps=a.steps)
