"""Shared training / evaluation loop."""
import time
import numpy as np
import torch
import torch.nn.functional as F

from . import metrics as M


def seed_all(s=0):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class Sampler:
    """Random slice batches with LR-flip augmentation, applied identically to x and y."""

    def __init__(self, xs, ys, rows, bs=16, aug=True, seed=0):
        self.xs, self.ys, self.rows, self.bs, self.aug = xs, ys, rows, bs, aug
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        p = self.rng.choice(len(self.rows), self.bs, replace=False)
        r = np.sort(self.rows[p])
        x = torch.from_numpy(self.xs[r].astype(np.float32)).cuda(non_blocking=True)
        y = torch.from_numpy(self.ys[r].astype(np.float32)).cuda(non_blocking=True)
        if self.aug and self.rng.random() < 0.5:
            x, y = torch.flip(x, [-1]), torch.flip(y, [-1])
        return x, y


def ssim_loss(a, b, mask):
    s = M.ssim_map(a, b)
    return 1 - (s * mask).sum() / mask.sum().clamp(min=1)


def train_regression(model, sampler, steps, lr=2e-4, l_ssim=0.0, log=None,
                     val_fn=None, val_every=1000, disc=None, l_adv=0.0, l_l1=1.0,
                     amp=True, warm=200):
    """L1 (+ optional masked-SSIM, + optional PatchGAN) supervised translation."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps,
                                                pct_start=max(0.05, 3.0 / steps))
    optd = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999)) if disc else None
    dt = torch.bfloat16 if amp else torch.float32
    best, hist = None, []
    t0 = time.time()
    for it in range(steps):
        x, y = sampler()
        if it == 0:
            # broadcasting a 1-channel prediction against a 2-channel target is
            # silent in torch and regresses the head onto the mean of both.
            p0 = model(x)
            assert p0.shape == y.shape, f"output {tuple(p0.shape)} != target {tuple(y.shape)}"
            del p0
        mask = (y > 0).float()          # 3T support; same shape as the output
        with torch.autocast("cuda", dtype=dt, enabled=amp):
            p = model(x)
            # full-field L1 keeps the skull-stripped background at zero; the mask
            # term re-weights the ~25% of pixels that actually carry anatomy.
            l1 = (p - y).abs()
            loss = l_l1 * (l1.mean() + (l1 * mask).sum() / mask.sum().clamp(min=1))
            if l_ssim:
                loss = loss + l_ssim * sum(ssim_loss(p.float()[:, c:c+1], y.float()[:, c:c+1],
                                                     mask[:, c:c+1]) for c in range(y.shape[1]))
            if disc is not None and it > warm:
                loss = loss + l_adv * (-disc(torch.cat([x, p], 1)).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if disc is not None and it > warm:
            with torch.autocast("cuda", dtype=dt, enabled=amp):
                dr = disc(torch.cat([x, y], 1))
                df = disc(torch.cat([x, p.detach()], 1))
                ld = (F.relu(1 - dr).mean() + F.relu(1 + df).mean()) / 2
            optd.zero_grad(set_to_none=True); ld.backward(); optd.step()

        if val_fn is not None and (it + 1) % val_every == 0:
            v = val_fn(model)
            hist.append((it + 1, v))
            if best is None or v["score"] > best[1]["score"]:
                best = (it + 1, v, {k: p_.detach().clone() for k, p_ in model.state_dict().items()})
            if log:
                log(f"  step {it+1:6d}  loss {loss.item():.4f}  "
                    + "  ".join(f"{k} {vv:.4f}" for k, vv in v.items())
                    + f"  [{time.time()-t0:.0f}s]")
    if best is not None:
        model.load_state_dict(best[2])
    return dict(best_step=best[0] if best else steps, hist=hist,
                best=best[1] if best else None, secs=time.time() - t0)


@torch.no_grad()
def evaluate(model, cohort, sids, in_fn, out_ch=1, bs=32, bands=True):
    """Per-subject metrics. in_fn(rows) -> (x, ulf_centre, hf) numpy arrays."""
    model.eval()
    rows = []
    for sid in sids:
        xs, us, hs, ps = [], [], [], []
        idx = cohort.subject_rows(sid) if hasattr(cohort, "subject_rows") else cohort.rows_for([sid])
        for i in range(0, len(idx), bs):
            x, u, h = in_fn(idx[i:i + bs])
            x = torch.from_numpy(x).cuda().float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(x).float().clamp(0, 1)
            ps.append(p); us.append(torch.from_numpy(u).cuda().float()); hs.append(torch.from_numpy(h).cuda().float())
        p, u, h = torch.cat(ps), torch.cat(us), torch.cat(hs)
        r = {"sid": sid}
        for c in range(out_ch):
            pre = "" if out_ch == 1 else f"c{c}_"
            m = M.volume_metrics(p[:, c:c+1], h[:, c:c+1], u[:, c:c+1])
            if bands:
                m.update(M.band_metrics(p[:, c:c+1], h[:, c:c+1], u[:, c:c+1]))
            m["hfpow"] = M.hf_power_error(p[:, c:c+1], h[:, c:c+1])
            r.update({pre + k: v for k, v in m.items()})
        rows.append(r)
    model.train()
    return rows
