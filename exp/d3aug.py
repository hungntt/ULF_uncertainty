"""Can polarity-violating augmentation stop the T2-driven route inverting lesions?

The polarity sweep showed route B (64 mT T2 -> 3T T1) is not transferring a lesion but
applying the population-average T2->T1 coupling: it inverts the lesion's sign at 95% of
sites when the lesion violates that coupling. Route C sees the 64 mT T1 channel as well,
and inverts at 94% anyway - the disambiguating evidence is in its input and it ignores it.

It ignores it for a reason: every training pair is normal anatomy, where the coupling
always holds, so reading the T1 channel for polarity never pays. The remedy is to make
it pay - insert lesions during training whose T1 contrast is sometimes the "wrong" sign.

The lesion is added to the REAL 64 mT input as the forward operator's response to it,
  x_aug = x_real + [ f(y + c*b) - f(y) ],
so the measured noise and texture of the real acquisition are preserved and only the
lesion's physical response is synthetic.

Control and treatment differ in one flag, so the comparison is clean.
"""
import argparse, json, sys
import numpy as np, torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T
from exp.d2 import mc_inputs, _SubjView, CENTRES
from exp.d3detect import operator, site_pool, blobs

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
RADII = (2, 3, 4, 6)
# magnitudes stay coupled; only the SIGN of the T1 lesion is free, so the model is not
# being taught that the two contrasts are unrelated - just that the sign is not a given.
FRACS = (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)


class LesionSampler:
    """Co-registered T1/T2 slabs with physically-inserted lesions of both T1 polarities.

    It draws its own batch rather than wrapping MCSampler: the lesion needs BOTH 3T
    contrasts to build its two arms, and MCSampler returns only the target channels.
    """

    def __init__(self, mc, rows, bs=16, seed=0, out_sel=(0,), aug=0.5):
        self.mc, self.rows, self.bs, self.aug = mc, rows, bs, aug
        self.out_sel = list(out_sel)
        # two streams: batch choice and flip must not depend on whether a lesion was
        # drawn, or control and treatment would see different slices
        self.brng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed + 9973)
        self.f1, self.f2 = operator("T1", "learned"), operator("T2", "learned")
        self.gen = torch.Generator(device="cuda"); self.gen.manual_seed(seed)
        self.want_mask = False
        self.last_lesion = None

    @torch.no_grad()
    def __call__(self):
        p = np.sort(self.brng.choice(len(self.rows), self.bs, replace=False))
        pos = self.rows[p]
        x = torch.from_numpy(self.mc.slab(pos).astype(np.float32)).cuda()
        _, hf = self.mc.batch(pos)
        h = torch.from_numpy(hf.astype(np.float32)).cuda()      # (B,2,H,W) 3T T1,T2
        a = torch.ones(len(pos), 2, 256, 256, device="cuda")
        B = x.shape[0]

        if self.aug > 0:
            y1, y2 = h[:, 0:1], h[:, 1:2]
            ok = site_pool(y1)
            cy, cx, rr, cc1, cc2, use = [], [], [], [], [], []
            for s in range(B):
                cand = torch.nonzero(ok[s])
                if self.rng.random() >= self.aug or len(cand) < 8:
                    use.append(0.0); cy.append(0); cx.append(0); rr.append(2)
                    cc1.append(0.0); cc2.append(0.0); continue
                r0, c0 = cand[self.rng.integers(len(cand))].tolist()
                use.append(1.0); cy.append(r0); cx.append(c0)
                rr.append(int(self.rng.choice(RADII)))
                v2 = float(self.rng.choice((0.12, 0.25)))
                cc2.append(v2); cc1.append(v2 * float(self.rng.choice(FRACS)))
            if any(use):
                t = lambda v: torch.tensor(v, device="cuda")
                b = blobs(t(cy), t(cx), t(rr)) * (y1 > 0).float() * t(use)[:, None, None, None]
                c1 = t(cc1)[:, None, None, None]; c2 = t(cc2)[:, None, None, None]
                y1l = (y1 + c1 * b).clamp(0, 1)
                y2l = (y2 + c2 * b).clamp(0, 1)
                # add only the operator's RESPONSE to the lesion, so the real 64 mT
                # noise and texture survive and only the lesion is synthetic.
                # The learned operator is NOT local - about 70% of its response to a
                # local patch lands elsewhere - and an unmasked delta would leave a
                # global shift correlated with the lesion's sign for the model to read
                # as a shortcut. The response is therefore confined to the lesion.
                near = (blobs(t(cy), t(cx), t([max(4 * v, 24) for v in rr])) > 0.05).float()
                x[:, 0:3] += (self.f1(y1l, self.gen) - self.f1(y1, self.gen)) * near
                x[:, 3:6] += (self.f2(y2l, self.gen) - self.f2(y2, self.gen)) * near
                h = torch.cat([y1l, y2l], 1)
                self.last_lesion = (b > 0.25).float()

        y = h[:, self.out_sel]
        lm = self.last_lesion if self.last_lesion is not None else torch.zeros_like(y)
        if self.brng.random() < 0.5:
            x, y, a = torch.flip(x, [-1]), torch.flip(y, [-1]), torch.flip(a, [-1])
            lm = torch.flip(lm, [-1])
        self.last_lesion = None
        if self.want_mask:
            return torch.cat([x, a], 1), y, lm
        return torch.cat([x, a], 1), y


def train_weighted(net, smp, steps, lw, lr=2e-4):
    """L1 as usual, plus an explicit term over the lesion footprint only.

    Without it a radius-2-6 lesion is well under 1% of the L1 gradient and the model has
    no reason to learn a rule that pays only there. lw=1 gives the lesion region a term
    equal in weight to the whole-brain one despite covering a hundredth of the pixels.
    """
    opt = torch.optim.AdamW(net.parameters(), lr, betas=(0.9, 0.999), weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps,
                                              pct_start=max(0.05, 3.0 / steps))
    smp.want_mask = True
    for it in range(steps):
        x, y, lm = smp()
        mask = (y > 0).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = net(x)
            l1 = (p.float() - y).abs()
            loss = (l1.mean() + (l1 * mask).sum() / mask.sum().clamp(min=1)
                    + lw * (l1 * lm).sum() / lm.sum().clamp(min=1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
        if (it + 1) % 2000 == 0:
            print(f"  step {it+1:6d}  loss {loss.item():.5f}", flush=True)
    return net


def main(a):
    T.seed_all(a.seed)
    mc_tr, mc_te = D.MultiContrastCohort("train"), D.MultiContrastCohort("test")
    tr_ids = [s for s in sorted(set(mc_tr.sid)) if s not in D.DEV_VAL_IDS]
    out_sel = [0]                                   # 3T T1 output, the route-C head
    net = Mo.UNet(8, 1, base=32, residual_from=[CENTRES[0]]).cuda()
    smp = LesionSampler(mc_tr, mc_tr.rows_for(tr_ids), bs=a.bs, seed=a.seed,
                        out_sel=out_sel, aug=a.aug)
    print(f"[d3aug] aug={a.aug}  lw={a.lw}  steps={a.steps}  -> d2_bothtoT1{a.tag}.pt", flush=True)
    if a.lw > 0:
        train_weighted(net, smp, a.steps, a.lw)
    else:
        T.train_regression(net, smp, a.steps, lr=2e-4, log=print)

    def sub(fn):
        def g(pos):
            x, u, h = fn(pos)
            return x, u[:, out_sel], h[:, out_sel]
        return g
    rows = T.evaluate(net, _SubjView(mc_te), sorted(set(mc_te.sid)),
                      sub(mc_inputs(mc_te, (1, 1))), out_ch=1)
    g = M.aggregate(rows)
    print(f"\n== normal-anatomy test [aug={a.aug}] ==")
    print(f"  3T T1: PSNR {g['psnr'][0]:.3f}  SSIM {g['ssim'][0]:.4f}  r_HH {g['r_HH'][0]:.3f}")
    json.dump({"cfg": vars(a), "agg": {k: list(v) for k, v in g.items()}},
              open(f"{OUT}/d3aug{a.tag}.json", "w"), indent=1, default=float)
    torch.save(net.state_dict(), f"{OUT}/d2_bothtoT1{a.tag}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="_aug")
    ap.add_argument("--lw", type=float, default=0.0,
                    help="weight on an explicit lesion-footprint L1 term; 0 = plain loss")
    main(ap.parse_args())
