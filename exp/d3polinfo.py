"""Is the lesion's T1 polarity recoverable from the 64 mT T1 input at all?

The augmentation result has two possible readings: the model cannot learn the rule because
the evidence is absent, or it will not learn it because the lesion is a five-thousandth of
the loss. They imply completely different fixes - a protocol change versus a training change
- so the evidence question is worth answering directly rather than inferring.

Measured here: the core-minus-ring contrast of the lesion response actually added to the
64 mT T1 slab, as a function of the inserted 3T T1 contrast. If the sign of the input
response tracks the sign of c1 with clean separation, the information is present and the
augmentation failure is about loss weighting.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, train as T
from exp.d3detect import operator, site_pool, blobs

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
CS = (-0.25, -0.12, 0.12, 0.25)
RADII = (2, 3, 4, 6)


@torch.no_grad()
def main(n_sites=4, seed=3):
    T.seed_all(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    mc = D.MultiContrastCohort("test")
    f1 = operator("T1", "learned")
    recs = []
    for sid in sorted(set(mc.sid)):
        pos = np.where(mc.sid == sid)[0]
        sel = pos[np.linspace(len(pos)//5, 4*len(pos)//5, 6).astype(int)]
        for i0 in range(0, len(sel), 3):
            p = sel[i0:i0+3]
            _, hf = mc.batch(p)
            y1 = torch.from_numpy(hf[:, 0:1].astype(np.float32)).cuda()
            xr = torch.from_numpy(mc.slab(p)[:, 0:3].astype(np.float32)).cuda()
            ok = site_pool(y1)
            jobs = []
            for s in range(len(p)):
                cand = torch.nonzero(ok[s])
                if len(cand) < 8: continue
                for _ in range(n_sites):
                    cy, cx = cand[rng.integers(len(cand))].tolist()
                    jobs.append((s, cy, cx, int(rng.choice(RADII)), float(rng.choice(CS))))
            if not jobs: continue
            si = torch.tensor([t[0] for t in jobs], device="cuda")
            cy = torch.tensor([t[1] for t in jobs], device="cuda")
            cx = torch.tensor([t[2] for t in jobs], device="cuda")
            rr = torch.tensor([t[3] for t in jobs], device="cuda")
            c1 = torch.tensor([t[4] for t in jobs], device="cuda")[:, None, None, None]
            b = blobs(cy, cx, rr) * (y1[si] > 0).float()
            y1l = (y1[si] + c1 * b).clamp(0, 1)
            near = (blobs(cy, cx, torch.clamp(rr.float() * 4, min=24)) > 0.05).float()
            resp = (f1(y1l, gen) - f1(y1[si], gen)) * near      # what training actually adds
            core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
            for k in range(len(jobs)):
                if core[k].sum() < 4 or ring[k].sum() < 8:
                    continue
                # channel 1 is the centre slice of the 3-slice slab
                d = resp[k, 1:2]
                recs.append(dict(c1=jobs[k][4], r=jobs[k][3],
                                 inp=float(d[core[k]].mean() - d[ring[k]].mean())))
        print(f"  {sid}: {len(recs)}", flush=True)

    C = np.array([r["c1"] for r in recs]); I = np.array([r["inp"] for r in recs])
    print(f"\n  lesion response in the 64 mT T1 input, by inserted 3T T1 contrast (n={len(C)})\n")
    print(f"  {'c1 (3T)':>9s} {'n':>5s} {'input core-ring':>17s} {'sd':>8s} {'% correct sign':>15s}")
    for c in sorted(set(C)):
        m = C == c
        print(f"  {c:+9.2f} {m.sum():5d} {I[m].mean():17.4f} {I[m].std(ddof=1):8.4f} "
              f"{100*np.mean(np.sign(I[m]) == np.sign(c)):14.0f}%")
    from sklearn.metrics import roc_auc_score
    au = roc_auc_score((C > 0).astype(int), I)
    print(f"\n  AUROC, telling T1-hyper from T1-hypo using the INPUT response alone: {au:.3f}")
    print(f"  overall correct-sign rate: {100*np.mean(np.sign(I) == np.sign(C)):.1f}%")
    json.dump(recs, open(f"{OUT}/d3polinfo.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
