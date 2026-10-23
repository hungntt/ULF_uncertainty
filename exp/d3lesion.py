"""Does the enhancer silently erase pathology, and does the QA map warn about it?

Natural pathology cannot be studied on this cohort, but controlled lesions can.
A lesion of known size and contrast is inserted into a held-out 3T image, pushed
through the forward operator to produce a 64 mT scan that genuinely contains it, and
then synthesised. Two questions:

  conspicuity  how much of the lesion's contrast survives synthesis?
  warning      does the QA map flag the lesion, or does it pass unremarked?

Insertion happens in the *3T* and is degraded forward, never into the 64 mT directly:
the question is whether a lesion that was really there survives the pipeline.

Both forward operators are run. The learned F is trained only on healthy anatomy and
may not degrade an unseen structure faithfully; the hand-crafted physics chain has an
explicit k-space truncation that does not care what it is truncating. Agreement
between them is the evidence that the result is about synthesis, not about F.
"""
import argparse, json, sys
import numpy as np
import torch
import torch.nn.functional as F_
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T
from exp.d1 import load_forward, handcrafted_D, radial_bins
from exp.d3qa import enhancer

OUT = "/mnt/G/Projects/ULF-MICCAI/out"


def blob(h, w, cy, cx, r, device="cuda"):
    """Smooth spherical-ish lesion profile, peak 1 at the centre."""
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                            indexing="ij")
    d2 = ((yy - cy) ** 2 + (xx - cx) ** 2).float() / (r ** 2)
    return torch.exp(-d2 * 2.0).clamp(max=1.0)


def pick_sites(y, mask, n, r, rng):
    """Interior white-matter-ish sites: bright enough to be tissue, far from the edge,
    and locally homogeneous so the inserted lesion is the only new structure."""
    g = (y[:, :, 1:, :] - y[:, :, :-1, :]).abs().mean()
    sites = []
    H, W = y.shape[-2:]
    er = F_.max_pool2d(-mask, 2 * r + 1, 1, r)          # erode the brain mask by r
    inner = (-er)[0, 0]
    cand = torch.nonzero((inner > 0.5) & (y[0, 0] > 0.12) & (y[0, 0] < 0.45))
    if len(cand) == 0:
        return []
    for _ in range(n):
        sites.append(tuple(cand[rng.integers(len(cand))].tolist()))
    return sites


@torch.no_grad()
def run(contrast="T2", radii=(2, 3, 4, 6), contrasts=(0.10, 0.20, 0.35), n_sites=4,
        n_slices=12, seed=0):
    T.seed_all(seed)
    rng = np.random.default_rng(seed)
    test = D.PairedCohort("test", contrast)
    enh = enhancer(); enh.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_{contrast}.pt")); enh.eval()
    qa = Mo.UNet(4, 1, base=32, residual_from=None).cuda()
    qa.load_state_dict(torch.load(f"{OUT}/d3qa_{contrast}.pt")); qa.eval()
    fwd = load_forward(contrast)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)

    def degrade(y3t, op):
        """3T slice -> a 3-slice 64 mT slab containing whatever is in y3t."""
        if op == "learned":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d = fwd(y3t).float().clamp(0, 1)
        else:
            d = handcrafted_D(y3t, gen=gen)
        return d.repeat(1, 3, 1, 1) * (y3t > 0)

    rows = []
    for op in ("learned", "handcrafted"):
        for sid in test.subjects:
            idx = test.subject_rows(sid)
            sel = idx[np.linspace(len(idx) // 5, 4 * len(idx) // 5, n_slices).astype(int)]
            for k in sel:
                y0 = torch.from_numpy(test.hf[k:k + 1].astype(np.float32)).cuda()
                m = (y0 > 0).float()
                # the lesion-free control for this slice: everything is measured as a
                # paired difference against it, so the underlying anatomy cancels and
                # the denominator is the inserted contrast rather than a local gradient
                x0 = degrade(y0, op)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p0 = enh(x0).float().clamp(0, 1)
                    q0 = qa(torch.cat([x0, p0], 1)).float()
                for r in radii:
                    sites = pick_sites(y0, m, n_sites, r, rng)
                    for (cy, cx) in sites:
                        b = blob(256, 256, cy, cx, r)[None, None] * m
                        core = b > 0.6
                        ring = (b > 0.05) & (b < 0.25)
                        if core.sum() < 4 or ring.sum() < 8:
                            continue
                        for c in contrasts:
                            yl = (y0 + c * b).clamp(0, 1)
                            dy = yl - y0
                            true_c = (dy[core].mean() - dy[ring].mean()).item()
                            if true_c < 0.25 * c:        # clipped against the ceiling
                                continue
                            xl = degrade(yl, op)
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                p = enh(xl).float().clamp(0, 1)
                                q = qa(torch.cat([xl, p], 1)).float()
                            dp = p - p0
                            rec_c = (dp[core].mean() - dp[ring].mean()).item()
                            rows.append(dict(op=op, sid=sid, r=r, c=c,
                                             true_contrast=true_c, recovered=rec_c,
                                             retention=rec_c / true_c,
                                             qa_lesion=q[core].mean().item(),
                                             qa_control=q0[core].mean().item(),
                                             qa_ratio=q[core].mean().item() / max(q0[core].mean().item(), 1e-9),
                                             err_lesion=(p - yl)[core].abs().mean().item()))
            print(f"  {op} {sid}: {len(rows)} measurements", flush=True)
    json.dump(rows, open(f"{OUT}/d3lesion_{contrast}.json", "w"), indent=1, default=float)
    return rows


def report(rows, contrast):
    import numpy as np
    print(f"\n== lesion conspicuity, {contrast}: fraction of contrast surviving synthesis ==")
    for op in ("learned", "handcrafted"):
        R = [x for x in rows if x["op"] == op]
        if not R: continue
        radii = sorted({x["r"] for x in R}); cs = sorted({x["c"] for x in R})
        print(f"\n  forward operator: {op}")
        print("    radius |" + "".join(f"  c={c:.2f}" for c in cs))
        for r in radii:
            cells = []
            for c in cs:
                v = [x["retention"] for x in R if x["r"] == r and x["c"] == c]
                cells.append(f"   {np.mean(v):5.2f}" if v else "      -")
            print(f"    {r*2:3d} mm |" + "".join(cells))
    print(f"\n== does the QA map warn? (score inside the lesion vs same site without one) ==")
    for op in ("learned", "handcrafted"):
        R = [x for x in rows if x["op"] == op]
        if not R: continue
        radii = sorted({x["r"] for x in R})
        print(f"\n  forward operator: {op}")
        for r in radii:
            lv = np.array([x["qa_lesion"] for x in R if x["r"] == r])
            cv = np.array([x["qa_control"] for x in R if x["r"] == r])
            ret = np.mean([x["retention"] for x in R if x["r"] == r])
            print(f"    {r*2:3d} mm: QA {cv.mean():.4f} -> {lv.mean():.4f} "
                  f"(x{lv.mean()/max(cv.mean(),1e-9):.2f})   retention {ret:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--n_slices", type=int, default=12)
    a = ap.parse_args()
    rows = run(a.contrast, n_slices=a.n_slices)
    report(rows, a.contrast)
