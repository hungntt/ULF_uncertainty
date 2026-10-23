"""Is the cross-subject residual cosine a ceiling, or only a ceiling for unpaired methods?

Earlier work on this cohort measured cos(d_gt_i, d_gt_j) between *different*
subjects at 0.50-0.67 and treated it as the achievable bar for the ULF->3T
correction direction. That quantity is how much of the correction is shared
across subjects, i.e. the best a method that cannot see this subject's pair can
do. It is not an upper bound on paired supervision, which gets to use the
subject's own measurement. This script measures both numbers on the same data
with the same wavelet and the same metric.
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D

BANDS = ["LL", "LH", "HL", "HH"]


def bands_of(x, dwt):
    yl, yh = dwt(x)
    return [yl] + [yh[0][:, :, i] for i in range(3)]


def main(contrast="T2"):
    from pytorch_wavelets import DWTForward
    dwt = DWTForward(J=1, wave="sym4", mode="zero").cuda()
    test = D.PairedCohort("test", contrast)
    # per-slice residual, tagged with subject and normalised axial position
    dd, sid, zpos = {b: [] for b in BANDS}, [], []
    for k, s in enumerate(test.man):
        i0, n = s["slice_start"], s["slice_count"]
        for i in range(0, n, 32):
            j = np.arange(i0 + i, min(i0 + i + 32, i0 + n))
            u = torch.from_numpy(test.ulf[j][:, 1:2].astype(np.float32)).cuda()
            h = torch.from_numpy(test.hf[j].astype(np.float32)).cuda()
            bu, bh = bands_of(u, dwt), bands_of(h, dwt)
            for b, a1, a2 in zip(BANDS, bu, bh):
                dd[b].append((a2 - a1).flatten(1).cpu())
            sid += [k] * len(j)
            zpos += list((np.arange(i, i + len(j)) / n))
    sid = np.array(sid); zpos = np.array(zpos)
    dd = {b: torch.cat(v).cuda() for b, v in dd.items()}
    print(f"=== {contrast}: cross-subject structure of the ULF->3T residual "
          f"({len(sid)} slices, {len(test.man)} subjects) ===")

    res = {}
    rng = np.random.default_rng(0)
    for b in BANDS:
        X = F.normalize(dd[b], dim=1)
        # pair slices from different subjects at a matched axial position
        same_z, rand_pair = [], []
        for _ in range(20000):
            i = rng.integers(len(sid))
            cand = np.where((sid != sid[i]) & (np.abs(zpos - zpos[i]) < 0.03))[0]
            if len(cand) == 0: continue
            same_z.append((i, cand[rng.integers(len(cand))]))
            j = rng.integers(len(sid))
            if sid[j] != sid[i]: rand_pair.append((i, j))
        def cosof(pairs):
            a = torch.tensor([p[0] for p in pairs], device="cuda")
            c = torch.tensor([p[1] for p in pairs], device="cuda")
            return (X[a] * X[c]).sum(1)
        cz, cr = cosof(same_z), cosof(rand_pair)
        # within-subject, adjacent slices: an upper reference for "shared" structure
        adj = [(i, i + 1) for i in range(len(sid) - 1) if sid[i] == sid[i + 1]]
        ca = cosof(adj)
        res[b] = dict(cross_subject_matched_z=float(cz.mean()), cross_subject_random=float(cr.mean()),
                      within_subject_adjacent=float(ca.mean()))
        print(f"  {b}: cross-subject @matched z {cz.mean():.3f}±{cz.std():.3f} | "
              f"cross-subject random {cr.mean():.3f} | within-subject adjacent slice {ca.mean():.3f}")
    json.dump(res, open(f"/mnt/G/Projects/ULF-MICCAI/out/ceiling_{contrast}.json", "w"), indent=1)


def pooling_probe(contrast="T2"):
    """Earlier probes measured this cosine in an 8x8 / 16x16 pooled LL embedding.
    Pooling averages away the subject-specific detail and leaves a smooth brain
    silhouette that every subject shares, so the number it reports is mostly
    'brains are brain-shaped'. This quantifies that inflation."""
    from pytorch_wavelets import DWTForward
    import torch.nn.functional as F
    dwt = DWTForward(J=1, wave="sym4", mode="zero").cuda()
    test = D.PairedCohort("test", contrast)
    dl, sid, zpos = [], [], []
    for k, s in enumerate(test.man):
        i0, n = s["slice_start"], s["slice_count"]
        for i in range(0, n, 32):
            j = np.arange(i0 + i, min(i0 + i + 32, i0 + n))
            u = torch.from_numpy(test.ulf[j][:, 1:2].astype(np.float32)).cuda()
            h = torch.from_numpy(test.hf[j].astype(np.float32)).cuda()
            dl.append((dwt(h)[0] - dwt(u)[0]).cpu())
            sid += [k] * len(j); zpos += list(np.arange(i, i + len(j)) / n)
    dl = torch.cat(dl).cuda(); sid = np.array(sid); zpos = np.array(zpos)
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(20000):
        i = rng.integers(len(sid))
        c = np.where((sid != sid[i]) & (np.abs(zpos - zpos[i]) < 0.03))[0]
        if len(c): pairs.append((i, c[rng.integers(len(c))]))
    a = torch.tensor([p[0] for p in pairs], device="cuda")
    b = torch.tensor([p[1] for p in pairs], device="cuda")
    print(f"\n=== {contrast}: same cross-subject LL cosine, as a function of pooling ===")
    out = {}
    for p in [1, 2, 4, 8, 16]:
        X = dl if p == 1 else F.avg_pool2d(dl, p)
        res = X.shape[-1]
        Xn = F.normalize(X.flatten(1), dim=1)
        c = (Xn[a] * Xn[b]).sum(1)
        out[f"{res}x{res}"] = float(c.mean())
        print(f"  LL at {res:3d}x{res:<3d} (pool {p:2d}): cross-subject cosine {c.mean():.3f} ± {c.std():.3f}")
    json.dump(out, open(f"/mnt/G/Projects/ULF-MICCAI/out/ceiling_pool_{contrast}.json", "w"), indent=1)


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "T2"
    main(c); pooling_probe(c)
