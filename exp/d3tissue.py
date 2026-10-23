"""A downstream proxy: does the synthesis survive tissue segmentation?

Every result in this programme is measured on the image itself - SSIM, retention, a
detectability map. The standing objection is that none of it is a task. A full segmentation
model is out of scope here, but a reproducible proxy is not: fit a 3-component Gaussian
mixture to the within-brain intensities of the REAL 3T, then apply that same fitted model to
the synthesised 3T and measure how much the tissue labelling agrees.

This is deliberately the easy version of the question. The mixture is fitted per subject on
the real image, so it is a best case: the classifier is tuned to the target, and only the
synthesised intensities have to land in the right places. A pipeline that fails here fails
a task no harder than thresholding.

Reported per subject: Dice per tissue class, and the fraction of brain voxels that change
class. Identity (the 64 mT input itself) is the floor.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from sklearn.mixture import GaussianMixture
from ulfkit import data as D, train as T
from exp.d3qa import enhancer
from exp.d3arch import ARCHS, load
from exp.d3detect import operator

OUT = "/mnt/G/Projects/ULF-MICCAI/out"


def seg(vals, gm):
    """Label by the fitted mixture, with components ordered by intensity."""
    lab = gm.predict(vals.reshape(-1, 1))
    order = np.argsort(gm.means_.ravel())
    remap = np.zeros(len(order), int)
    for i, o in enumerate(order):
        remap[o] = i
    return remap[lab]


@torch.no_grad()
def main(contrast="T2"):
    T.seed_all(0)
    test = D.PairedCohort("test", contrast)
    fwd = operator(contrast, "learned")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    nets = []
    for nm, path, kind in ARCHS:
        try:
            nets.append((nm, load(kind, path)))
        except FileNotFoundError:
            pass
    names = ["identity (64 mT)"] + [nm for nm, _ in nets]
    per = {nm: {"dice": [], "chg": []} for nm in names}

    for sid in test.subjects:
        idx = test.subject_rows(sid)
        acc = {nm: [[], [], []] for nm in names}
        chg = {nm: [] for nm in names}
        for i0 in range(0, len(idx), 8):
            j = idx[i0:i0 + 8]
            y = torch.from_numpy(test.hf[j].astype(np.float32)).cuda()
            x = fwd(y, gen)
            preds = {"identity (64 mT)": x[:, 1:2]}
            for nm, net in nets:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    preds[nm] = net(x).float().clamp(0, 1)
            yc = y[:, 0].cpu().numpy()
            for k in range(len(j)):
                m = yc[k] > 0
                if m.sum() < 2000:
                    continue
                v = yc[k][m]
                gm = GaussianMixture(3, random_state=0, max_iter=60).fit(v.reshape(-1, 1))
                ref = seg(v, gm)
                for nm in names:
                    pv = preds[nm][k, 0].cpu().numpy()[m]
                    lab = seg(pv, gm)
                    chg[nm].append(float(np.mean(lab != ref)))
                    for c in range(3):
                        a, b = ref == c, lab == c
                        d = 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
                        acc[nm][c].append(d)
        for nm in names:
            if acc[nm][0]:
                per[nm]["dice"].append([float(np.mean(acc[nm][c])) for c in range(3)])
                per[nm]["chg"].append(float(np.mean(chg[nm])))
        print(f"  {sid} done", flush=True)

    print(f"\n== tissue-labelling agreement, mixture fitted on the REAL 3T ==")
    print(f"  {'pipeline':22s} {'CSF/dark':>9s} {'GM/mid':>8s} {'WM/bright':>10s} "
          f"{'mean Dice':>10s} {'% voxels reclassified':>22s}")
    summ = {}
    for nm in names:
        if not per[nm]["dice"]:
            continue
        Dd = np.array(per[nm]["dice"]); ch = np.array(per[nm]["chg"])
        summ[nm] = dict(dice=Dd.mean(0).tolist(), mean=float(Dd.mean()),
                        chg=float(ch.mean()), chg_sd=float(ch.std(ddof=1)))
        print(f"  {nm:22s} {Dd[:,0].mean():9.3f} {Dd[:,1].mean():8.3f} {Dd[:,2].mean():10.3f} "
              f"{Dd.mean():10.3f} {ch.mean()*100:20.1f}%")
    json.dump(summ, open(f"{OUT}/d3tissue_{contrast}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "T2")
