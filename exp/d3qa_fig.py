"""What the reader would actually see: the synthesised image with its QA overlay."""
import json, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, models as Mo, train as T
from exp.d3qa import enhancer

OUT, FIG = "/mnt/G/Projects/ULF-MICCAI/out", "/mnt/G/Projects/ULF-MICCAI/fig"
C = {"ink": "#0e1419", "ink2": "#4a5560", "grid": "#e4e3df"}
plt.rcParams.update({"font.size": 8, "text.color": C["ink"], "figure.facecolor": "white"})


@torch.no_grad()
def main(contrast="T2", n_sub=3, alpha=0.15):
    T.seed_all(0)
    test = D.PairedCohort("test", contrast)
    enh = enhancer(); enh.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_{contrast}.pt")); enh.eval()
    qa = Mo.UNet(4, 1, base=32, residual_from=None).cuda()
    qa.load_state_dict(torch.load(f"{OUT}/d3qa_{contrast}.pt")); qa.eval()
    lam = json.load(open(f"{OUT}/d3qa_deploy_{contrast}.json"))["lambda"]

    fig, axes = plt.subplots(n_sub, 5, figsize=(13.2, 2.75 * n_sub))
    for r, sid in enumerate(test.subjects[:n_sub]):
        idx = test.subject_rows(sid); k = idx[len(idx) // 2]
        x = torch.from_numpy(test.ulf[k:k+1].astype(np.float32)).cuda()
        y = torch.from_numpy(test.hf[k:k+1].astype(np.float32)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = enh(x).float().clamp(0, 1)
            q = qa(torch.cat([x, p], 1)).float()
        m = ((x[:, 1:2] > 0) | (y > 0)).float()
        p, q, e = p * m, q * m, (p - y).abs() * m
        ulf, hf = x[0, 1].cpu(), y[0, 0].cpu()
        pr, qq, er, mm = p[0, 0].cpu(), q[0, 0].cpu(), e[0, 0].cpu(), m[0, 0].cpu()

        panels = [("64 mT input", ulf, "gray", dict(vmin=0, vmax=.8)),
                  ("synthesised 3T", pr, "gray", dict(vmin=0, vmax=.8)),
                  ("3T reference\n(absent at deployment)", hf, "gray", dict(vmin=0, vmax=.8)),
                  ("QA map — predicted error", qq, "magma", dict(vmin=0, vmax=.35)),
                  ("true error (for validation only)", er, "magma", dict(vmin=0, vmax=.35))]
        for c, (t, im, cm, kw) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im, cmap=cm, **kw)
            if c == 1:          # outline what the calibrated threshold flags
                ax.contour(np.asarray(qq > lam, float), levels=[.5], colors=["#36d399"], linewidths=.8)
            if r == 0: ax.set_title(t, fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
        flagged = float(((qq > lam) & (mm > 0)).sum() / mm.sum())
        axes[r, 0].set_ylabel(f"{sid}\n{flagged*100:.0f}% flagged", fontsize=7.5)
    fig.suptitle(f"{contrast}: QA overlay at the conformally calibrated threshold "
                 f"(guaranteed miss rate {alpha}); green outline = flagged", fontsize=9.5, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig7_qa_{contrast}.png", dpi=180, bbox_inches="tight")
    print("saved", f"{FIG}/fig7_qa_{contrast}.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "T2")
