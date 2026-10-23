"""Side-by-side panels for whichever checkpoints have finished."""
import json, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, models as Mo

OUT, FIG = "/mnt/G/Projects/ULF-MICCAI/out", "/mnt/G/Projects/ULF-MICCAI/fig"

CANDS = [("base_unet", "U-Net (L1)", "unet"), ("base_pix2pix", "cGAN", "unet"),
         ("base_lohiresgan", "LoHiResGAN", "lohires"),
         ("d1_none", "paired only", "unet"), ("d1_learned_cal", "D1: calibrated corpus", "unet")]


def build(kind, base=32):
    return Mo.LoHiResGen(3, 1).cuda() if kind == "lohires" else Mo.UNet(3, 1, base=base, residual_from=(1, 2)).cuda()


@torch.no_grad()
def predict(net, ulf, bs=16):
    net.eval(); o = []
    for i in range(0, len(ulf), bs):
        x = torch.from_numpy(ulf[i:i + bs].astype(np.float32)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o.append(net(x).float().clamp(0, 1).cpu().numpy())
    return np.concatenate(o)


def main(contrast="T2", sid=None, frac=0.5, zoom=(70, 190, 60, 200)):
    test = D.PairedCohort("test", contrast)
    sid = sid or test.subjects[1]
    idx = test.subject_rows(sid)
    k = idx[int(len(idx) * frac)]
    ulf = test.ulf[k:k + 1]; hf = test.hf[k:k + 1, 0]
    panels = [("64 mT input", ulf[0, 1])]
    for tag, lab, kind in CANDS:
        p = f"{OUT}/{tag}_{contrast}.pt"
        if not os.path.exists(p): continue
        net = build(kind); net.load_state_dict(torch.load(p))
        panels.append((lab, predict(net, ulf)[0, 0]))
        del net; torch.cuda.empty_cache()
    panels.append(("3T reference", hf[0]))

    y0, y1, x0, x1 = zoom
    n = len(panels)
    fig, ax = plt.subplots(3, n, figsize=(1.9 * n, 5.9))
    for j, (lab, im) in enumerate(panels):
        ax[0, j].imshow(im, cmap="gray", vmin=0, vmax=0.8)
        ax[0, j].set_title(lab, fontsize=8)
        ax[1, j].imshow(im[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=0.8)
        e = np.abs(im - hf[0])
        ax[2, j].imshow(e, cmap="magma", vmin=0, vmax=0.25)
        if j == 0: ax[2, j].set_ylabel("|error| vs 3T", fontsize=7)
        for r in range(3): ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
    ax[1, 0].set_ylabel("zoom", fontsize=7)
    fig.suptitle(f"{sid}  {contrast}  (slice {int(frac*100)}% of z-window)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_qualitative_{contrast}.png", dpi=190, bbox_inches="tight")
    print("saved", f"{FIG}/fig_qualitative_{contrast}.png", f"({len(panels)} panels)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "T2")
