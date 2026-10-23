"""Real focal structure against a matched Gaussian, through the same pipeline."""
import json, sys
import numpy as np, torch
import scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, train as T
from exp.d3qa import enhancer
from exp.d3detect import operator, blobs
from exp.d3realret import feature_mask, inpaint, fold_for

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
FIG = "/tmp/claude-17055/-mnt-G/5ec3b7fc-b916-4b70-887e-dbe4d6709b39/scratchpad"
C = {"ink": "#0e1419", "ink2": "#4a5560"}
plt.rcParams.update({"font.size": 8, "text.color": C["ink"], "figure.facecolor": "white"})
H = 30


@torch.no_grad()
def main(tag="_loose", n=4):
    T.seed_all(0)
    rows = json.load(open(f"{OUT}/d3realret{tag}.json"))
    # representative, not best-case: one site per subject, the one whose real-minus-
    # phantom gap is that subject's MEDIAN. Sorting by largest gap would show the
    # effect at its most flattering and misstate the typical size of it.
    for r in rows:
        r["gap"] = r["retention"] - r["phantom"]
    by = {}
    for r in rows:
        by.setdefault(r["sid"], []).append(r)
    reps = []
    for sid, rs in by.items():
        rs.sort(key=lambda r: r["gap"])
        reps.append(rs[len(rs) // 2])
    # keep locked-test subjects first, then fill, so held-out cases are visible
    reps.sort(key=lambda r: (r["split"] != "test", abs(r["gap"] - np.median([q["gap"] for q in rows]))))
    pick = reps[:n]
    nets = {}
    for f in range(4):
        m = enhancer(); m.load_state_dict(torch.load(f"{OUT}/d3qa_fold{f}_T2.pt")); m.eval()
        nets[f] = m
    fwd = operator("T2", "learned")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)

    fig, axes = plt.subplots(len(pick), 5, figsize=(11.0, 2.35 * len(pick)))
    for i, r in enumerate(pick):
        coh = D.PairedCohort(r["split"], "T2")
        y = coh.hf[r["row"], 0].astype(np.float32)
        fm, bg = feature_mask(y, r["cy"], r["cx"])
        y0 = inpaint(y, fm, bg)
        yt = torch.from_numpy(y).cuda()[None, None]
        y0t = torch.from_numpy(y0).cuda()[None, None]
        b = blobs(torch.tensor([r["cy"]], device="cuda"), torch.tensor([r["cx"]], device="cuda"),
                  torch.tensor([r["r_eq"]], device="cuda")) * (y0t > 0).float()
        yph = (y0t + r["contrast"] * b).clamp(0, 1)
        f = fold_for(r["sid"]); enh = nets[f if f is not None else 0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pw = enh(fwd(yt, gen)).float().clamp(0, 1)
            pp = enh(fwd(yph, gen)).float().clamp(0, 1)
            po = enh(fwd(y0t, gen)).float().clamp(0, 1)
        cy, cx = r["cy"], r["cx"]
        sl = (slice(max(0, cy-H), cy+H), slice(max(0, cx-H), cx+H))
        cut = lambda a: a[0, 0].cpu().numpy()[sl]
        panels = [("3T, real feature", y[sl], None),
                  ("inpainted baseline", y0[sl], None),
                  ("matched Gaussian", cut(yph), None),
                  (f"synthesised from real\nretention {r['retention']:.2f}", cut(pw), "win"),
                  (f"synthesised from phantom\nretention {r['phantom']:.2f}", cut(pp), "loss")]
        for j, (ttl, img, kind) in enumerate(panels):
            ax = axes[i][j] if len(pick) > 1 else axes[j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=0.8)
            ax.add_patch(plt.Circle((cx - sl[1].start, cy - sl[0].start),
                                    max(4, r["r_eq"] * 1.8), fill=False,
                                    color="#c4652f", lw=1.1))
            if i == 0:
                ax.set_title(ttl, fontsize=8, color=C["ink"], pad=6)
            elif j >= 3:
                ax.set_title(ttl.split("\n")[1], fontsize=8,
                             color="#1b7f55" if kind == "win" else "#b2403f", pad=4)
            ax.axis("off")
        (axes[i][0] if len(pick) > 1 else axes[0]).text(
            -0.06, 0.5, f"{r['sid']}\n{r['split']}", transform=(axes[i][0] if len(pick) > 1 else axes[0]).transAxes,
            rotation=90, va="center", ha="center", fontsize=7, color=C["ink2"])
    plt.tight_layout()
    plt.savefig(f"{FIG}/fig8_real.png", dpi=150, bbox_inches="tight")
    print("wrote", f"{FIG}/fig8_real.png")


if __name__ == "__main__":
    main()
