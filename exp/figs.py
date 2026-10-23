"""Paper figures. Direct-labelled lines (two series sit below 3:1 on white)."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT, FIG = "/mnt/G/Projects/ULF-MICCAI/out", "/mnt/G/Projects/ULF-MICCAI/fig"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": C["ink2"], "axes.linewidth": 0.6,
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"],
                     "text.color": C["ink"], "axes.labelcolor": C["ink2"],
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def label_right(ax, items, log=True):
    """Direct labels at the right edge, pushed apart so none overlap."""
    items = sorted(items, key=lambda t: t[1])
    ys = [np.log10(y) if log else y for _, y, _ in items]
    a0, a1 = ax.get_ylim()                  # space the labels by axis height, not data spread
    gap = (np.log10(a1 / a0) if log else (a1 - a0)) * 0.062
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap: ys[i] = ys[i - 1] + gap
    for (lab, _, col), y in zip(items, ys):
        ax.annotate(lab, (1.02, 10 ** y if log else y), xycoords=("axes fraction", "data"),
                    color=col, fontsize=6.5, va="center")


def style(ax):
    ax.grid(True, color=C["grid"], lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def fig_spectra():
    d = json.load(open(f"{OUT}/d1_spectra.json"))
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.2), gridspec_kw={"wspace": 0.75})
    series = [("3T target (POCEMR)", "pocemr_3t", C["blue"]),
              ("64 mT input (POCEMR)", "pocemr_ulf", C["orange"]),
              ("3T corpus (IXI/FOMO-50k)", "ixi_3t", C["aqua"]),
              ("64 mT, 48 unpaired subj", "ulfonly", C["yellow"])]
    for ax, c in zip(axes[:2], ["T2", "T1"]):
        ends = []
        for lab, k, col in series:
            y = np.array(d[c][k]); y = y / y.sum()
            ax.semilogy(np.arange(2, 125), y[2:125], color=col, lw=2)
            ends.append((lab, y[124], col))
        label_right(ax, ends)
        ax.set_title(f"{c}: radial power spectrum", color=C["ink"], fontsize=9)
        ax.set_xlabel("radial frequency (cycles / FOV)"); ax.set_xlim(0, 126)
        style(ax)
    axes[0].set_ylabel("normalised power")
    ax = axes[2]
    for c, col in [("T2", C["blue"]), ("T1", C["orange"])]:
        t = np.array(d[c]["pocemr_3t"]); i = np.array(d[c]["ixi_3t"])
        r = (t / t.sum()) / (i / i.sum())
        ax.plot(np.arange(2, 125), r[2:125], color=col, lw=2)
        ax.annotate(c, (1.02, r[124]), xycoords=("axes fraction", "data"),
                    color=col, fontsize=7.5, va="center")
    ax.axhline(1, color=C["ink2"], lw=0.8, ls="--")
    ax.set_title("target / corpus power ratio", color=C["ink"], fontsize=9)
    ax.set_xlabel("radial frequency"); ax.set_ylabel("ratio"); ax.set_xlim(0, 126)
    style(ax)
    fig.savefig(f"{FIG}/fig1_reference_gap.png", dpi=200, bbox_inches="tight")
    print("fig1 saved")


def fig_ceiling():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    ax = axes[0]
    res = [int(k.split("x")[0]) for k in json.load(open(f"{OUT}/ceiling_pool_T2.json"))]
    for c, col in [("T2", C["blue"]), ("T1", C["orange"])]:
        p = json.load(open(f"{OUT}/ceiling_pool_{c}.json"))
        y = list(p.values())
        ax.plot(range(len(y)), y, "o-", color=col, lw=2, ms=6)
        ax.annotate(c, (len(y) - 1, y[-1]), color=col, fontsize=7.5,
                    xytext=(5, 0), textcoords="offset points", va="center")
    ax.axhspan(0.50, 0.67, color=C["ink2"], alpha=0.12, lw=0)
    ax.annotate("range earlier work reported\nas the achievable ceiling", (0.15, 0.585),
                xycoords=("axes fraction", "data"), fontsize=6.5, color=C["ink2"])
    ax.set_xticks(range(len(res))); ax.set_xticklabels([f"{r}²" for r in res])
    ax.set_xlabel("LL sub-band resolution (pooled)")
    ax.set_ylabel("cross-subject residual cosine")
    ax.set_title("A pooled embedding inflates 'shared direction'", color=C["ink"], fontsize=9)
    style(ax)

    ax = axes[1]
    bands = ["LL", "LH", "HL", "HH"]
    w = 0.38
    for i, (c, col) in enumerate([("T2", C["blue"]), ("T1", C["orange"])]):
        d = json.load(open(f"{OUT}/ceiling_{c}.json"))
        x = np.arange(4) + (i - 0.5) * w
        ax.bar(x, [d[b]["within_subject_adjacent"] for b in bands], w * 0.92,
               color=col, label=f"{c} within-subject (adjacent slice)")
        ax.bar(x, [d[b]["cross_subject_matched_z"] for b in bands], w * 0.92,
               color=col, alpha=0.35, hatch="///", edgecolor="white", lw=0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(bands)
    ax.set_ylabel("residual cosine"); ax.set_xlabel("sym4 sub-band")
    ax.set_title("Solid: within subject.  Hatched: across subjects.", color=C["ink"], fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    style(ax)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_ceiling.png", dpi=200, bbox_inches="tight")
    print("fig2 saved")




def _load(p):
    try: return json.load(open(p))
    except Exception: return None


def fig_d1():
    arms = [("none", "real pairs only", C["ink2"]),
            ("handcrafted", "+ hand-crafted physics D", C["yellow"]),
            ("learned", "+ learned F", C["orange"]),
            ("learned_cal", "+ learned F, calibrated corpus", C["blue"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.1), gridspec_kw={"wspace": .62,
                                                                    "width_ratios": [1.25, 1]})
    # (a) finetuned quality, against the strongest model trained without any corpus
    ax = axes[0]
    ref = _load(f"{OUT}/base_unet_T2.json")
    if ref:
        rv = np.mean([r["ssim"] for r in ref["test_rows"]])
        ax.axvline(rv, color=C["ink2"], lw=1, ls=(0, (4, 3)))
        ax.annotate("best model with\nno corpus", (rv, -0.78), fontsize=6.5, color=C["ink2"],
                    ha="center", va="top")
    ys = []
    for i, (m, lab, col) in enumerate(arms):
        d = _load(f"{OUT}/d1_{m}_T2.json")
        if not d: continue
        v = np.array([r["ssim"] for r in d["test_rows"]])
        y = len(arms) - 1 - i
        ax.plot([v.min(), v.max()], [y, y], color=col, lw=1.4, alpha=.35, solid_capstyle="round")
        ax.plot(v.mean(), y, "o", color=col, ms=8, markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.annotate(f"{v.mean():.3f}", (v.mean(), y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=6.5, color=C["ink"])
        ys.append((y, lab))
    ax.set_yticks([y for y, _ in ys]); ax.set_yticklabels([l for _, l in ys], fontsize=7.5)
    ax.set_ylim(-0.6, len(arms) + 0.05); ax.set_xlabel("masked SSIM after finetuning")
    ax.grid(True, axis="x", color=C["grid"], lw=.5); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # (b) what transfers with no real pair ever seen
    ax = axes[1]
    labs, vals, cols = [], [], []
    for m, lab, col in arms[1:]:
        d = _load(f"{OUT}/d1_{m}_T2.json")
        if d and d.get("pre_dev"):
            labs.append(lab.replace("+ ", "")); vals.append(d["pre_dev"]["cosLL"]); cols.append(col)
    if vals:
        y = np.arange(len(vals))[::-1]
        ax.barh(y, vals, 0.55, color=cols)
        for yy, v in zip(y, vals):
            ax.annotate(f"{v:+.3f}", (v, yy), xytext=(6 if v > 0 else -6, 0),
                        textcoords="offset points", va="center",
                        ha="left" if v > 0 else "right", fontsize=7, color=C["ink"])
        ax.axvline(0, color=C["ink2"], lw=.9)
        ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7.5)
        ax.set_xlim(-0.22, 0.88)
        ax.set_xlabel("LL directional cosine, zero-shot on real 64 mT")
        ax.set_title("pretrained only \u2014 no real pair seen", fontsize=8.5, color=C["ink"])
        ax.grid(True, axis="x", color=C["grid"], lw=.5); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.savefig(f"{FIG}/fig3_d1.png", dpi=200, bbox_inches="tight"); print("fig3 saved")


def fig_nsubj():
    """Three curves, because the honest and the confounded sweep say different things:
    fitting the operator on the same n subjects removes the apparent low-n advantage,
    and the gap between the two corpus curves is the value of borrowing an operator."""
    series = [("no corpus", C["ink2"], lambda n: f"d1_n{n}_none_T2", None),
              ("+ corpus, operator fitted on the same n", C["blue"],
               lambda n: f"d1_cn{n}_learned_cal_T2", "d1_learned_cal_T2"),
              ("+ corpus, borrowed 32-subject operator", C["orange"],
               lambda n: f"d1_n{n}_learned_cal_T2", "d1_learned_cal_T2")]
    # the two corpus curves meet at n=32 by construction, so label them apart
    at = {"no corpus": -1, "+ corpus, operator fitted on the same n": 1,
          "+ corpus, borrowed 32-subject operator": -1}
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    any_ = False
    for lab, col, fn, fallback in series:
        xs, ys, es = [], [], []
        for n in [4, 8, 16, 32]:
            d = _load(f"{OUT}/{fn(n)}.json") or (_load(f"{OUT}/{fallback}.json") if (fallback and n == 32) else None)
            if not d: continue
            v = [r["ssim"] for r in d["test_rows"]]
            xs.append(n); ys.append(np.mean(v)); es.append(np.std(v, ddof=1))
        if not xs: continue
        any_ = True
        ax.errorbar(xs, ys, yerr=es, color=col, lw=2, marker="o", ms=5, capsize=2,
                    ls="--" if "borrowed" in lab else "-")
        i = at.get(lab, -1)
        if i == -1:
            ax.annotate(lab, (xs[-1], ys[-1]), color=col, fontsize=6.8, xytext=(6, 0),
                        textcoords="offset points", va="center", ha="left")
        else:   # tuck it under its own curve, between the other two
            ax.annotate(lab, (xs[i], ys[i]), color=col, fontsize=6.8, xytext=(0, -13),
                        textcoords="offset points", va="top", ha="center")
    if not any_:
        plt.close(fig); print("fig6: waiting on the paired-cohort sweep"); return
    ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16, 32]); ax.set_xticklabels([4, 8, 16, 32])
    ax.set_xlim(3.4, 150)
    ax.set_xlabel("paired training subjects"); ax.set_ylabel("masked SSIM")
    ax.set_title("the operator is the asset, not the cohort size", fontsize=9, color=C["ink"])
    style(ax)
    fig.savefig(f"{FIG}/fig6_nsubj.png", dpi=200, bbox_inches="tight"); print("fig6 saved")


def fig_d2():
    """Dot plot: a zero-based bar hides a 0.5 dB effect on a 17 dB scale, and zero
    is not a meaningful origin here - the do-nothing input is."""
    runs = {
        ("T1", "own"):   ("d2_T1only", None, None),
        ("T1", "other"): ("d2_T2toT1", None, None),
        ("T1", "both"):  ("d2_joint", "T1+T2", 0),
        ("T2", "own"):   ("d2_T2only", None, None),
        ("T2", "other"): ("d2_T1toT2", None, None),
        ("T2", "both"):  ("d2_joint", "T1+T2", 1),
    }
    label = {"own": "own contrast only", "other": "other contrast only", "both": "both contrasts"}
    col = {"own": C["ink2"], "other": C["orange"], "both": C["blue"]}
    donothing = {}
    for t in ("T1", "T2"):
        d = _load(f"{OUT}/base_identity_{t}.json")
        if d: donothing[t] = {k: np.mean([r[k] for r in d["test_rows"]]) for k in ("psnr", "ssim")}
    vals = {}
    for key, (tag, reg, ci) in runs.items():
        d = _load(f"{OUT}/{tag}.json")
        if not d: continue
        reg = reg or list(d["results"])[0]
        pre = "" if ci is None else f"c{ci}_"
        r = d["results"][reg]["rows"]
        vals[key] = {m: np.array([x[pre + m] for x in r]) for m in ("psnr", "ssim")}
    if not vals:
        print("fig4: waiting on runs"); return

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"wspace": 0.42})
    order = [("T1", "own"), ("T1", "other"), ("T1", "both"),
             ("T2", "own"), ("T2", "other"), ("T2", "both")]
    ypos = {k: y for k, y in zip(order, [4.6, 3.9, 3.2, 1.6, 0.9, 0.2])}
    for ax, m, nm in [(axes[0], "psnr", "masked PSNR (dB)"), (axes[1], "ssim", "masked SSIM")]:
        for t, y0, y1 in [("T1", 3.0, 4.8), ("T2", 0.0, 1.8)]:
            if t in donothing:
                ax.vlines(donothing[t][m], y0, y1, color=C["ink2"], lw=1, ls=(0, (4, 3)))
        for k in order:
            if k not in vals: continue
            v = vals[k][m]
            ax.plot([v.min(), v.max()], [ypos[k]] * 2, color=col[k[1]], lw=1.4, alpha=.35,
                    solid_capstyle="round")
            ax.plot(v.mean(), ypos[k], "o", color=col[k[1]], ms=8,
                    markeredgecolor=plt.rcParams["axes.facecolor"], markeredgewidth=1.2, zorder=3)
            ax.annotate(f"{v.mean():.3f}" if m == "ssim" else f"{v.mean():.2f}",
                        (v.mean(), ypos[k]), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=6.5, color=C["ink"])
        ax.set_yticks([3.9, 0.9]); ax.set_yticklabels(["3T T1\ntarget", "3T T2\ntarget"], fontsize=8)
        ax.set_ylim(-0.6, 5.15); ax.set_xlabel(nm)
        ax.grid(True, axis="x", color=C["grid"], lw=.5); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for k in ("own", "other", "both"):
        axes[0].plot([], [], "o", color=col[k], ms=7, label=label[k])
    axes[0].plot([], [], ls=(0, (4, 3)), color=C["ink2"], lw=1, label="64 mT input, unchanged")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper center",
                   bbox_to_anchor=(1.12, 1.17), ncol=4, handletextpad=.4, columnspacing=1.2)
    fig.savefig(f"{FIG}/fig4_d2.png", dpi=200, bbox_inches="tight"); print("fig4 saved")


def fig_d3():
    d = _load(f"{OUT}/d3_cqr_T2.json")
    if not d or "calibration_size" not in d:
        print("fig5: waiting on runs"); return
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.1), gridspec_kw={"wspace": .38})
    tgt = 1 - d["alpha"]

    ax = axes[0]
    for nm, col, lab in [("hetero", C["blue"], "heteroscedastic (CQR)"),
                         ("const", C["orange"], "constant width")]:
        if nm not in d: continue
        r = d[nm]["rows"]
        ax.scatter([x["width"] for x in r], [x["coverage"] for x in r], s=36, color=col,
                   edgecolor="white", lw=.9, zorder=3)
        ax.scatter([], [], s=36, color=col, label=lab)
    ax.axhline(tgt, color=C["ink2"], ls=(0, (4, 3)), lw=.9)
    ax.annotate(f"target {tgt:.2f}", (0.02, tgt), xycoords=("axes fraction", "data"),
                fontsize=6.5, color=C["ink2"], va="bottom")
    ax.set_xlabel("mean interval width"); ax.set_ylabel("per-subject coverage")
    ax.set_title("one dot per test subject", fontsize=8.5, color=C["ink"])
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    style(ax)

    ss = d["calibration_size"]; ks = sorted(int(k) for k in ss)
    ax = axes[1]
    m = np.array([ss[str(k)]["mean"] for k in ks])
    sd = np.array([ss[str(k)]["sd"] for k in ks])
    p05 = np.array([ss[str(k)]["p05"] for k in ks])
    ax.fill_between(ks, m - sd, m + sd, color=C["blue"], alpha=.18, lw=0)
    ax.plot(ks, m, color=C["blue"], lw=2, marker="o", ms=5)
    ax.plot(ks, p05, color=C["orange"], lw=2, marker="s", ms=5)
    ax.annotate("mean \u00b1 SD", (ks[-1], m[-1] + sd[-1]), xytext=(5, 2),
                textcoords="offset points", color=C["blue"], fontsize=6.5, va="bottom")
    ax.annotate("5th percentile", (ks[-1], p05[-1]), xytext=(5, 0), textcoords="offset points",
                color=C["orange"], fontsize=6.5, va="center")
    ax.axhline(tgt, color=C["ink2"], ls=(0, (4, 3)), lw=.9)
    ax.set_xlabel("calibration subjects"); ax.set_ylabel("per-subject coverage")
    ax.set_xlim(1, 17); ax.set_xticks(ks)
    ax.set_title("more calibration does not help", fontsize=8.5, color=C["ink"])
    style(ax)

    ax = axes[2]
    ax.plot(ks, sd, color=C["blue"], lw=2, marker="o", ms=5)
    ax.plot(ks, [ss[str(k)]["sd_split_mean"] for k in ks], color=C["orange"], lw=2,
            marker="s", ms=5)
    ax.annotate("total spread\nacross subjects", (ks[-1], sd[-1]), xytext=(5, 0),
                textcoords="offset points", color=C["blue"], fontsize=6.5, va="center")
    ax.annotate("calibration\nnoise only", (ks[-1], ss[str(ks[-1])]["sd_split_mean"]),
                xytext=(5, 0), textcoords="offset points", color=C["orange"], fontsize=6.5,
                va="center")
    ax.set_ylim(0, max(sd) * 1.25); ax.set_xlim(1, 17); ax.set_xticks(ks)
    ax.set_xlabel("calibration subjects"); ax.set_ylabel("SD of coverage")
    ax.set_title("the variance is the subjects", fontsize=8.5, color=C["ink"])
    style(ax)
    fig.savefig(f"{FIG}/fig5_d3.png", dpi=200, bbox_inches="tight"); print("fig5 saved")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k, fn in [("1", fig_spectra), ("2", fig_ceiling), ("3", fig_d1),
                  ("4", fig_d2), ("5", fig_d3), ("6", fig_nsubj)]:
        if which in ("all", k):
            try: fn()
            except Exception as e: print(f"fig{k}: {type(e).__name__}: {e}")
