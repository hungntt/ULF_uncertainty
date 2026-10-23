"""Assemble every finished run into comparison tables (markdown + LaTeX)."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit.metrics import paired_test

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
KEYS = [("psnr", "PSNR", 3, 1), ("ssim", "SSIM", 4, 1), ("nrmse", "NRMSE", 4, -1),
        ("cos_LL", "cos LL", 3, 1), ("cos_HH", "cos HH", 3, 1),
        ("r_HH", "r HH", 3, 0), ("hfpow", "HF-pow", 3, 0),
        ("psnr_full", "PSNR-full", 2, 1), ("ssim_full", "SSIM-full", 4, 1)]


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def rowvals(d, key, pre=""):
    return [r[pre + key] for r in d["test_rows"]]


def table(entries, ref=None, keys=KEYS, title=""):
    """entries: list of (label, dict-with-test_rows). ref: label to test against."""
    print(f"\n### {title}")
    hdr = "| method | " + " | ".join(k[1] for k in keys) + " |"
    print(hdr); print("|" + "---|" * (len(keys) + 1))
    refd = dict(entries).get(ref)
    for lab, d in entries:
        cells = []
        for k, _, nd, _ in keys:
            v = rowvals(d, k)
            s = f"{np.mean(v):.{nd}f}±{np.std(v, ddof=1):.{nd}f}"
            if refd is not None and d is not refd and k in ("psnr", "ssim"):
                p = paired_test(v, rowvals(refd, k))
                if p == p and p < 0.05:
                    s += "*"
            cells.append(s)
        print(f"| {lab} | " + " | ".join(cells) + " |")


def section_baselines():
    for c in ["T2", "T1"]:
        ents = []
        for m, lab in [("identity", "64 mT input (do nothing)"), ("unet", "U-Net (L1)"),
                       ("unet_ssim", "U-Net (L1+SSIM)"), ("pix2pix", "cGAN / pix2pix"),
                       ("lohiresgan", "LoHiResGAN [Islam 2023]")]:
            d = load(f"{OUT}/base_{m}_{c}.json")
            if d: ents.append((lab, d))
        if ents: table(ents, ref="64 mT input (do nothing)", title=f"Baselines — {c}")


def section_d1():
    for c in ["T2", "T1"]:
        print(f"\n#### D1 zero-shot transfer to real 64 mT, before any finetuning ({c})")
        print("| pretraining corpus | dev SSIM | dev PSNR | dev cos LL |")
        print("|---|---|---|---|")
        for m, lab in [("handcrafted", "hand-crafted physics D"),
                       ("learned", "learned F, raw corpus"),
                       ("learned_cal", "learned F, calibrated corpus")]:
            d = load(f"{OUT}/d1_{m}_{c}.json")
            if d and d.get("pre_dev"):
                p_ = d["pre_dev"]
                print(f"| {lab} | {p_['ssim']:.4f} | {p_['psnr']:.3f} | {p_['cosLL']:+.3f} |")
        ents = []
        for m, lab in [("none", "real pairs only (32 subj)"),
                       ("handcrafted", "+ hand-crafted D pretrain"),
                       ("learned", "+ learned F pretrain"),
                       ("learned_cal", "+ learned F on spectrally calibrated corpus")]:
            d = load(f"{OUT}/d1_{m}_{c}.json")
            if d: ents.append((lab, d))
        if ents: table(ents, ref="real pairs only (32 subj)", title=f"D1 corpus amplification — {c}")


def section_d2():
    rows = []
    for tag in ["T1only", "T2only", "joint", "T2toT1", "T1toT2", "joint_drop"]:
        d = load(f"{OUT}/d2_{tag}.json")
        if not d: continue
        for reg, r in d["results"].items():
            cfg = d["cfg"]
            outs = [0, 1] if cfg["out"] == "both" else ([0] if cfg["out"] == "T1" else [1])
            for ci, oi in enumerate(outs):
                pre = "" if len(outs) == 1 else f"c{ci}_"
                v = [x[pre + "psnr"] for x in r["rows"]]
                s = [x[pre + "ssim"] for x in r["rows"]]
                rows.append((tag, reg, ["T1", "T2"][oi], np.mean(v), np.std(v, ddof=1),
                             np.mean(s), np.std(s, ddof=1)))
    if not rows: return
    print("\n### D2 cross-contrast fusion")
    print("| run | ULF input | target | PSNR | SSIM |"); print("|---|---|---|---|---|")
    for t, reg, o, pm, ps, sm, ss in rows:
        print(f"| {t} | {reg} | {o} | {pm:.3f}±{ps:.3f} | {sm:.4f}±{ss:.4f} |")


def section_d3():
    for c in ["T2", "T1"]:
        for m in ["cqr", "gauss"]:
            d = load(f"{OUT}/d3_{m}_{c}.json")
            if not d: continue
            print(f"\n### D3 {m} {c} (alpha={d['alpha']})")
            for nm in ["hetero", "const"]:
                if nm in d:
                    g = d[nm]["agg"]
                    print(f"  {nm:7s} Q={d[nm]['Q']:.4f}  coverage {g['coverage'][0]:.4f}"
                          f"±{g['coverage'][1]:.4f}  mean width {g['width'][0]:.4f}"
                          f"±{g['width'][1]:.4f}")
            if "crc" in d:
                for al, v in d["crc"].items():
                    print(f"  CRC a={al}: {'feasible' if v['feasible'] else 'INFEASIBLE (floor %.3f)'%v['alpha_floor']}"
                          f"  lambda={v['lambda']:.4f}  test risk {v['test_risk_mean']:.4f} "
                          f"(worst {v['test_risk_max']:.4f})")
            if "width_quality" in d:
                w = d["width_quality"]
                print(f"  width localises error: spearman {w['spearman']:.3f}, "
                      f"AUROC(top-10% err) {w['auroc_top10pct_error']:.3f}")
            if "calibration_size" in d:
                print("  calibration-set size study:")
                for k, v in d["calibration_size"].items():
                    print(f"    n={k:>2}: per-subject coverage {v['mean']:.4f}±{v['sd']:.4f}"
                          f"  5th pct {v['p05']:.4f}  SD(split mean)={v['sd_split_mean']:.4f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "base"): section_baselines()
    if which in ("all", "d1"): section_d1()
    if which in ("all", "d2"): section_d2()
    if which in ("all", "d3"): section_d3()
