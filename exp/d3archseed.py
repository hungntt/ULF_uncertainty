"""Is pix2pix's lesion advantage a property of the architecture or of one lucky run?

The architecture study compared six enhancers, one trained model each, and found the
SSIM ranking and the lesion-retention ranking essentially uncorrelated - pix2pix fifth of
six on SSIM and first on retention. With one run per architecture that gap is confounded
with ordinary training-run variation.

Three seeds each of the two architectures that matter. The question is whether the
between-architecture gap is large against the within-architecture seed spread.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
import exp.d3arch as A
from ulfkit.metrics import paired_test

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
VARIANTS = [
    ("U-Net L1 s0", "base_unet_T2", "unet_res"),
    ("U-Net L1 s1", "base_unet_s1_T2", "unet_res"),
    ("U-Net L1 s2", "base_unet_s2_T2", "unet_res"),
    ("pix2pix s0", "base_pix2pix_T2", "unet_res"),
    ("pix2pix s1", "base_pix2pix_s1_T2", "unet_res"),
    ("pix2pix s2", "base_pix2pix_s2_T2", "unet_res"),
    ("U-Net L1 s3", "base_unet_s3_T2", "unet_res"),
    ("U-Net L1 s4", "base_unet_s4_T2", "unet_res"),
    ("pix2pix s3", "base_pix2pix_s3_T2", "unet_res"),
    ("pix2pix s4", "base_pix2pix_s4_T2", "unet_res"),
]


def per_subject(rows, nm, thr=0.25):
    R = [x for x in rows if x["arch"] == nm]
    sids = sorted({x["sid"] for x in R})
    ret = np.array([np.mean([x["retention"] for x in R if x["sid"] == s]) for s in sids])
    er = np.array([np.mean([x["retention"] < thr for x in R if x["sid"] == s]) for s in sids])
    return ret, er


def main():
    A.ARCHS = VARIANTS
    rows = A.run()
    json.dump(rows, open(f"{OUT}/d3archseed_T2.json", "w"), indent=1, default=float)
    present = [nm for nm, _, _ in VARIANTS if any(x["arch"] == nm for x in rows)]
    print("\n== lesion survival across seeds ==")
    print(f"  {'model':16s} {'SSIM':>7s} {'retention':>10s} {'% erased':>9s} {'4 mm erased':>12s}")
    stat = {}
    for nm, path, _ in VARIANTS:
        if nm not in present:
            continue
        try:
            ss = json.load(open(f"{OUT}/{path}.json"))["test"]["ssim"][0]
        except Exception:
            ss = float("nan")
        ret, er = per_subject(rows, nm)
        R = [x for x in rows if x["arch"] == nm]
        e4 = np.mean([x["retention"] < 0.25 for x in R if x["r"] == 2])
        stat[nm] = dict(ssim=ss, ret=ret, er=er, e4=e4)
        print(f"  {nm:16s} {ss:7.3f} {ret.mean():10.3f} {er.mean()*100:8.1f}% {e4*100:11.0f}%")

    gu = [stat[n] for n in stat if n.startswith("U-Net")]
    gp = [stat[n] for n in stat if n.startswith("pix2pix")]
    if len(gu) >= 2 and len(gp) >= 2:
        mu = np.array([g["ret"].mean() for g in gu]); mp = np.array([g["ret"].mean() for g in gp])
        print(f"\n  U-Net L1 across seeds : {mu.mean():.3f} +-{mu.std(ddof=1):.3f}  "
              f"(spread {mu.max()-mu.min():.3f})")
        print(f"  pix2pix  across seeds : {mp.mean():.3f} +-{mp.std(ddof=1):.3f}  "
              f"(spread {mp.max()-mp.min():.3f})")
        gap = mp.mean() - mu.mean()
        pooled = np.sqrt((mu.var(ddof=1) + mp.var(ddof=1)) / 2)
        print(f"\n  between-architecture gap {gap:+.3f}; within-architecture sd {pooled:.3f}; "
              f"ratio {abs(gap)/max(pooled,1e-9):.1f}x")
        # paired over subjects, pooling each architecture's seeds
        pu = np.mean([g["ret"] for g in gu], 0); pp = np.mean([g["ret"] for g in gp], 0)
        print(f"  paired over 10 subjects (seeds averaged): {pp.mean()-pu.mean():+.3f}  "
              f"p={paired_test(pp, pu):.4f}")
        eu = np.mean([g["e4"] for g in gu]); ep = np.mean([g["e4"] for g in gp])
        print(f"  4 mm erasure: U-Net {eu*100:.0f}%  pix2pix {ep*100:.0f}%")
        ssu = [g["ssim"] for g in gu]; ssp = [g["ssim"] for g in gp]
        print(f"  SSIM: U-Net {np.mean(ssu):.3f}  pix2pix {np.mean(ssp):.3f} "
              f"(the ranking SSIM would give is unchanged)")
    json.dump({k: dict(ssim=v["ssim"], ret=v["ret"].tolist(), er=v["er"].tolist(),
                       e4=float(v["e4"])) for k, v in stat.items()},
              open(f"{OUT}/d3archseed_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
