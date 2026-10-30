"""Which lesion types does cross-contrast synthesis invert?

Healthy anatomy couples T1 and T2 inversely (fluid and oedema are T2-bright, T1-dark).
A model that synthesises 3T T1 with access to 64 mT T2 can learn that coupling instead
of reading the T1 input. d3polarity.py flipped only the T1 sign with the lesion always
T2-bright. Here both signs vary, so every T1/T2 polarity quadrant is tested, each tied
to the lesion types that occupy it in standard neuroradiology descriptions:

  T1-dark,   T2-bright   canonical: white-matter disease, oedema, infarct, demyelination
  T1-bright, T2-dark     still inversely coupled: melanin, early-subacute (intracellular
                         methaemoglobin) haemorrhage, some calcification
  T1-bright, T2-bright   violates: late-subacute (extracellular methaemoglobin)
                         haemorrhage, fat, proteinaceous fluid
  T1-dark,   T2-dark     violates: acute (deoxyhaemoglobin) haemorrhage, haemosiderin,
                         some calcification

T1 and T2 magnitudes are equal, so only the sign pattern changes. Sites are mid-intensity
in both contrasts so a dark lesion cannot clip. Route C is scored for four independently
trained checkpoints (two of them trained with polarity-mixed lesions, one also with a
lesion-weighted loss), so inversion cannot be a property of one training run.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from exp.d3pipe import run

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
QUADS = [  # (key, T1 sign, T2 sign, label, examples)
    ("t1d_t2b", -1, +1, "T1-dark / T2-bright", "WM disease, oedema, infarct"),
    ("t1b_t2d", +1, -1, "T1-bright / T2-dark", "melanin, early-subacute blood"),
    ("t1b_t2b", +1, +1, "T1-bright / T2-bright", "late-subacute blood, fat, protein"),
    ("t1d_t2d", -1, -1, "T1-dark / T2-dark", "acute blood, haemosiderin"),
]
EXTRA = ("bothtoT1_ctl", "bothtoT1_aug", "bothtoT1_augw")
ROUTES = ["A_T1in", "B_T2in", "C_both"] + [f"C_{t}" for t in EXTRA]


def per_subject(rows, route, thr=0.25):
    R = [x for x in rows if x["route"] == route]
    sids = sorted({x["sid"] for x in R})
    f = lambda key: np.array([np.mean([key(x) for x in R if x["sid"] == s]) for s in sids])
    return dict(ret=f(lambda x: x["retention"]), erased=f(lambda x: x["retention"] < thr),
                inverted=f(lambda x: x["retention"] < 0), n=len(R))


def boot_ci(v, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = v[rng.integers(len(v), size=(n, len(v)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(split="test"):
    sfx = "" if split == "test" else f"_{split}"
    store = {}
    for key, s1, s2, label, ex in QUADS:
        rows = run(c1_frac=float(s1), c2_sign=float(s2), extra_c=EXTRA, both_pool=True,
                   split=split)
        json.dump(rows, open(f"{OUT}/d3lesiontype_rows_{key}{sfx}.json", "w"), default=float)
        store[key] = dict(label=label, examples=ex, routes={})
        for r in ROUTES:
            ps = per_subject(rows, r)
            store[key]["routes"][r] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                       for k, v in ps.items()}
            store[key]["routes"][r]["inv_ci"] = boot_ci(ps["inverted"])
            store[key]["routes"][r]["ret_ci"] = boot_ci(ps["ret"])
        json.dump(store, open(f"{OUT}/d3lesiontype{sfx}.json", "w"), indent=1)

    print("\n== % of lesions rendered with the wrong sign in synthesised 3T T1 ==")
    print("   (mean over 10 test subjects, 95% subject-bootstrap CI)\n")
    print(f"  {'quadrant':24s}" + "".join(f"{r:>22s}" for r in ROUTES))
    for key, *_ , label, ex in QUADS:
        cells = ""
        for r in ROUTES:
            d = store[key]["routes"][r]
            lo, hi = d["inv_ci"]
            cells += f"{np.mean(d['inverted'])*100:8.0f}% [{lo*100:3.0f},{hi*100:3.0f}]    "
        print(f"  {label:24s}{cells}")
    print("\n== mean retention (negative = inverted) ==")
    for key, *_ , label, ex in QUADS:
        print(f"  {label:24s}" + "".join(f"{np.mean(store[key]['routes'][r]['ret']):22.3f}"
                                        for r in ROUTES))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
