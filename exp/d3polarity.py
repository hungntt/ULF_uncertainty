"""Does route B transfer the lesion, or does it apply a learned contrast inversion?

The cross-pipeline result said: synthesise 3T T1 from 64 mT T2, not from 64 mT T1
(retention 0.671 vs 0.153). That was measured with the clinically common polarity -
T2-hyperintense, T1-hypointense - which is also the relationship the model saw in
every training pair.

So the advantage may not be "T2 carries the lesion better". It may be "the model
knows T2-bright means T1-dark", which only helps while the lesion obeys that rule.
Lesions that violate it are routine: haemorrhage, fat, melanin, proteinaceous fluid
and contrast enhancement are all T1-hyperintense.

The sweep holds |c1| fixed and flips its sign, so magnitude cannot explain any gap.
A negative retention means the synthesised lesion has the WRONG SIGN - the reader is
shown a dark lesion where the truth is bright, which is worse than an erasure.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from exp.d3pipe import run
from ulfkit.metrics import paired_test

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
ROUTES = [("A_T1in", "A  64 mT T1"), ("B_T2in", "B  64 mT T2"), ("C_both", "C  both")]
FRACS = [(-1.0, "T1-hypo, strong (canonical)"), (-0.5, "T1-hypo, half (canonical)"),
         (+0.5, "T1-HYPER, half (violates)"), (+1.0, "T1-HYPER, strong (violates)")]


def agg(rows, key, thr=0.25):
    R = [x for x in rows if x["route"] == key]
    sids = sorted({x["sid"] for x in R})
    ret = np.array([np.mean([x["retention"] for x in R if x["sid"] == s]) for s in sids])
    inv = np.array([np.mean([x["retention"] < 0 for x in R if x["sid"] == s]) for s in sids])
    er = np.array([np.mean([x["retention"] < thr for x in R if x["sid"] == s]) for s in sids])
    return ret, er, inv


def main(cmodel="bothtoT1", label=""):
    store = {}
    print("lesion survival in synthesised 3T T1 as the T1/T2 polarity relationship changes")
    print("retention is measured against the 3T T1 lesion; negative = inverted polarity\n")
    print(f"  {'polarity':30s}" + "".join(f"{nm:>26s}" for _, nm in ROUTES))
    print(f"  {'':30s}" + "".join(f"{'ret':>10s}{'%eras':>8s}{'%inv':>8s}" for _ in ROUTES))
    for frac, label in FRACS:
        rows = run(c1_frac=frac, cmodel=cmodel)
        json.dump(rows, open(f"{OUT}/d3polarity{label}_{frac}.json", "w"), indent=1, default=float)
        store[str(frac)] = {}
        cells = ""
        for key, _ in ROUTES:
            ret, er, inv = agg(rows, key)
            store[str(frac)][key] = dict(ret=ret.tolist(), erased=er.tolist(), inverted=inv.tolist())
            cells += f"{ret.mean():10.3f}{er.mean()*100:7.0f}%{inv.mean()*100:7.0f}%"
        print(f"  {label:30s}{cells}", flush=True)

    print("\n  B vs A at each polarity (paired over 10 subjects):")
    for frac, label in FRACS:
        a = np.array(store[str(frac)]["A_T1in"]["ret"]); b = np.array(store[str(frac)]["B_T2in"]["ret"])
        print(f"    {label:30s} retention {b.mean()-a.mean():+.3f}  p={paired_test(b, a):.4f}")
    json.dump(store, open(f"{OUT}/d3polarity{label}.json", "w"), indent=1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmodel", default="bothtoT1", help="checkpoint behind route C")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    print(f"route C = d2_{a.cmodel}.pt\n")
    main(a.cmodel, a.label)
