"""Every number the ECO-MT write-up quotes, computed in one place.

Seed-replicated arms are summarised over seeds; the single-seed ladder arms are
reported as such. Subject-level contrasts pool (seed, subject) pairs because the
10 test subjects are the same in every run, so the pairing is exact.
"""
import glob, json, os, sys
import numpy as np
from scipy import stats

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
C = sys.argv[1] if len(sys.argv) > 1 else "T2"
KEYS = ["ssim", "psnr", "cos_LL", "cos_HH", "r_HH", "hfpow"]


def seeds_of(mode):
    s = [0] if os.path.exists(f"{OUT}/eco_{mode}_{C}.json") else []
    for f in glob.glob(f"{OUT}/eco_{mode}_s*_{C}.json"):
        s.append(int(f.split("_s")[-1].split("_")[0]))
    return sorted(set(s))


def load(mode, s):
    return json.load(open(f"{OUT}/eco_{mode}_{C}.json" if s == 0
                          else f"{OUT}/eco_{mode}_s{s}_{C}.json"))


LADDER = [("B1  paired-only", "b1"), ("B2  vanilla MT", "b2"),
          ("B3  teacher-variance gate", "b3"), ("B4  + MIND structural", "b4"),
          ("B5  calibrated risk (scalar)", "b5"), ("B6  observability only", "b6"),
          ("ECO-MT  scalar risk x obs", "ecomt"),
          ("B5v calibrated risk (voxel)", "b5vox"),
          ("ECO-MT-v  voxel risk x obs", "ecomtvox"),
          ("ctrl  random gate", "b2rand")]

print(f"# ECO-MT summary, {C}, locked 10-subject test\n")
print("## Per-arm, over seeds\n")
print("| arm | seeds | SSIM mean | SSIM SD | PSNR | cos_LL | best steps |")
print("|---|---|---|---|---|---|---|")
V, R = {}, {}
for nm, m in LADDER:
    ss = seeds_of(m)
    if not ss:
        continue
    d = [load(m, s) for s in ss]
    V[m] = np.array([x["test"]["ssim"][0] for x in d])
    R[m] = {s: {r["sid"]: r for r in load(m, s)["test_rows"]} for s in ss}
    sd = f"{V[m].std(ddof=1):.4f}" if len(ss) > 1 else "-"
    print(f"| {nm} | {len(ss)} | {V[m].mean():.4f} | {sd} | "
          f"{np.mean([x['test']['psnr'][0] for x in d]):.3f} | "
          f"{np.mean([x['test']['cos_LL'][0] for x in d]):.4f} | "
          f"{[x['info']['best_step'] for x in d]} |")

multi = [m for m in V if len(V[m]) > 2]
print("\n## Seed variance against the paired-only control\n")
print("| arm | SD | F vs B1 | p |")
print("|---|---|---|---|")
n1 = len(V["b1"]) - 1
for m in multi:
    if m == "b1":
        continue
    F = V["b1"].var(ddof=1) / V[m].var(ddof=1)
    n2 = len(V[m]) - 1
    p = 2 * min(stats.f.sf(F, n1, n2), stats.f.cdf(F, n1, n2))
    print(f"| {m} | {V[m].std(ddof=1):.4f} | {F:.1f} | {p:.4f} |")
print(f"\nB1 SD {V['b1'].std(ddof=1):.4f} over {len(V['b1'])} seeds "
      f"(range {V['b1'].min():.4f}-{V['b1'].max():.4f}).")

print("\n## Paired contrasts\n")
print("| contrast | per-seed dSSIM | mean | seeds +ve | subject-seed Wilcoxon p |")
print("|---|---|---|---|---|")
for a, b in [("b5vox", "b2rand"), ("b5vox", "b2"), ("b5vox", "b1"), ("b5vox", "ecomtvox"),
             ("ecomtvox", "b2rand"), ("b2rand", "b1"), ("b2", "b1")]:
    if a not in R or b not in R:
        continue
    ss = sorted(set(R[a]) & set(R[b]))
    per = [np.mean([R[a][s][k]["ssim"] - R[b][s][k]["ssim"] for k in R[a][s]]) for s in ss]
    allp = np.array([R[a][s][k]["ssim"] - R[b][s][k]["ssim"] for s in ss for k in sorted(R[a][s])])
    w = stats.wilcoxon(allp).pvalue if len(allp) > 5 else float("nan")
    print(f"| {a} - {b} | {' '.join(f'{x:+.4f}' for x in per)} | {np.mean(per):+.4f} | "
          f"{sum(x > 0 for x in per)}/{len(per)} | {w:.4f} |")

print("\n## Overfitting decay: dev-val masked SSIM, peak to step 12000\n")
print("| arm | peak | at 12000 | decay |")
print("|---|---|---|---|")
for nm, m in LADDER:
    if m not in R:
        continue
    h = load(m, seeds_of(m)[0]).get("hist")
    if not h:
        continue
    v = {s: x["score"] for s, x in h}
    pk, end = max(v.values()), v[max(v)]
    print(f"| {nm} | {pk:.4f} | {end:.4f} | {end - pk:+.4f} |")

cf = f"{OUT}/eco_calib_{C}.json"
if os.path.exists(cf):
    d = json.load(open(cf))
    print("\n## Gate weight actually applied (mean over the unpaired pool)\n")
    print("| arm | fine | med | coarse |")
    print("|---|---|---|---|")
    for nm, m in LADDER:
        if m not in R:
            continue
        g = load(m, seeds_of(m)[0]).get("gate")
        if g:
            print(f"| {nm} | " + " | ".join(f"{v:.3f}" for v in g[-1][2:5]) + " |")
