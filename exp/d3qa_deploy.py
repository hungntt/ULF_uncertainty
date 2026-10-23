"""Does the QA map still mean anything where there is no 3T to check it against?

At deployment the reader has the 64 mT scan and the synthesised 3T, and nothing else.
The conformal threshold was chosen on subjects that *did* have a reference, and that
transfers only if the new subjects are exchangeable with the calibration ones. The 48
ULF-only subjects are the honest test: same scanner, no reference, never used for
anything. Error cannot be measured there, but the QA score distribution can, and if it
shifts, the guarantee does not hold.
"""
import argparse, json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T
from exp.d3qa import enhancer

OUT = "/mnt/G/Projects/ULF-MICCAI/out"


def subject_runs(x, thresh=0.95):
    """Split the unlabelled ULF stack where consecutive slices stop correlating."""
    a = torch.from_numpy(x[:, 1].astype(np.float32)).cuda()
    a = F.avg_pool2d(a[:, None], 4)[:, 0].flatten(1)
    a = (a - a.mean(1, keepdim=True)) / (a.std(1, keepdim=True) + 1e-6)
    c = (a[:-1] * a[1:]).mean(1)
    b = torch.cat([torch.tensor([0], device="cuda"), (c < thresh).nonzero()[:, 0] + 1,
                   torch.tensor([len(x)], device="cuda")]).unique().cpu().numpy()
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


@torch.no_grad()
def qa_scores(ulf, enh, qa, bs=32, budget=120_000, seed=0):
    Q = []
    for i in range(0, len(ulf), bs):
        x = torch.from_numpy(ulf[i:i + bs].astype(np.float32)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = enh(x).float().clamp(0, 1)
            q = qa(torch.cat([x, p], 1)).float()
        Q.append(q[x[:, 1:2] > 0])
    v = torch.cat(Q).cpu().numpy().astype(np.float32)
    if len(v) > budget:
        v = v[np.sort(np.random.default_rng(seed).choice(len(v), budget, replace=False))]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="T2")
    a = ap.parse_args()
    T.seed_all(0)
    enh = enhancer(); enh.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_{a.contrast}.pt")); enh.eval()
    qa = Mo.UNet(4, 1, base=32, residual_from=None).cuda()
    qa.load_state_dict(torch.load(f"{OUT}/d3qa_{a.contrast}.pt")); qa.eval()

    # reference distribution: the paired test subjects (these had a 3T, deployment will not)
    test = D.PairedCohort("test", a.contrast)
    ref = {s: qa_scores(test.ulf[test.subject_rows(s)], enh, qa) for s in test.subjects}

    ulf = D.unpaired_ulf(a.contrast)
    runs = subject_runs(ulf)
    print(f"[{a.contrast}] unlabelled pool: {len(ulf)} slices -> {len(runs)} subject runs "
          f"(median {int(np.median([b-a_ for a_,b in runs]))} slices)")
    dep = {f"ULF{i:03d}": qa_scores(ulf[s:e], enh, qa, seed=i) for i, (s, e) in enumerate(runs)}

    crc = json.load(open(f"{OUT}/d3qa_{a.contrast}.json"))["crc"]
    # alpha=0.10 is infeasible with 8 calibration subjects (floor 1/(n+1)=0.111),
    # so use the smallest feasible level that still flags a useful fraction
    key = next(k for k in ("0.15_qa", "0.2_qa") if k in crc)
    lam = crc[key]["lambda"]
    print(f"  using the alpha={key.split('_')[0]} threshold")
    from scipy.stats import ks_2samp
    rp = np.concatenate(list(ref.values())); dp = np.concatenate(list(dep.values()))
    res = {"contrast": a.contrast, "lambda": lam,
           "ks": float(ks_2samp(rp[::7], dp[::7]).statistic),
           "ref_flagged": {s: float((v > lam).mean()) for s, v in ref.items()},
           "dep_flagged": {s: float((v > lam).mean()) for s, v in dep.items()}}
    rf = np.array(list(res["ref_flagged"].values()))
    df = np.array(list(res["dep_flagged"].values()))
    for q in (50, 90, 99):
        res[f"p{q}_ref"] = float(np.percentile(rp, q)); res[f"p{q}_dep"] = float(np.percentile(dp, q))
    print(f"\n== QA score distribution: paired test (has a 3T) vs unlabelled pool (none) ==")
    for q in (50, 90, 99):
        print(f"  p{q}:  paired {res[f'p{q}_ref']:.5f}   unlabelled {res[f'p{q}_dep']:.5f}   "
              f"ratio {res[f'p{q}_dep']/max(res[f'p{q}_ref'],1e-9):.2f}")
    print(f"  two-sample KS statistic {res['ks']:.4f}")
    print(f"\n== fraction of brain flagged at the calibrated threshold {lam:.5f} ==")
    print(f"  paired test subjects   {rf.mean()*100:5.1f}% ± {rf.std(ddof=1)*100:.1f}  "
          f"[{rf.min()*100:.1f}, {rf.max()*100:.1f}]")
    print(f"  unlabelled subjects    {df.mean()*100:5.1f}% ± {df.std(ddof=1)*100:.1f}  "
          f"[{df.min()*100:.1f}, {df.max()*100:.1f}]")
    print(f"  -> {'consistent; the guarantee plausibly transfers' if abs(df.mean()-rf.mean())<0.05 else 'SHIFTED; recalibrate before deployment'}")
    json.dump(res, open(f"{OUT}/d3qa_deploy_{a.contrast}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
