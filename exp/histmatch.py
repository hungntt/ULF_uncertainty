"""Histogram-matching baseline: how much of the ULF->3T gap is a global intensity map?

The transfer function is fitted on the *training* 3T pool only, so no test
subject's own reference is used.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
NB = 4096


def cdf_of(arrs, nb=NB):
    h = np.zeros(nb, np.float64)
    for a in arrs:
        v = a[a > 0]
        h += np.histogram(v, bins=nb, range=(0, 1))[0]
    c = np.cumsum(h); return c / c[-1]


def main(contrast="T2"):
    coh, tr, val = D.dev_split(contrast)
    test = D.PairedCohort("test", contrast)
    rows_tr = coh.rows_for(tr)
    c_ulf = cdf_of([coh.ulf[rows_tr][:, 1].astype(np.float32)])
    c_hf = cdf_of([coh.hf[rows_tr][:, 0].astype(np.float32)])
    edges = np.linspace(0, 1, NB + 1)[1:]
    lut = np.interp(c_ulf, c_hf, edges)          # ULF value -> 3T value

    out = []
    for sid in test.subjects:
        idx = test.subject_rows(sid)
        u = test.ulf[idx][:, 1:2].astype(np.float32)
        h = test.hf[idx].astype(np.float32)
        p = np.where(u > 0, np.interp(u, edges, lut), 0.0).astype(np.float32)
        pt, ht, ut = (torch.from_numpy(z).cuda() for z in (p, h, u))
        r = {"sid": sid}
        r.update(M.volume_metrics(pt, ht, ut))
        r.update(M.band_metrics(pt, ht, ut))
        r["hfpow"] = M.hf_power_error(pt, ht)
        out.append(r)
    agg = M.aggregate(out)
    json.dump({"test": agg, "test_rows": out}, open(f"{OUT}/base_histmatch_{contrast}.json", "w"),
              indent=1, default=float)
    print(f"== histogram matching [{contrast}] ==")
    for k in ["psnr", "ssim", "nrmse", "cos_LL", "r_LL", "cos_HH", "hfpow", "psnr_full", "ssim_full"]:
        print(f"  {k:10s} {agg[k][0]:.4f} +- {agg[k][1]:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "T2")
