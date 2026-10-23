"""Quantify the spectral gap between the clinical 3T target and the auxiliary 3T corpus."""
import json, sys
import numpy as np
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D

OUT = "/mnt/G/Projects/ULF-MICCAI/out"


def radial(a, nbins=128):
    """a: (N,H,W) float32 -> mean radial power spectrum."""
    A = np.abs(np.fft.fftshift(np.fft.fft2(a), axes=(-2, -1))) ** 2
    H, W = a.shape[-2:]
    yy, xx = np.mgrid[:H, :W]
    r = np.minimum(np.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2).astype(int), nbins - 1)
    cnt = np.bincount(r.ravel(), minlength=nbins)
    return np.bincount(r.ravel(), weights=A.mean(0).ravel(), minlength=nbins) / np.maximum(cnt, 1)


def subject_spectra(cohort, sids, which):
    out = {}
    for s in sids:
        idx = cohort.subject_rows(s)
        a = (cohort.hf[idx][:, 0] if which == "hf" else cohort.ulf[idx][:, 1]).astype(np.float32)
        out[s] = radial(a)
    return out


res = {}
for c in ["T1", "T2"]:
    coh, tr, val = D.dev_split(c)
    sp_hf = subject_spectra(coh, tr, "hf")        # clinical 3T target (POCEMR)
    sp_ulf = subject_spectra(coh, tr, "ulf")      # 64 mT input
    ixi = D.unpaired_3t(c)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(ixi), 3000, replace=False)
    sp_ixi = radial(ixi[np.sort(sel)][:, 0].astype(np.float32))
    ulfonly = D.unpaired_ulf(c)
    sel2 = rng.choice(len(ulfonly), 3000, replace=False)
    sp_ulfonly = radial(ulfonly[np.sort(sel2)][:, 1].astype(np.float32))

    H = np.mean([v for v in sp_hf.values()], 0)
    U = np.mean([v for v in sp_ulf.values()], 0)
    res[c] = dict(pocemr_3t=H.tolist(), pocemr_ulf=U.tolist(),
                  ixi_3t=sp_ixi.tolist(), ulfonly=sp_ulfonly.tolist(),
                  per_subject_3t={k: v.tolist() for k, v in sp_hf.items()})

    print(f"\n=== {c}: radial power, normalised to total power ===")
    nH, nU, nI, nO = [x / x.sum() for x in (H, U, sp_ixi, sp_ulfonly)]
    print(f"{'r':>5} {'POCEMR-3T':>11} {'POCEMR-ULF':>11} {'IXI-3T':>11} {'ULFonly':>11} "
          f"{'3T/ULF':>8} {'3T/IXI':>8}")
    for b in [8, 16, 32, 48, 64, 80, 96, 112]:
        print(f"{b:5d} {nH[b]:11.3e} {nU[b]:11.3e} {nI[b]:11.3e} {nO[b]:11.3e} "
              f"{nH[b]/nU[b]:8.2f} {nH[b]/nI[b]:8.2f}")
    for lo in [32, 64]:
        print(f"  HF power fraction (r>={lo}):  POCEMR-3T {nH[lo:].sum():.4f}   "
              f"POCEMR-ULF {nU[lo:].sum():.4f}   IXI-3T {nI[lo:].sum():.4f}   ULFonly {nO[lo:].sum():.4f}")
        print(f"     -> target/corpus HF ratio = {nH[lo:].sum()/nI[lo:].sum():.2f}x ; "
              f"target/input = {nH[lo:].sum()/nU[lo:].sum():.2f}x")
    # between-subject spread of the target, to see if the corpus gap exceeds normal variation
    fr = np.array([ (v/v.sum())[64:].sum() for v in sp_hf.values() ])
    print(f"  POCEMR-3T per-subject HF fraction (r>=64): {fr.mean():.4f} +- {fr.std(ddof=1):.4f} "
          f"[{fr.min():.4f}, {fr.max():.4f}]  | IXI {nI[64:].sum():.4f} "
          f"-> {(fr.mean()-nI[64:].sum())/fr.std(ddof=1):.1f} SD below the target cohort mean")
json.dump(res, open(f"{OUT}/d1_spectra.json", "w"))
print("\nsaved", f"{OUT}/d1_spectra.json")
