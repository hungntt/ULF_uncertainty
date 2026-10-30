"""Does this cohort contain real pathology, and if so where?

Every lesion result in this programme is a digital phantom: Gaussian blobs inserted
into healthy-looking anatomy. The standing objection is that a simulated lesion is
not a real one. Before that can be answered, the cohort has to be screened - nobody
has told us which of these 50 subjects have findings.

Screen, not diagnose. A focal T2-hyperintense cluster is flagged as a CANDIDATE for
visual inspection; nothing here is a radiological read, and the output is a ranked
shortlist plus montages, not labels.

Method: per slice, compare the 3T T2 against a large-kernel median of itself (the
smooth tissue expectation), scale by a robust within-brain sigma, and keep compact
suprathreshold clusters that sit well inside the brain - which rejects the skull
rim, the CSF/parenchyma boundary and partial-volume edges.
"""
import json, os, sys
import numpy as np
import scipy.ndimage as ndi
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
FIG = "/mnt/G/Projects/ULF-MICCAI/datasets/scratch/real"
import os as _os
# thresholds overridable so the screen can be loosened to grow n without rewriting it
Z = float(_os.environ.get("REAL_Z", 4.0))
ZA = float(_os.environ.get("REAL_ZA", 3.0))
RUN = int(_os.environ.get("REAL_RUN", 3))
MIN_A, MAX_A, EDGE = 15, 300, 6
SCREEN_TAG = _os.environ.get("REAL_TAG", "")


def mirror_about_centroid(y, m):
    """Reflect the slice about the brain's own vertical midline."""
    cx = float(np.nonzero(m)[1].mean())
    xs = np.arange(y.shape[1])
    src = np.clip(np.rint(2 * cx - xs).astype(int), 0, y.shape[1] - 1)
    return y[:, src], m[:, src]


def screen_slice(y):
    """-> (area, peak_z, cy, cx) for compact, interior, ASYMMETRIC hyperintensities.

    Normal anatomy is broadly symmetric, so requiring a site to be much brighter than
    its contralateral counterpart rejects ventricles, cortex and deep grey, which a
    local-median test alone flags by the hundred.
    """
    m = y > 0
    if m.sum() < 2000:
        return []
    bg = ndi.median_filter(y, size=15)
    r = (y - bg) * m
    s = 1.4826 * np.median(np.abs(r[m] - np.median(r[m])))
    if s < 1e-6:
        return []
    z = r / s
    ymir, mmir = mirror_about_centroid(y, m)
    za = (y - ymir) / s                      # contralateral asymmetry, same scale
    deep = ndi.binary_erosion(m, np.ones((2 * EDGE + 1, 2 * EDGE + 1)))
    # CSF is the brightest tissue on T2 and dominates an unrestricted screen: the
    # first pass returned ventricle margins for all 50 subjects. Exclude it and its
    # surround, and require the site to sit on a white-matter-like background.
    hi = np.quantile(y[m], 0.90)
    csf = ndi.binary_dilation(y > hi, np.ones((9, 9)))
    wm = (bg > np.quantile(y[m], 0.25)) & (bg < np.quantile(y[m], 0.70))
    lab, n = ndi.label((z > Z) & (za > ZA) & deep & mmir & wm & ~csf)
    out = []
    for i in range(1, n + 1):
        sel = lab == i
        a = int(sel.sum())
        if not (MIN_A <= a <= MAX_A):
            continue
        ys, xs = np.nonzero(sel)
        h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
        if a / float(h * w) < 0.45 or max(h, w) / max(1, min(h, w)) > 3.0:
            continue
        out.append((a, float(z[sel].max()), int(ys.mean()), int(xs.mean())))
    return out


def persist(hits, rows):
    """Keep only sites recurring at the same place in >= RUN consecutive slices.

    A focal lesion is a 3D object. Noise, a vessel caught in cross-section and a
    partial-volume edge rarely survive this.
    """
    byrow = {}
    for h in hits:
        byrow.setdefault(h["row"], []).append(h)
    keep = []
    for h in hits:
        run = 1
        for d in (-1, 1):
            k = 1
            while h["row"] + d * k in byrow and any(
                    abs(o["cy"] - h["cy"]) <= 4 and abs(o["cx"] - h["cx"]) <= 4
                    for o in byrow[h["row"] + d * k]):
                run += 1; k += 1
        if run >= RUN:
            keep.append(dict(h, run=run))
    return keep


def main():
    os.makedirs(FIG, exist_ok=True)
    found = {}
    for split in ("train", "test"):
        c = D.PairedCohort(split, "T2")
        for sid in c.subjects:
            rows = c.subject_rows(sid)
            hits = []
            for r in rows:
                y = c.hf[r, 0].astype(np.float32)
                for (a, pz, cy, cx) in screen_slice(y):
                    hits.append(dict(row=int(r), area=a, z=pz, cy=cy, cx=cx))
            hits = persist(hits, rows)
            if hits:
                hits.sort(key=lambda h: (-h["run"], -h["z"]))
                found[sid] = dict(split=split, n=len(hits), top=hits[:40],
                                  max_z=hits[0]["z"], max_run=hits[0]["run"],
                                  max_area=max(h["area"] for h in hits))
            print(f"  {sid} ({split}): {len(hits)} candidate clusters", flush=True)
    rank = sorted(found.items(), key=lambda kv: -kv[1]["n"])
    print(f"\n{len(found)} of 50 subjects have at least one candidate\n")
    print(f"  {'subject':12s} {'split':6s} {'sites':>6s} {'max z':>7s} {'slices':>7s} {'max area':>9s}")
    for sid, d in rank[:12]:
        print(f"  {sid:12s} {d['split']:6s} {d['n']:6d} {d['max_z']:7.1f} {d['max_run']:7d} {d['max_area']:9d}")
    json.dump(found, open(f"{OUT}/d3real_screen{SCREEN_TAG}.json", "w"), indent=1)
    return rank


if __name__ == "__main__":
    main()
