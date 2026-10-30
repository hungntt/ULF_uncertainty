"""Does a publicly released synthesis tool invert coupling-violating lesions too?

d3lesiontype.py shows the inversion for models trained in this programme. SynthSR
(Iglesias et al., 2023) is a released tool that turns a scan of any contrast into a 1 mm
MPRAGE-like T1, and FreeSurfer ships a Hyperfine variant that takes the 64 mT T1 and T2
together. Both are the clinical counterpart of routes B and C.

Per locked test subject (3D, 61-65 contiguous 1 mm axial slices): 3D Gaussian lesions are
inserted into the 3T T1 and T2 at well-separated interior sites, with the four T1/T2 sign
patterns of d3lesiontype.py at equal magnitude; both contrasts pass slice-wise through the
learned 64 mT operators; each SynthSR variant runs on the lesioned and on the lesion-free
volumes. A lesion is inverted when the core-minus-ring change of the synthesised T1 has the
opposite sign to that of the 3T T1 (scale-free, so SynthSR's intensity scale is irrelevant).

  make     write the simulated 64 mT volumes (lesioned per quadrant, and lesion-free)
  run      SynthSR variants on every volume
  score    -> out/d3synthsr.json
"""
import glob, json, os, subprocess, sys
import numpy as np
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")

ROOT = "/mnt/G/Projects/ULF-MICCAI"
OUT = f"{ROOT}/out"
WORK = f"{ROOT}/datasets/scratch/synthsr"
FS = "/mnt/G/tools/fs/freesurfer"
QUADS = [("t1d_t2b", -1, +1), ("t1b_t2d", +1, -1), ("t1b_t2b", +1, +1), ("t1d_t2d", -1, -1)]
R_VOX, C, N_LES, MIN_SEP = 4.0, 0.25, 12, 28


def to_ras(v):
    """(slice,row,col) -> (row, col=posterior->anterior, slice=inferior->superior)."""
    return np.transpose(v, (1, 2, 0))


def sites_3d(y1, y2, rng):
    """Interior, mid-intensity in both contrasts, pairwise >= MIN_SEP voxels apart."""
    from scipy.ndimage import binary_erosion
    m = binary_erosion((y1 > 0) & (y2 > 0), iterations=6)
    ok = m & (y1 > 0.12) & (y1 < 0.6) & (y2 > 0.12) & (y2 < 0.45)
    ok[:, :, :6] = ok[:, :, -6:] = False                  # keep the sphere inside the slab
    cand = np.argwhere(ok); rng.shuffle(cand)
    picked = []
    for c in cand:
        if all(np.linalg.norm(c - p) >= MIN_SEP for p in picked):
            picked.append(c)
        if len(picked) == N_LES: break
    return np.array(picked)


def blob3d(shape, centres):
    zz = np.indices(shape).astype(np.float32)
    out = np.zeros(shape, np.float32); lab = np.zeros(shape, np.int16)
    for k, c in enumerate(centres):
        d2 = sum((zz[i] - c[i]) ** 2 for i in range(3))
        b = np.exp(-2.0 * d2 / R_VOX ** 2)
        out = np.maximum(out, b); lab[b > 0.05] = k + 1
    return out, lab


def make():
    import nibabel as nib, torch
    from ulfkit import data as D
    from exp.d3detect import operator
    mc = D.MultiContrastCohort("test")
    f1, f2 = operator("T1", "learned"), operator("T2", "learned")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    rng = np.random.default_rng(0)
    meta = {}

    def degrade(f, vol):                                   # slice-wise learned operator
        out = []
        for i in range(0, len(vol), 16):
            y = torch.from_numpy(vol[i:i + 16, None].astype(np.float32)).cuda()
            with torch.no_grad():
                out.append(f(y, gen)[:, 0].cpu().numpy())
        return np.concatenate(out)

    for sid in sorted(set(mc.sid)):
        pos = np.where(mc.sid == sid)[0]
        _, hf = mc.batch(pos)
        y1, y2 = to_ras(hf[:, 0].astype(np.float32)), to_ras(hf[:, 1].astype(np.float32))
        cen = sites_3d(y1, y2, rng)
        b, lab = blob3d(y1.shape, cen)
        b *= (y1 > 0)
        meta[sid] = dict(centres=cen.tolist())
        d = f"{WORK}/in/{sid}"; os.makedirs(d, exist_ok=True)
        save = lambda v, n: nib.save(nib.Nifti1Image(v.astype(np.float32), np.eye(4)), f"{d}/{n}.nii.gz")
        save(y1, "3t_t1_clean"); save(lab.astype(np.float32), "lesion_labels")
        save(b, "lesion_profile")
        bk = lambda v: to_ras(v)
        x1 = bk(degrade(f1, np.transpose(y1, (2, 0, 1)))) * (y1 > 0)
        x2 = bk(degrade(f2, np.transpose(y2, (2, 0, 1)))) * (y2 > 0)
        save(x1, "ulf_t1_clean"); save(x2, "ulf_t2_clean")
        for q, s1, s2 in QUADS:
            y1l = np.clip(y1 + s1 * C * b, 0, 1); y2l = np.clip(y2 + s2 * C * b, 0, 1)
            save(y1l, f"3t_t1_{q}")
            save(bk(degrade(f1, np.transpose(y1l, (2, 0, 1)))) * (y1 > 0), f"ulf_t1_{q}")
            save(bk(degrade(f2, np.transpose(y2l, (2, 0, 1)))) * (y2 > 0), f"ulf_t2_{q}")
        print(f"  {sid}: {len(cen)} lesions", flush=True)
    json.dump(meta, open(f"{WORK}/meta.json", "w"))


NATIVE = (1.5, 1.5, 5.0)      # Hyperfine axial acquisition, the input mri_synthsr_hyperfine expects


def to_native(src, dst):
    """1 mm volume -> 1.5 x 1.5 x 5 mm axial: anti-alias, subsample, and shift the affine so
    each new voxel centre sits at the centre of the block it summarises."""
    import nibabel as nib
    from scipy.ndimage import gaussian_filter, zoom
    img = nib.load(src)
    v = gaussian_filter(img.get_fdata(), [s / 2.355 for s in NATIVE])   # FWHM = new spacing
    v = zoom(v, [1 / s for s in NATIVE], order=1)
    aff = img.affine.copy()
    aff[:3, :3] = img.affine[:3, :3] @ np.diag(NATIVE)
    aff[:3, 3] = img.affine[:3, :3] @ ((np.array(NATIVE) - 1) / 2) + img.affine[:3, 3]
    nib.save(nib.Nifti1Image(v.astype(np.float32), aff), dst)


def run(threads=10, workers=3, geom="native"):
    """SynthSR (low-field model) from T1 alone and from T2 alone, and the Hyperfine T1+T2
    variant. geom='native' feeds 1.5 x 1.5 x 5 mm inputs (primary); '1mm' feeds the 1 mm grid."""
    env = dict(os.environ, FREESURFER_HOME=FS, FS_LICENSE=f"{FS}/license.txt")
    suf = "_native" if geom == "native" else ""
    jobs = []
    for d in sorted(glob.glob(f"{WORK}/in/*")):
        o = d.replace("/in/", "/out/"); os.makedirs(o, exist_ok=True)
        for q in ["clean"] + [q for q, _, _ in QUADS]:
            t1, t2 = f"{d}/ulf_t1_{q}{suf}.nii.gz", f"{d}/ulf_t2_{q}{suf}.nii.gz"
            if suf:
                for c in ("t1", "t2"):
                    if not os.path.exists(f"{d}/ulf_{c}_{q}{suf}.nii.gz"):
                        to_native(f"{d}/ulf_{c}_{q}.nii.gz", f"{d}/ulf_{c}_{q}{suf}.nii.gz")
            for tag, cmd in (
                    ("sr_t1", [f"{FS}/bin/mri_synthsr", "--lowfield", "--i", t1, "--o", f"{o}/sr_t1_{q}{suf}.nii.gz"]),
                    ("sr_t2", [f"{FS}/bin/mri_synthsr", "--lowfield", "--i", t2, "--o", f"{o}/sr_t2_{q}{suf}.nii.gz"]),
                    ("sr_hf", [f"{FS}/bin/mri_synthsr_hyperfine", "--t1", t1, "--t2", t2,
                               "--o", f"{o}/sr_hf_{q}{suf}.nii.gz"])):
                if not os.path.exists(f"{o}/{tag}_{q}{suf}.nii.gz"):
                    jobs.append(cmd + ["--threads", str(threads), "--cpu"])
    print(f"  {len(jobs)} SynthSR runs", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    def go(cmd):
        r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode: print("  FAILED", cmd[-5], r.stderr[-300:], flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(go, jobs))


def resample_to(src_img, ref_img):
    """Trilinear resampling of src onto ref's voxel grid via the two affines."""
    from scipy.ndimage import map_coordinates
    ref_ijk = np.indices(ref_img.shape).reshape(3, -1)
    world = ref_img.affine @ np.vstack([ref_ijk, np.ones(ref_ijk.shape[1])])
    src_ijk = np.linalg.inv(src_img.affine) @ world
    v = map_coordinates(src_img.get_fdata(), src_ijk[:3], order=1, mode="nearest")
    return v.reshape(ref_img.shape)


def score(geom="native"):
    import nibabel as nib
    suf = "_native" if geom == "native" else ""
    rows, fit = [], {}
    for d in sorted(glob.glob(f"{WORK}/in/*")):
        sid, o = os.path.basename(d), d.replace("/in/", "/out/")
        ref = nib.load(f"{d}/3t_t1_clean.nii.gz")
        y1 = ref.get_fdata(); lab = nib.load(f"{d}/lesion_labels.nii.gz").get_fdata()
        b = nib.load(f"{d}/lesion_profile.nii.gz").get_fdata()
        for tag in ("sr_t1", "sr_t2", "sr_hf"):
            f0 = f"{o}/{tag}_clean{suf}.nii.gz"
            if not os.path.exists(f0): continue
            p0 = resample_to(nib.load(f0), ref)
            m = y1 > 0
            k = np.polyfit(y1[m], p0[m], 1)[0]              # SynthSR intensity per 3T unit
            fit.setdefault(tag, []).append(float(np.corrcoef(p0[m], y1[m])[0, 1]))
            for q, s1, s2 in QUADS:
                fq = f"{o}/{tag}_{q}{suf}.nii.gz"
                if not os.path.exists(fq): continue
                dp = resample_to(nib.load(fq), ref) - p0
                dy = nib.load(f"{d}/3t_t1_{q}.nii.gz").get_fdata() - y1
                for L in range(1, int(lab.max()) + 1):
                    near = lab == L
                    core, ring = near & (b > 0.6), near & (b > 0.05) & (b < 0.25)
                    if core.sum() < 8 or ring.sum() < 16: continue
                    ty = dy[core].mean() - dy[ring].mean()
                    tp = dp[core].mean() - dp[ring].mean()
                    if abs(ty) < 0.25 * C: continue
                    rows.append(dict(sid=sid, tool=tag, quad=q, retention=float(tp / k / ty),
                                     inverted=bool(np.sign(tp) != np.sign(ty))))
    json.dump(rows, open(f"{OUT}/d3synthsr_rows{suf}.json", "w"))
    res = {"corr_with_3T_T1": {t: float(np.mean(v)) for t, v in fit.items()}}
    print("  lesion-free output vs 3T T1, brain correlation:",
          {t: round(float(np.mean(v)), 3) for t, v in fit.items()})
    rng = np.random.default_rng(0)
    for tag in ("sr_t1", "sr_t2", "sr_hf"):
        for q, _, _ in QUADS:
            R = [r for r in rows if r["tool"] == tag and r["quad"] == q]
            if not R: continue
            sids = sorted({r["sid"] for r in R})
            inv = np.array([np.mean([r["inverted"] for r in R if r["sid"] == s]) for s in sids])
            bs = [inv[rng.integers(len(inv), size=len(inv))].mean() for _ in range(10000)]
            res[f"{tag}/{q}"] = dict(n=len(R), inverted=float(inv.mean()),
                                     ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                     ret=float(np.median([r["retention"] for r in R])))
            print(f"  {tag:6s} {q:8s} n={len(R):3d}  wrong sign {inv.mean()*100:5.1f}% "
                  f"[{np.percentile(bs, 2.5)*100:.0f},{np.percentile(bs, 97.5)*100:.0f}]  "
                  f"median retention {res[f'{tag}/{q}']['ret']:+.2f}")
    json.dump(res, open(f"{OUT}/d3synthsr{suf}.json", "w"), indent=1)


if __name__ == "__main__":
    geom = sys.argv[2] if len(sys.argv) > 2 else "native"
    if sys.argv[1] == "make": make()
    else: {"run": run, "score": score}[sys.argv[1]](geom=geom)
