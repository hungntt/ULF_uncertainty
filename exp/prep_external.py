"""Build paired 64 mT / 3 T arrays for two external cohorts, in the POCEMR array format.

Cohorts (raw data in datasets/raw):
  ext_on_hfc  OpenNeuro ds006557 (Vasa et al. 2025): 23 subjects, GE SIGNA Premier 3 T,
              Hyperfine Swoop session HFC (same day as the 3 T), axial T1w / T2w.
  ext_on_hfe  the same subjects and 3 T, second Swoop system (session HFE, <= 36 days later).
  ext_v5      Zenodo 20281403 (van den Broek et al. 2026): the 11 subjects with 3 T,
              Philips Achieva 3 T (acq-highres), Swoop session ses-01.

Each volume goes through the harmonisation of the POCEMR arrays
(FlowPoC utils/mri_preprocess.py): RAS -> SynthStrip -> N4 -> 1 mm, frame (in-plane
centroid and the middle 45% of the brain's axial extent) taken from the 3 T, the 64 mT
following the same frame, brain-mask 0.5/99.5 percentile scaling to [0, 1], and slab-safe
slices of the band. The POCEMR raw volumes arrived co-registered; these do not, so every
volume is rigidly registered to the subject's skull-stripped 3 T T1 on its 1 mm grid with
SynthMorph (FreeSurfer 8.2), which handles the change of contrast and field strength. One
frame (from the 3 T T1) is shared by all four volumes, so T1 and T2 rows of a (subject, z)
key show the same anatomy.

    python exp/prep_external.py run   [--only ext_v5] [--workers 6]
    python exp/prep_external.py pack  -> datasets/npy/paired_{split}_{ulf,hf}_{T1,T2}.npy
    python exp/prep_external.py qc    -> registration NCC and a montage per cohort
"""
import argparse, glob, json, os, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, "/mnt/G/Projects/FlowPoC")
ROOT = "/mnt/G/Projects/ULF-MICCAI"
RAW = f"{ROOT}/datasets/raw"
WORK = f"{ROOT}/datasets/scratch/external"
NPY = f"{ROOT}/datasets/npy"
FS = "/mnt/G/tools/fs/freesurfer"
os.environ.setdefault("FREESURFER_HOME", FS)
os.environ.setdefault("SYNTHSTRIP_BIN", f"{FS}/bin/mri_synthstrip")


def pairs():
    """-> [(split, subject_key, {hf_T1, hf_T2, lf_T1, lf_T2})]"""
    out = []
    on = f"{RAW}/ds006557"
    for d in sorted(glob.glob(f"{on}/sub-HYPE*")):
        s = os.path.basename(d)
        for ses, split in (("HFC", "ext_on_hfc"), ("HFE", "ext_on_hfe")):
            p = dict(hf_T1=f"{d}/ses-GE/anat/{s}_ses-GE_acq-MPRAGE_T1w.nii.gz",
                     hf_T2=f"{d}/ses-GE/anat/{s}_ses-GE_T2w.nii.gz",
                     lf_T1=f"{d}/ses-{ses}/anat/{s}_ses-{ses}_acq-axi_T1w.nii.gz",
                     lf_T2=f"{d}/ses-{ses}/anat/{s}_ses-{ses}_acq-axi_T2w.nii.gz")
            if all(os.path.exists(v) for v in p.values()):
                out.append((split, f"ON-{s[4:]}", p))
    v5 = f"{RAW}/zenodo_v5/ex"
    for d in sorted(glob.glob(f"{v5}/3T data/sub-*")):
        s = os.path.basename(d)
        p = dict(hf_T1=f"{d}/anat/{s}_acq-highres_T1w.nii.gz",
                 hf_T2=f"{d}/anat/{s}_acq-highres_T2w.nii.gz",
                 lf_T1=f"{v5}/64mT data/{s}/ses-01/anat/{s}_ses-01_run-1_T1w.nii.gz",
                 lf_T2=f"{v5}/64mT data/{s}/ses-01/anat/{s}_ses-01_run-1_T2w.nii.gz")
        if all(os.path.exists(v) for v in p.values()):
            out.append(("ext_v5", f"V5-{s[4:]}", p))
    return out


def strip_n4(path, out_path):
    """RAS -> SynthStrip -> N4 in native geometry (the POCEMR order); writes out_path."""
    import nibabel as nib
    from utils import mri_preprocess as mp
    img = mp.reorient_to_ras(mp.load_nifti(path))
    img, mask = mp.synthstrip(img)
    img = mp.n4_bias_correct(img, mask=mask)
    nib.save(img, out_path)
    return out_path


def register(moving, fixed, out, threads=4):
    """Rigid SynthMorph, moving resampled onto the fixed grid."""
    env = dict(os.environ, FREESURFER_HOME=FS)
    cmd = [f"{FS}/bin/mri_synthmorph", "register", "-m", "rigid", "-j", str(threads),
           "-o", out, moving, fixed]
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode or not os.path.exists(out):
        raise RuntimeError(f"synthmorph failed: {r.stderr[-400:]}")
    return out


def process(job):
    split, key, p = job
    import nibabel as nib
    from utils import mri_preprocess as mp
    sys.path.insert(0, "/mnt/G/Projects/FlowPoC")
    from data_process import extract_paired_slices
    od = f"{WORK}/{split}/{key}"
    if os.path.exists(f"{od}/arrays.npz"):
        return key, "cached"
    os.makedirs(od, exist_ok=True)
    try:
        # 3 T T1: native strip + N4, then the 1 mm reference grid
        strip_n4(p["hf_T1"], f"{od}/hf_T1_native.nii.gz")
        ref = mp.resample_to_isotropic(nib.load(f"{od}/hf_T1_native.nii.gz"), 1.0, order=1)
        nib.save(ref, f"{od}/hf_T1.nii.gz")
        for k in ("hf_T2", "lf_T1", "lf_T2"):
            strip_n4(p[k], f"{od}/{k}_native.nii.gz")
            register(f"{od}/{k}_native.nii.gz", f"{od}/hf_T1.nii.gz", f"{od}/{k}.nii.gz")
        vols, masks = {}, {}
        for k in ("hf_T1", "hf_T2", "lf_T1", "lf_T2"):
            v = np.asarray(nib.load(f"{od}/{k}.nii.gz").dataobj, dtype=np.float32)
            v[~np.isfinite(v)] = 0
            v = np.clip(v, 0, None)
            vols[k] = v
            masks[k] = v > 1e-3 * np.percentile(v[v > 0], 99)
        # the 3 T T1 sets the frame for all four volumes
        n1, m1, frame = mp.frame_and_normalize(vols["hf_T1"], masks["hf_T1"])
        normed = {"hf_T1": n1}
        for k in ("hf_T2", "lf_T1", "lf_T2"):
            normed[k], _, _ = mp.frame_and_normalize(vols[k], masks[k], frame=frame)
        res, meta = {}, {"split": split, "key": key, "paths": p,
                         "frame": [frame.z_start, frame.z_end, *frame.centroid_xy]}
        for c in ("T1", "T2"):
            # skull-stripped like POCEMR: zero outside the 3 T brain of this contrast
            hfm = normed[f"hf_{c}"] > 0
            ulf, hf, _, z = extract_paired_slices(normed[f"lf_{c}"] * (normed[f"lf_{c}"] > 0),
                                                  normed[f"hf_{c}"] * hfm,
                                                  slice_selection="middle", middle_fraction=1.0,
                                                  filter_empty=False)
            res[f"ulf_{c}"], res[f"hf_{c}"] = ulf.astype(np.float16), hf.astype(np.float16)
            meta[f"z_{c}"] = [int(x) for x in z]
        np.savez(f"{od}/arrays.npz", **res)
        json.dump(meta, open(f"{od}/meta.json", "w"), indent=1)
        for f in glob.glob(f"{od}/*_native.nii.gz"):
            os.remove(f)
        return key, f"ok {res['hf_T1'].shape[0]} slices"
    except Exception as e:  # noqa: BLE001 - report and continue with the other subjects
        return key, f"FAILED {type(e).__name__}: {e}"


def run(only=None, workers=6):
    jobs = [j for j in pairs() if only is None or j[0] in only]
    print(f"{len(jobs)} pairs", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for key, msg in ex.map(process, jobs):
            print(f"  {key}: {msg}", flush=True)


def pack():
    for split in sorted({j[0] for j in pairs()}):
        dirs = sorted(glob.glob(f"{WORK}/{split}/*/arrays.npz"))
        for c in ("T1", "T2"):
            U, H, man, off = [], [], [], 0
            for f in dirs:
                a = np.load(f); m = json.load(open(f.replace("arrays.npz", "meta.json")))
                U.append(a[f"ulf_{c}"]); H.append(a[f"hf_{c}"])
                n = a[f"hf_{c}"].shape[0]
                man.append(dict(subject_id=m["key"], split=split, contrast=c, slice_start=off,
                                slice_count=n, volume_depth=int(m["frame"][1] - m["frame"][0]),
                                z_indices=m[f"z_{c}"], raw=m["paths"][f"lf_{c}"]))
                off += n
            np.save(f"{NPY}/paired_{split}_ulf_{c}.npy", np.concatenate(U))
            np.save(f"{NPY}/paired_{split}_hf_{c}.npy", np.concatenate(H))
            json.dump(dict(contrast=c, total_slices=off, num_subjects=len(man), subjects=man),
                      open(f"{NPY}/manifest_{split}_{c}.json", "w"), indent=1)
            print(f"  {split} {c}: {len(man)} subjects, {off} slices")


def qc():
    """Registration quality: brain NCC of the 64 mT centre slice with the 3 T, per contrast,
    for each external cohort and for the POCEMR test set as the reference."""
    sys.path.insert(0, ROOT)
    from ulfkit import data as D
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = {}
    for split in ("test", "ext_on_hfc", "ext_on_hfe", "ext_v5"):
        for c in ("T1", "T2"):
            try:
                coh = D.PairedCohort(split, c)
            except FileNotFoundError:
                continue
            ncc = []
            for sid in coh.subjects:
                r = coh.subject_rows(sid)
                u = coh.ulf[r][:, 1].astype(np.float32); h = coh.hf[r][:, 0].astype(np.float32)
                m = h > 0
                ncc.append(float(np.corrcoef(u[m], h[m])[0, 1]))
            out[f"{split}/{c}"] = dict(n=len(ncc), ncc=float(np.mean(ncc)), min=float(np.min(ncc)))
            print(f"  {split:11s} {c}: {len(ncc)} subjects, brain NCC 64 mT vs 3 T "
                  f"{np.mean(ncc):.3f} (min {np.min(ncc):.3f})")
            if c == "T2":
                f, ax = plt.subplots(2, 6, figsize=(18, 6))
                for k, sid in enumerate(coh.subjects[:6]):
                    r = coh.subject_rows(sid); mid = r[len(r) // 2]
                    ax[0, k].imshow(coh.hf[mid, 0].astype(float), cmap="gray"); ax[0, k].set_title(sid)
                    ax[1, k].imshow(coh.ulf[mid, 1].astype(float), cmap="gray")
                for a in ax.ravel(): a.axis("off")
                plt.savefig(f"{WORK}/qc_{split}.png", dpi=50, bbox_inches="tight"); plt.close()
    json.dump(out, open(f"{ROOT}/out/prep_external_qc.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "pack", "qc", "list"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.stage == "list":
        for j in pairs(): print(j[0], j[1])
    elif a.stage == "run": run(a.only, a.workers)
    elif a.stage == "pack": pack()
    else: qc()
