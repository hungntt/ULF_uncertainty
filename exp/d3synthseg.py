"""A clinically used segmentation model as the downstream endpoint: SynthSeg.

The Gaussian-mixture proxy of d3tissue.py is fitted on the target and is not a model anyone
deploys. SynthSeg (Billot et al., 2023) is: it segments brain MRI of any contrast and
resolution, and is used on portable low-field scans. Here it segments the 3T reference and
every enhancer's output on the ACQUIRED 64 mT input (and the 64 mT input itself), and each
output is scored against SynthSeg on the 3T scan of the same subject:

  dice     mean Dice over the SynthSeg labels present in the reference, left and right
           merged (the in-plane left-right direction of the arrays is not recorded);
  vol_err  mean absolute relative volume error over seven structures used in volumetry:
           cerebral white matter, cortex, lateral ventricle, thalamus, caudate, putamen,
           hippocampus.

Volumes cover 61-65 contiguous 1 mm axial slices (mid-pons to the bodies of the lateral
ventricles), so structure volumes are partial; both arms see the same field of view.

  prepare  reorient d3zoo.py's NIfTIs (array order slice,row,col) to RAS-like axes
           (row, col=posterior->anterior, slice=inferior->superior); add 3T and 64 mT
  run      mri_synthseg --robust on every folder (FreeSurfer 8.2)
  score    -> out/d3zoo_synthseg_T2.json (or _{split}_ with ULF_SPLIT set)
"""
import glob, json, os, subprocess, sys
import numpy as np
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")

ROOT = "/mnt/G/Projects/ULF-MICCAI"
OUT = f"{ROOT}/out"
SPLIT = os.environ.get("ULF_SPLIT", "test")   # an external cohort from prep_external.py
SRC = f"{ROOT}/datasets/scratch/zoo_nii" + ("" if SPLIT == "test" else f"_{SPLIT}")
WORK = f"{ROOT}/datasets/scratch/synthseg" + ("" if SPLIT == "test" else f"_{SPLIT}")
RESULT = f"{OUT}/d3zoo_synthseg_T2.json" if SPLIT == "test" else f"{OUT}/d3zoo_synthseg_{SPLIT}_T2.json"
FS = "/mnt/G/tools/fs/freesurfer"
RIGHT_TO_LEFT = {41: 2, 42: 3, 43: 4, 44: 5, 46: 7, 47: 8, 49: 10, 50: 11, 51: 12, 52: 13,
                 53: 17, 54: 18, 58: 26, 60: 28}
VOL_LABELS = {"cerebral WM": 2, "cortex": 3, "lateral ventricle": 4, "thalamus": 10,
              "caudate": 11, "putamen": 12, "hippocampus": 17}
REF, ULF = "_3T_reference", "_64mT_input"


def prepare():
    import nibabel as nib
    from ulfkit import data as D
    test = D.PairedCohort(SPLIT, "T2")
    def save(arr, d, sid):
        os.makedirs(f"{WORK}/in/{d}", exist_ok=True)
        nib.save(nib.Nifti1Image(np.transpose(arr, (1, 2, 0)).astype(np.float32), np.eye(4)),
                 f"{WORK}/in/{d}/{sid}.nii.gz")
    for sid in test.subjects:
        idx = test.subject_rows(sid)
        hf = test.hf[idx][:, 0].astype(np.float32)
        save(hf, REF, sid)
        save(test.ulf[idx][:, 1].astype(np.float32) * (hf > 0), ULF, sid)
    for d in sorted(glob.glob(f"{SRC}/*")):
        name = os.path.basename(d)
        if os.path.exists(f"{WORK}/in/{name}") and \
           len(os.listdir(f"{WORK}/in/{name}")) == len(os.listdir(d)):
            continue
        for f in glob.glob(f"{d}/*.nii.gz"):
            save(nib.load(f).get_fdata(), name, os.path.basename(f)[:-7])
    print(f"  prepared {len(os.listdir(WORK + '/in'))} folders")


def run(threads=32, only=None, workers=1):
    """workers > 1 segments several folders at once, each with threads // workers cores."""
    from concurrent.futures import ThreadPoolExecutor
    env = dict(os.environ, FREESURFER_HOME=FS, FS_LICENSE=f"{FS}/license.txt")
    todo = []
    for d in sorted(glob.glob(f"{WORK}/in/*")):
        name = os.path.basename(d)
        if only and name not in only: continue
        o = f"{WORK}/seg/{name}"
        if os.path.exists(o) and len(os.listdir(o)) >= len(os.listdir(d)): continue
        os.makedirs(o, exist_ok=True)
        todo.append((name, d, o))
    def go(t):
        name, d, o = t
        cmd = [f"{FS}/bin/mri_synthseg", "--i", d, "--o", o, "--robust",
               "--threads", str(max(1, threads // workers)), "--cpu"]
        print("  ", name, flush=True)
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(go, todo))


def merged(a):
    a = a.astype(np.int32).copy()
    for r, l in RIGHT_TO_LEFT.items():
        a[a == r] = l
    return a


def score():
    import nibabel as nib
    from exp.d3zoo import registry, safe
    names = {safe(r[0]): r[0] for r in registry()}
    names[ULF] = "64 mT input (no enhancement)"
    ref = {os.path.basename(f)[:-7].replace("_synthseg", ""): merged(nib.load(f).get_fdata())
           for f in glob.glob(f"{WORK}/seg/{REF}/*.nii.gz")}
    sids = sorted(ref)
    res = {"sids": sids, "runs": {}}
    for d in sorted(glob.glob(f"{WORK}/seg/*")):
        name = os.path.basename(d)
        if name == REF or name not in names: continue
        dice, verr = [], []
        for sid in sids:
            f = glob.glob(f"{d}/{sid}*.nii.gz")
            if not f: break
            s, r = merged(nib.load(f[0]).get_fdata()), ref[sid]
            labs = [l for l in np.unique(r) if l != 0]
            dice.append(float(np.mean([2 * ((s == l) & (r == l)).sum() /
                                       ((s == l).sum() + (r == l).sum()) for l in labs])))
            verr.append(float(np.mean([abs((s == l).sum() - (r == l).sum()) / max((r == l).sum(), 1)
                                       for l in VOL_LABELS.values()])))
        if len(dice) == len(sids):
            res["runs"][names[name]] = {"dice": dice, "vol_err": verr}
            print(f"  {names[name]:34s} Dice {np.mean(dice):.3f}  |vol err| {np.mean(verr)*100:.1f}%")
    json.dump(res, open(RESULT, "w"), indent=1)


if __name__ == "__main__":
    stage = sys.argv[1]
    {"prepare": prepare, "run": run, "score": score}[stage]()
