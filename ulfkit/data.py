"""Data access for the POCEMR paired ULF/3T cohort and the unpaired pools."""
import json, os
import numpy as np

DATA = os.environ.get("ULF_DATA", "/tmp/claude-17055/-mnt-G/5ec3b7fc-b916-4b70-887e-dbe4d6709b39/scratchpad/data")

# 8 of the 40 training subjects, held out for model selection. The 10-subject
# test list in manifest_test_* stays untouched until a run is final.
DEV_VAL_IDS = ["POCEMR005", "POCEMR023", "POCEMR038", "POCEMR064",
               "POCEMR080", "POCEMR092", "POCEMR099", "POCEMR104"]


def _load(name):
    return np.load(f"{DATA}/{name}.npy")


def manifest(split, contrast):
    return json.load(open(f"{DATA}/manifest_{split}_{contrast}.json"))["subjects"]


def slice_index(split, contrast):
    """(subject_id, z) -> row in the npy array."""
    idx = {}
    for s in manifest(split, contrast):
        for k, z in enumerate(s["z_indices"]):
            idx[(s["subject_id"], z)] = s["slice_start"] + k
    return idx


class PairedCohort:
    """ULF slabs + 3T centre slices for one contrast, keyed by subject."""

    def __init__(self, split, contrast):
        self.split, self.contrast = split, contrast
        self.ulf = _load(f"paired_{split}_ulf_{contrast}")   # (N,3,256,256) f16
        self.hf = _load(f"paired_{split}_hf_{contrast}")     # (N,1,256,256) f16
        self.man = manifest(split, contrast)
        self.index = slice_index(split, contrast)
        self.subjects = [s["subject_id"] for s in self.man]

    def rows_for(self, sids):
        sids = set(sids)
        r = [i for (sid, _), i in sorted(self.index.items()) if sid in sids]
        return np.array(sorted(r), dtype=np.int64)

    def subject_rows(self, sid):
        s = next(x for x in self.man if x["subject_id"] == sid)
        return np.arange(s["slice_start"], s["slice_start"] + s["slice_count"])


def dev_split(contrast):
    """Training subjects minus DEV_VAL_IDS, and DEV_VAL_IDS."""
    c = PairedCohort("train", contrast)
    val = [s for s in c.subjects if s in DEV_VAL_IDS]
    tr = [s for s in c.subjects if s not in DEV_VAL_IDS]
    return c, tr, val


class MultiContrastCohort:
    """Rows where both T1 and T2 exist for the same (subject, z)."""

    def __init__(self, split):
        self.t1 = PairedCohort(split, "T1")
        self.t2 = PairedCohort(split, "T2")
        keys = sorted(set(self.t1.index) & set(self.t2.index))
        self.keys = keys
        self.r1 = np.array([self.t1.index[k] for k in keys], dtype=np.int64)
        self.r2 = np.array([self.t2.index[k] for k in keys], dtype=np.int64)
        self.sid = np.array([k[0] for k in keys])

    def rows_for(self, sids):
        m = np.isin(self.sid, list(sids))
        return np.where(m)[0]

    def batch(self, pos):
        """-> ulf (B,2,256,256) centre slices, hf (B,2,256,256); channel order T1,T2."""
        a, b = self.r1[pos], self.r2[pos]
        ulf = np.stack([self.t1.ulf[a, 1], self.t2.ulf[b, 1]], 1)
        hf = np.stack([self.t1.hf[a, 0], self.t2.hf[b, 0]], 1)
        return ulf, hf

    def slab(self, pos):
        """-> ulf (B,6,256,256) both contrasts as 2.5D slabs."""
        a, b = self.r1[pos], self.r2[pos]
        return np.concatenate([self.t1.ulf[a], self.t2.ulf[b]], 1)


def unpaired_3t(contrast):
    return _load(f"fomo_hf_{contrast}")       # (M,1,256,256) IXI / FOMO-50k


def unpaired_ulf(contrast):
    return _load(f"ulfonly_{contrast}")       # (M,3,256,256) 48 ULF-only subjects
