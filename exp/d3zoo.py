"""Do global scores rank ULF enhancers by lesion retention? The statistically defensible version.

Table 1 of the draft rested on six enhancers with one trained model each. Here:
  * the six architectures are trained with five seeds each at the SAME 12000-step budget
    (seed 0 is the Table-1 model; run_seeds12k.sh adds seeds 1-4);
  * nine further enhancers already trained in this programme join them, chosen to differ
    in training regime rather than to be weak: longer supervised schedule, corpus
    pretraining with a learned or a hand-designed physics operator, three Mean-Teacher
    variants on unpaired 64 mT, heteroscedastic and quantile regression heads, and a
    residual-shift wavelet diffusion model. Models trained on fewer subjects or on
    deliberately mismatched pairs are excluded: they would widen the range trivially.

Every model sees identical lesions at identical sites (d3arch.run, seed 11), so the seed-0
rows reproduce Table 1. Per model and test subject this stage writes: brain-masked
SSIM/PSNR on the ACQUIRED 64 mT input, lesion retention and % erased (closed loop), the
Gaussian-mixture tissue proxy, and the enhanced volume as NIfTI for SynthSeg.
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, models as Mo, train as T
from exp.d3arch import run as retention_run
from exp.d3detect import operator

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
NII = "/mnt/G/Projects/ULF-MICCAI/datasets/scratch/zoo_nii"

SIX = [  # (architecture, method tag used by baselines.py, seed-0 checkpoint, kind)
    ("U-Net L1", "unet", "base_unet_T2", "unet_res"),
    ("U-Net L1+SSIM", "unet_ssim", "base_unet_ssim_T2", "unet_res"),
    ("U-Net absolute", "unet_noresid", "base_unet_noresid_T2", "unet_abs"),
    ("cGAN / pix2pix", "pix2pix", "base_pix2pix_T2", "unet_res"),
    ("LoHiResGAN", "lohiresgan", "base_lohiresgan_T2", "lohires"),
    ("corpus-calibrated", None, "d1_learned_cal_T2", "unet_res"),
]
EXTRA = [  # (enhancer, checkpoint, kind)
    ("U-Net L1, long schedule", "d1_none_T2", "unet_res"),
    ("corpus, uncalibrated", "d1_learned_T2", "unet_res"),
    ("corpus, physics operator", "d1_handcrafted_T2", "unet_res"),
    ("Mean Teacher", "eco_b2_T2", "unet_abs"),
    ("Mean Teacher + MIND", "eco_b4_T2", "unet_abs"),
    ("gated Mean Teacher", "eco_ecomt_T2", "unet_abs"),
    ("Gaussian-NLL head", "d3_gauss_T2", "gauss"),
    ("quantile (CQR) head", "d3_cqr_T2", "cqr"),
    ("wavelet diffusion", "d3v2_diff_T2", "diff"),
]


def registry():
    """-> [(run_name, enhancer, seed, checkpoint, kind)]"""
    reg = []
    for arch, meth, ck0, kind in SIX:
        for s in range(5):
            if s == 0:
                ck = ck0
            elif meth is None:
                ck = f"d1_learned_cal_seed{s}_T2"
            else:
                ck = f"base_{meth}_seed{s}_T2"
            reg.append((f"{arch} s{s}", arch, s, ck, kind))
    for nm, ck, kind in EXTRA:
        reg.append((nm, nm, 0, ck, kind))
    return reg


class Head(torch.nn.Module):
    """Point prediction from a multi-output head: Gaussian mean (ch 0), CQR median (ch 1)."""
    def __init__(self, net, ch):
        super().__init__(); self.net, self.ch = net, ch
    def forward(self, x):
        return self.net(x)[:, self.ch:self.ch + 1]


class Diffusion(torch.nn.Module):
    """Posterior mean of K draws with a FIXED noise bank shared by every image, so the
    lesioned and lesion-free inputs see identical noise and their difference is the
    model's response to the lesion, not sampling noise."""
    def __init__(self, K=4, n_steps=16):
        super().__init__()
        import exp.d3v2 as d3
        from exp.d3v2_analyze import load as dload
        self.d3 = d3
        self.net, self.sch, self.ns = dload("T2", "")
        self.K, self.n = K, n_steps
        g = torch.Generator(device="cuda"); g.manual_seed(0)
        self.bank = torch.randn(K, n_steps, 1, 4, 128, 128, device="cuda", generator=g)

    @torch.no_grad()
    def forward(self, x):
        d3, sch = self.d3, self.sch
        ns = torch.ones(1, 4, 1, 1, device="cuda") if self.ns is None else self.ns.cuda()
        w_cond = d3.to_wav(x.float()) * ns.repeat(1, 3, 1, 1)
        w_lr = w_cond[:, 4:8]
        idxs = torch.linspace(sch.T - 1, 0, self.n).long().cuda()
        out = 0
        for k in range(self.K):
            z = w_lr + sch.gamma * sch.b[idxs[0]].sqrt() * self.bank[k, 0]
            for s in range(self.n):
                t = idxs[s].repeat(len(z))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x0 = self.net(torch.cat([z, w_cond], 1), t.float() / sch.T).float()
                if s < self.n - 1:
                    b = sch.b[idxs[s + 1]]
                    z = x0 + (w_lr - x0) * b + sch.gamma * b.sqrt() * self.bank[k, s + 1]
                else:
                    z = x0
            out = out + d3.from_wav(z / ns).clamp(0, 1)
        return out / self.K


def load(kind, ck):
    if kind == "diff":
        return Diffusion().eval()
    if kind == "lohires":
        n = Mo.LoHiResGen(3, 1)
    elif kind == "unet_abs":
        n = Mo.UNet(3, 1, base=32, residual_from=None)
    elif kind == "gauss":
        n = Mo.UNet(3, 2, base=32, residual_from=[1, 1])
    elif kind == "cqr":
        n = Mo.UNet(3, 3, base=32, residual_from=[1, 1, 1])
    else:
        n = Mo.UNet(3, 1, base=32, residual_from=(1, 2))
    n = n.cuda(); n.load_state_dict(torch.load(f"{OUT}/{ck}.pt")); n.eval()
    if kind == "gauss": return Head(n, 0).eval()
    if kind == "cqr": return Head(n, 1).eval()
    return n


@torch.no_grad()
def tissue_proxy(nets, contrast="T2", split="test"):
    """d3tissue.py's Gaussian-mixture proxy, per subject, for arbitrary nets."""
    from sklearn.mixture import GaussianMixture
    from exp.d3tissue import seg
    T.seed_all(0)
    test = D.PairedCohort(split, contrast)
    fwd = operator(contrast, "learned")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    per = {nm: [] for nm, _ in nets}
    for sid in test.subjects:
        idx = test.subject_rows(sid)
        acc = {nm: [] for nm, _ in nets}
        for i0 in range(0, len(idx), 8):
            j = idx[i0:i0 + 8]
            y = torch.from_numpy(test.hf[j].astype(np.float32)).cuda()
            x = fwd(y, gen)
            preds = {}
            for nm, net in nets:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    preds[nm] = net(x).float().clamp(0, 1).cpu().numpy()
            yc = y[:, 0].cpu().numpy()
            for k in range(len(j)):
                m = yc[k] > 0
                if m.sum() < 2000: continue
                v = yc[k][m]
                gm = GaussianMixture(3, random_state=0, max_iter=60).fit(v.reshape(-1, 1))
                ref = seg(v, gm)
                for nm in preds:
                    lab = seg(preds[nm][k, 0][m], gm)
                    acc[nm].append(np.mean([2 * ((ref == c) & (lab == c)).sum() /
                                            max((ref == c).sum() + (lab == c).sum(), 1)
                                            for c in range(3)]))
        for nm in per:
            per[nm].append(float(np.mean(acc[nm])))
        print(f"    tissue {sid}", flush=True)
    return per


@torch.no_grad()
def export_nii(name, net, contrast="T2", split="test"):
    """Enhanced test volumes from the ACQUIRED 64 mT input, for SynthSeg. Stored in array
    order (slice, row, col) with an identity affine; d3synthseg.py reorients to RAS."""
    import nibabel as nib
    test = D.PairedCohort(split, contrast)
    nii = nii_dir(split)
    os.makedirs(f"{nii}/{name}", exist_ok=True)
    for sid in test.subjects:
        idx = test.subject_rows(sid)
        ps = []
        for i in range(0, len(idx), 16):
            x = torch.from_numpy(test.ulf[idx[i:i + 16]].astype(np.float32)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ps.append(net(x).float().clamp(0, 1)[:, 0].cpu().numpy())
        vol = np.concatenate(ps) * (test.hf[idx][:, 0] > 0)
        nib.save(nib.Nifti1Image(vol.astype(np.float32), np.eye(4)),
                 f"{nii}/{name}/{sid}.nii.gz")


def nii_dir(split):
    return NII if split == "test" else f"{NII}_{split}"


def store_path(split):
    return f"{OUT}/d3zoo_T2.json" if split == "test" else f"{OUT}/d3zoo_{split}_T2.json"


def safe(nm):
    return nm.replace(" ", "_").replace("/", "-").replace(",", "").replace("+", "p")


def main(chunk=6, skip_done=True, split="test", reuse_from=None):
    """split: 'test' (POCEMR locked test) or an external cohort from prep_external.py.
    reuse_from: a split with byte-identical 3 T arrays (ext_on_hfe shares ext_on_hfc's GE 3 T).
    The closed-loop quantities (retention, erased, tissue proxy) are functions of the 3 T alone,
    so they are copied; only SSIM/PSNR on the acquired 64 mT input and the export are computed."""
    test = D.PairedCohort(split, "T2")
    if reuse_from:
        src = json.load(open(store_path(reuse_from)))
        ref = D.PairedCohort(reuse_from, "T2")
        assert np.array_equal(np.asarray(ref.hf), np.asarray(test.hf)), "3 T arrays differ"
    else:
        src = None
    in_fn = lambda rows: (test.ulf[rows].astype(np.float32),
                          test.ulf[rows][:, 1:2].astype(np.float32),
                          test.hf[rows].astype(np.float32))
    store_p = store_path(split)
    store = json.load(open(store_p)) if (skip_done and os.path.exists(store_p)) else {}
    todo = [r for r in registry() if r[0] not in store and (src is None or r[0] in src)]
    missing = [r for r in todo if r[4] != "diff" and not os.path.exists(f"{OUT}/{r[3]}.pt")]
    if missing:
        print("  not yet trained, skipped:", [r[0] for r in missing])
    todo = [r for r in todo if r not in missing]
    for c0 in range(0, len(todo), chunk):
        part = todo[c0:c0 + chunk]
        nets = [(r[0], load(r[4], r[3])) for r in part]
        print(f"\n  chunk: {[r[0] for r in part]}", flush=True)
        if src is None:
            rows = retention_run(nets=nets, split=split)
            tis = tissue_proxy(nets, split=split)
        else:
            rows, tis = [], {nm: src[nm]["tissue"] for nm, _ in nets}
        for (nm, enh, seed, ck, kind), (_, net) in zip(part, nets):
            q = T.evaluate(net, test, test.subjects, in_fn, bands=False)
            net.eval()                      # evaluate() leaves the model in train mode
            R = [x for x in rows if x["arch"] == nm]
            sids = [r["sid"] for r in q]
            if src is None:
                closed = dict(
                    ret=[float(np.mean([x["retention"] for x in R if x["sid"] == s])) for s in sids],
                    erased=[float(np.mean([x["retention"] < 0.25 for x in R if x["sid"] == s])) for s in sids],
                    erased4=float(np.mean([x["retention"] < 0.25 for x in R if x["r"] == 2])))
            else:
                assert src[nm]["sids"] == sids
                closed = {k: src[nm][k] for k in ("ret", "erased", "erased4")}
            store[nm] = dict(enhancer=enh, seed=seed, ckpt=ck, sids=sids,
                             ssim=[r["ssim"] for r in q], psnr=[r["psnr"] for r in q],
                             tissue=tis[nm], **closed)
            export_nii(safe(nm), net, split=split)
            print(f"  {nm:32s} SSIM {np.mean(store[nm]['ssim']):.3f}  ret {np.mean(store[nm]['ret']):.3f}"
                  f"  erased {np.mean(store[nm]['erased'])*100:.1f}%  tissue {np.mean(tis[nm]):.3f}", flush=True)
        json.dump(store, open(store_p, "w"), indent=1)
        del nets; torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--split", default="test")
    ap.add_argument("--reuse_from", default=None)
    a = ap.parse_args()
    main(a.chunk, split=a.split, reuse_from=a.reuse_from)
