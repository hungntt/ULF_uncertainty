"""The usability number: at the certified threshold, what fraction of BRAIN is flagged?

The finding-aware CRC table reports the fraction of lesion *sites* crossing the
threshold. Sites are drawn from eroded mid-intensity tissue, so that is not the
quantity a reader pays, which is how much of the image carries a warning.

The map is conditional on the lesion it is asked about, so a single number is
misleading. The honest output is size-stratified: "for a lesion this big and this
faint, how much of the brain is unreliable?"
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, models as Mo, train as T
from exp.d3qa import enhancer
from exp.d3detect import operator

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
SIZES = [(2, "4 mm"), (3, "6 mm"), (4, "8 mm"), (6, "12 mm")]
CONTRASTS = [(0.12, "faint"), (0.25, "clear")]


@torch.no_grad()
def flagged_fraction(test, enh, det, fwd, gen, lam, r, c):
    fr = []
    for sid in test.subjects:
        idx = test.subject_rows(sid)
        f = n = 0.0
        for i in range(0, len(idx), 8):
            j = idx[i:i + 8]
            y = torch.from_numpy(test.hf[j].astype(np.float32)).cuda()
            x = fwd(y, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = enh(x).float().clamp(0, 1)
                rmap = torch.full_like(p, r / 6.0)
                cmap = torch.full_like(p, c / 0.25)
                q = det(torch.cat([x, p, rmap, cmap], 1)).float()
            m = y > 0
            f += float(((q < lam) & m).sum()); n += float(m.sum())
        fr.append(f / max(n, 1))
    return np.array(fr)


def main(contrast="T2", op="learned"):
    T.seed_all(0)
    test = D.PairedCohort("test", contrast)
    enh = enhancer(); enh.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_{contrast}.pt")); enh.eval()
    det = Mo.UNet(6, 1, base=32, residual_from=None).cuda()
    det.load_state_dict(torch.load(f"{OUT}/d3detect_{contrast}.pt")); det.eval()
    fwd = operator(contrast, op)
    crc = json.load(open(f"{OUT}/d3finding_{contrast}.json"))["crc"]
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)

    print(f"[{contrast}] fraction of BRAIN carrying a warning, by the lesion asked about\n")
    rows = {}
    for alpha in sorted(crc, key=float):
        d = crc[alpha]["detect"]
        lam = d["lam"]
        print(f"  alpha = {alpha}   lambda {lam:+.3f}   "
              f"({d['lesion_flagged']*100:.0f}% of mixed lesion sites flagged)")
        print(f"    {'lesion':>8s}" + "".join(f"{nm:>16s}" for _, nm in CONTRASTS))
        for r, rn in SIZES:
            cells = []
            for c, _ in CONTRASTS:
                fr = flagged_fraction(test, enh, det, fwd, gen, lam, r, c)
                rows[f"a{alpha}_r{r}_c{c}"] = fr.tolist()
                cells.append(f"{fr.mean()*100:10.1f}% +-{fr.std(ddof=1)*100:3.0f}")
            print(f"    {rn:>8s}" + "".join(cells))
        print()
    json.dump(rows, open(f"{OUT}/d3brainfrac_{contrast}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "T2")
