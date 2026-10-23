"""Re-score saved Direction-2 checkpoints with the availability they were trained on."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, metrics as M, models as Mo, train as T
from exp.d2 import CENTRES, mc_inputs, _SubjView

OUT = "/mnt/G/Projects/ULF-MICCAI/out"


def main(tags):
    mc_te = D.MultiContrastCohort("test")
    view = _SubjView(mc_te)
    sids = sorted(set(mc_te.sid))
    for tag in tags:
        f = f"{OUT}/d2_{tag}.json"
        if not (os.path.exists(f) and os.path.exists(f"{OUT}/d2_{tag}.pt")):
            print(f"  {tag}: missing, skip"); continue
        d = json.load(open(f)); cfg = d["cfg"]
        use = tuple(cfg["use"].split(","))
        out_sel = {"both": [0, 1], "T1": [0], "T2": [1]}[cfg["out"]]
        net = Mo.UNet(8, len(out_sel), base=cfg["base"],
                      residual_from=[CENTRES[i] for i in out_sel]).cuda()
        net.load_state_dict(torch.load(f"{OUT}/d2_{tag}.pt"))

        def sub(fn):
            def g(pos):
                x, u, h = fn(pos)
                return x, u[:, out_sel], h[:, out_sel]
            return g

        regimes = [("T1", "T2"), ("T1",), ("T2",)] if cfg["drop"] else [use]
        d["results"] = {}
        for reg in regimes:
            rows = T.evaluate(net, view, sids,
                              sub(mc_inputs(mc_te, (int("T1" in reg), int("T2" in reg)))),
                              out_ch=len(out_sel))
            d["results"]["+".join(reg)] = {"agg": M.aggregate(rows), "rows": rows}
            g = d["results"]["+".join(reg)]["agg"]
            for c, oi in enumerate(out_sel):
                pre = "" if len(out_sel) == 1 else f"c{c}_"
                print(f"  {tag:12s} in={'+'.join(reg):6s} -> {['T1','T2'][oi]}: "
                      f"PSNR {g[pre+'psnr'][0]:6.3f}  SSIM {g[pre+'ssim'][0]:.4f}  "
                      f"cosLL {g[pre+'cos_LL'][0]:.3f}  rHH {g[pre+'r_HH'][0]:.3f}")
        d["reevaluated"] = True
        json.dump(d, open(f, "w"), indent=1, default=float)
        del net; torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or ["T1only", "T2only", "joint", "T2toT1", "T1toT2", "joint_drop",
                          "bothtoT1", "bothtoT2"])
