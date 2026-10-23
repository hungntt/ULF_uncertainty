"""The same real features, end to end, with no simulation anywhere.

Every retention figure in this programme feeds the enhancer a 64 mT image produced by the
learned forward operator. That is a closed loop: the operator decides how visible a lesion
is in the input, and the enhancer is then judged on recovering it.

The paired cohort has REAL 64 mT acquisitions. Feeding those instead removes the operator
from the measurement entirely. There is no counterfactual available - a real lesion cannot
be un-scanned - so retention is measured directly as the ratio of the feature's core-minus-
ring contrast in the synthesised 3T to the same contrast in the true 3T:

    retention = [p(core) - p(ring)] / [y(core) - y(ring)]

This is a different estimator from the inpainting one and it carries the local tissue
gradient in both terms, so the two numbers are not interchangeable. What it does do is
answer the end-to-end question without the operator in the loop.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, models as Mo, train as T
from exp.d3qa import enhancer
from exp.d3detect import blobs, operator
from exp.d3realret import feature_mask, fold_for

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
TAG = os.environ.get("REAL_TAG", "_loose")


@torch.no_grad()
def main():
    T.seed_all(0)
    rows = json.load(open(f"{OUT}/d3realret{TAG}.json"))
    nets = {}
    for f in range(4):
        n = enhancer(); n.load_state_dict(torch.load(f"{OUT}/d3qa_fold{f}_T2.pt")); n.eval()
        nets[f] = n
    dets = {}
    for nm, ck in (("simulated-input map", f"{OUT}/d3detect_T2.pt"),
                   ("same trainer, sim input", f"{OUT}/d3detect_T2_simctl.pt"),
                   ("real-input map", f"{OUT}/d3detect_T2_realin.pt")):
        if not os.path.exists(ck):
            print(f"  ({nm}: {os.path.basename(ck)} not found, skipped)")
            continue
        d_ = Mo.UNet(6, 1, base=32, residual_from=None).cuda()
        d_.load_state_dict(torch.load(ck)); d_.eval(); dets[nm] = d_
    det = dets["simulated-input map"]
    fwd = operator("T2", "learned")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    out = []
    print(f"  {'subject':11s} {'row':>5s} {'r_fit':>6s} {'true c':>8s} {'pred c':>8s} "
          f"{'retention':>10s}  {'input'}")
    for r in rows:
        coh = D.PairedCohort(r["split"], "T2")
        y = coh.hf[r["row"], 0].astype(np.float32)
        fm, _ = feature_mask(y, r["cy"], r["cx"])
        if fm is None:
            continue
        # the REAL 64 mT slab for this slice, not an operator's rendering of the 3T
        x = torch.from_numpy(coh.ulf[r["row"]:r["row"] + 1].astype(np.float32)).cuda()
        yt = torch.from_numpy(y).cuda()[None, None]
        f = fold_for(r["sid"]); enh = nets[f if f is not None else 0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = enh(x).float().clamp(0, 1)
        b = blobs(torch.tensor([r["cy"]], device="cuda"),
                  torch.tensor([r["cx"]], device="cuda"),
                  torch.tensor([r["r_fit"]], device="cuda")) * (yt > 0).float()
        core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
        tc = float(yt[core].mean() - yt[ring].mean())
        pc = float(p[core].mean() - p[ring].mean())
        if abs(tc) < 0.01:
            continue
        ret = float(np.clip(pc / tc, -0.5, 1.5))
        # query the map the way it would be DEPLOYED: on the real 64 mT slab and the
        # prediction made from it, not on the simulated pair it was scored with before
        qs = {}
        for nm, d_ in dets.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qq = d_(torch.cat([x, p, torch.full_like(p, r["r_fit"] / 6.0),
                                   torch.full_like(p, tc / 0.25)], 1)).float()
            qs[nm] = float(qq[0, 0, r["cy"], r["cx"]])
        qraw = qs["simulated-input map"]
        # CONTROL: same direct estimator, but through the SIMULATED pipeline. If the
        # correlation collapses here too, the estimator is the cause, not the real input.
        xs = fwd(yt, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ps = enh(xs).float().clamp(0, 1)
        pcs = float(ps[core].mean() - ps[ring].mean())
        rets = float(np.clip(pcs / tc, -0.5, 1.5))
        out.append(dict(sid=r["sid"], split=r["split"], row=r["row"], r_fit=r["r_fit"],
                        true_c=tc, pred_c=pc, retention=ret, map=r["map_fit"],
                        map_raw=qraw, ret_sim_direct=rets, qs=qs))
        print(f"  {r['sid']:11s} {r['row']:5d} {r['r_fit']:6.2f} {tc:8.3f} {pc:8.3f} "
              f"{ret:10.3f}  real 64 mT")
    if not out:
        print("  nothing measurable"); return
    R = np.array([o["retention"] for o in out])
    objs = {}
    for o in out:
        objs.setdefault((o["sid"], o["row"] // 3, 0), []).append(o["retention"])
    oR = np.array([np.mean(v) for v in objs.values()])
    print(f"\n  {len(R)} sites / {len(oR)} objects, REAL 64 mT input, no forward operator")
    print(f"    retention {R.mean():.3f} +-{R.std(ddof=1):.3f}   {100*np.mean(R < 0.25):.0f}% erased")
    te = [o["retention"] for o in out if o["split"] == "test"]
    if te:
        print(f"    locked-test subjects only ({len(te)} sites): {np.mean(te):.3f}")
    from scipy.stats import spearmanr
    Q = np.array([o["map"] for o in out])
    Qr = np.array([o["map_raw"] for o in out])
    from sklearn.metrics import roc_auc_score
    y = (R < 0.25).astype(int)
    print(f"    map queried on SIMULATED inputs vs this retention: spearman "
          f"{spearmanr(Q, R).statistic:+.3f}")
    print(f"    map queried on the REAL 64 mT, as deployed:        spearman "
          f"{spearmanr(Qr, R).statistic:+.3f}   AUROC {roc_auc_score(y, -Qr):.3f}")
    S = np.array([o["ret_sim_direct"] for o in out])
    ys = (S < 0.25).astype(int)
    print(f"\n  CONTROL - same DIRECT estimator through the SIMULATED pipeline:")
    print(f"    retention {S.mean():.3f}   {100*ys.mean():.0f}% erased   "
          f"spearman vs map {spearmanr(Q, S).statistic:+.3f}   "
          f"AUROC {roc_auc_score(ys, -Q):.3f}")
    print(f"    (the inpainting estimator on the same pipeline gave AUROC 0.78)")
    print(f"\n  == maps scored on the REAL-input pipeline (the deployment question) ==")
    print(f"  {'map':26s} {'spearman':>9s} {'AUROC':>7s}")
    for nm in out[0]["qs"]:
        q_ = np.array([o["qs"][nm] for o in out])
        au = roc_auc_score(y, -q_) if 0 < y.sum() < len(y) else float("nan")
        print(f"  {nm:26s} {spearmanr(q_, R).statistic:+9.3f} {au:7.3f}")
    json.dump(out, open(f"{OUT}/d3realraw{TAG}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
