"""Is one detectability map enough, or does every enhancer need its own?

The map was trained against one pipeline (base_unet_noresid). It answers a counterfactual
about THAT pipeline's behaviour, so in principle it should not describe a different one -
and the architecture study showed the pipelines differ a lot, pix2pix erasing 63% of 4 mm
lesions where the U-Nets erase 85-90%.

Two deployment questions follow:

  1. Does the map still rank sites correctly under an enhancer it was not trained on?
     (If yes, one map serves a lab's whole model zoo. If no, a map is part of a model.)
  2. Does the map's own predicted retention track each architecture's ACTUAL retention?
     Ranking sites within a pipeline and ranking pipelines are different abilities, and
     a map that cannot do the second must not be used to choose a model.

Identical lesions at identical sites across all architectures, one fixed map throughout.
"""
import argparse, json, sys
import numpy as np, torch
sys.path.insert(0, "/mnt/G/Projects/ULF-MICCAI")
from ulfkit import data as D, models as Mo, train as T
from exp.d3qa import enhancer
from exp.d3detect import operator, site_pool, blobs
from exp.d3arch import ARCHS, load

OUT = "/mnt/G/Projects/ULF-MICCAI/out"
HOME = "U-Net absolute"          # the pipeline the map was trained against


@torch.no_grad()
def run(contrast="T2", n_sites=5, radii=(2, 3, 4, 6), cs=(0.12, 0.25), seed=11, map_tag=""):
    T.seed_all(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    test = D.PairedCohort(contrast=contrast, split="test")
    fwd = operator(contrast, "learned")
    det = Mo.UNet(6, 1, base=32, residual_from=None).cuda()
    det.load_state_dict(torch.load(f"{OUT}/d3detect_{contrast}{map_tag}.pt")); det.eval()
    nets = []
    for nm, path, kind in ARCHS:
        try:
            nets.append((nm, load(kind, path)))
        except FileNotFoundError:
            print(f"  (missing {path}, skipped)")
    rows = []
    for sid in test.subjects:
        idx = test.subject_rows(sid)
        sel = idx[np.linspace(len(idx)//5, 4*len(idx)//5, 8).astype(int)]
        for i0 in range(0, len(sel), 4):
            j = sel[i0:i0+4]
            y0 = torch.from_numpy(test.hf[j].astype(np.float32)).cuda()
            ok = site_pool(y0)
            x0 = fwd(y0, gen)
            base = {}
            for nm, net in nets:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    base[nm] = net(x0).float().clamp(0, 1)
            jobs = []
            for s in range(len(j)):
                cand = torch.nonzero(ok[s])
                if len(cand) < 8: continue
                for _ in range(n_sites):
                    cy, cx = cand[rng.integers(len(cand))].tolist()
                    jobs.append((s, cy, cx, int(rng.choice(radii)), float(rng.choice(cs))))
            for b0 in range(0, len(jobs), 12):
                ch = jobs[b0:b0+12]
                si = torch.tensor([t[0] for t in ch], device="cuda")
                cy = torch.tensor([t[1] for t in ch], device="cuda")
                cx = torch.tensor([t[2] for t in ch], device="cuda")
                rr = torch.tensor([t[3] for t in ch], device="cuda")
                cc = torch.tensor([t[4] for t in ch], device="cuda")[:, None, None, None]
                b = blobs(cy, cx, rr) * (y0[si] > 0).float()
                yl = (y0[si] + cc * b).clamp(0, 1)
                dy = yl - y0[si]
                core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
                good = torch.tensor([bool(core[k].sum() >= 4 and ring[k].sum() >= 8)
                                     for k in range(len(ch))], device="cuda")
                tru = torch.stack([(dy[k][core[k]].mean() - dy[k][ring[k]].mean())
                                   if good[k] else torch.tensor(0., device="cuda")
                                   for k in range(len(ch))])
                xl = fwd(yl, gen)
                for nm, net in nets:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pl = net(xl).float().clamp(0, 1)
                        # the map is conditioned on the pipeline's own output, so it is
                        # re-queried per architecture; only its WEIGHTS are fixed
                        q = det(torch.cat([x0[si], base[nm][si],
                                           (rr.float() / 6.0)[:, None, None, None].expand_as(base[nm][si]),
                                           (cc / 0.25).expand_as(base[nm][si])], 1)).float()
                    dp = pl - base[nm][si]
                    rec = torch.stack([(dp[k][core[k]].mean() - dp[k][ring[k]].mean())
                                       if good[k] else torch.tensor(0., device="cuda")
                                       for k in range(len(ch))])
                    for k in range(len(ch)):
                        if not good[k] or tru[k].abs() < 0.25 * cc[k, 0, 0, 0].abs():
                            continue
                        rows.append(dict(arch=nm, sid=sid, r=ch[k][3], c=ch[k][4],
                                         retention=float((rec[k] / tru[k]).clamp(-0.5, 1.5)),
                                         q=float(q[k, 0, ch[k][1], ch[k][2]])))
        print(f"  {sid}: {len(rows)} measurements", flush=True)
    return rows


def report(rows, thr=0.25, held=()):
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    archs = [nm for nm, _, _ in ARCHS if any(x["arch"] == nm for x in rows)]
    print("\n== one fixed map, every enhancer ==")
    print(f"  map trained against: {HOME}\n")
    print(f"  {'enhancer':22s} {'actual ret':>11s} {'%erased':>8s} {'AUROC':>7s} "
          f"{'within-size':>12s} {'mean pred':>10s}")
    summ = {}
    for nm in archs:
        R = [x for x in rows if x["arch"] == nm]
        ret = np.array([x["retention"] for x in R]); q = np.array([x["q"] for x in R])
        y = (ret < thr).astype(int)
        au = roc_auc_score(y, -q) if 0 < y.sum() < len(y) else float("nan")
        ws = []
        for r in (2, 3, 4, 6):
            m = np.array([x["r"] == r for x in R])
            if m.sum() > 20 and 0 < y[m].sum() < m.sum():
                ws.append(roc_auc_score(y[m], -q[m]))
        summ[nm] = dict(ret=float(ret.mean()), er=float(y.mean()), auroc=float(au),
                        within=float(np.mean(ws)) if ws else float("nan"),
                        pred=float(q.mean()))
        star = ("  <- HELD OUT" if nm in held else ("  <- home" if nm == HOME else ""))
        print(f"  {nm:22s} {ret.mean():11.3f} {y.mean()*100:7.1f}% {au:7.3f} "
              f"{(np.mean(ws) if ws else float('nan')):12.3f} {q.mean():10.3f}{star}")
    A = np.array([summ[n]["ret"] for n in archs])
    P = np.array([summ[n]["pred"] for n in archs])
    print(f"\n  across architectures: spearman(mean predicted, mean actual) "
          f"{spearmanr(P, A).statistic:+.3f}")
    print(f"    predicted spread {P.max()-P.min():.4f}   actual spread {A.max()-A.min():.4f}   "
          f"ratio {(P.max()-P.min())/max(A.max()-A.min(),1e-9):.2f}")
    if held:
        hn = [n for n in archs if n in held]
        if len(hn) >= 3:
            Ah = np.array([summ[n]["ret"] for n in hn]); Ph = np.array([summ[n]["pred"] for n in hn])
            print(f"  HELD-OUT architectures only (n={len(hn)}): "
                  f"spearman {spearmanr(Ph, Ah).statistic:+.3f}   "
                  f"predicted spread {Ph.max()-Ph.min():.4f}  actual {Ah.max()-Ah.min():.4f}")
    print("  (ranking SITES within a pipeline and ranking PIPELINES are different abilities)")
    json.dump(summ, open(f"{OUT}/d3xarch_T2.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_tag", default="")
    ap.add_argument("--held", default="")
    a = ap.parse_args()
    print(f"detectability map: d3detect_T2{a.map_tag}.pt\n")
    rows = run(map_tag=a.map_tag)
    json.dump(rows, open(f"{OUT}/d3xarch_rows_T2{a.map_tag}.json", "w"), indent=1, default=float)
    report(rows, held=tuple(x for x in a.held.split("|") if x))
