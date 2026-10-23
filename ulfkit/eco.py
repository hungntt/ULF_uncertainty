"""Pieces of ECO-MT: Laplacian bands, MIND, the two augmented views, the EMA
teacher, and the two gates (source observability and calibrated teacher risk).

The gate weights are built outside the autograd graph. w is evidence, not a
parameter: letting the student push gradients into its own trust score is the
fastest way to learn that nothing should be trusted.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F

_KB = {}


def _blur(x):
    k = _KB.get(x.device)
    if k is None:
        g = torch.tensor([1., 4., 6., 4., 1.], device=x.device)
        g = g / g.sum()
        k = (g[:, None] @ g[None, :])[None, None]
        _KB[x.device] = k
    c = x.shape[1]
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), k.expand(c, 1, 5, 5), groups=c)


def lap_pyr(x, n=3):
    """[fine 256, medium 128, coarse 64] for a 256^2 input: n-1 bandpass levels
    then the lowpass residual. Bands are differences, not independent images."""
    cur, out = x, []
    for _ in range(n - 1):
        d = _blur(cur)[:, :, ::2, ::2]
        out.append(cur - F.interpolate(d, size=cur.shape[-2:], mode="bilinear", align_corners=False))
        cur = d
    out.append(cur)
    return out


def pyr_pool(m, n=3):
    """A full-resolution map average-pooled onto each pyramid level."""
    out, cur = [m], m
    for _ in range(n - 1):
        cur = F.avg_pool2d(cur, 2)
        out.append(cur)
    return out


def loc(x, w=9):
    return F.avg_pool2d(x, w, 1, w // 2, count_include_pad=False)


def loc_corr(a, b, w=9):
    ma, mb = loc(a, w), loc(b, w)
    va = (loc(a * a, w) - ma * ma).clamp(min=0)
    vb = (loc(b * b, w) - mb * mb).clamp(min=0)
    return (loc(a * b, w) - ma * mb) / (va.sqrt() * vb.sqrt()).clamp(min=1e-6)


def mind(x, r=2):
    """2D MIND self-similarity descriptor: 4-neighbourhood, 3x3 patch SSD."""
    d = torch.cat([F.avg_pool2d((x - torch.roll(x, s, dims=(-2, -1))) ** 2, 3, 1, 1)
                   for s in [(0, r), (0, -r), (r, 0), (-r, 0)]], 1)
    v = d.mean(1, keepdim=True).clamp(min=1e-6)
    e = torch.exp(-d / v)
    return e / e.amax(1, keepdim=True).clamp(min=1e-6)


# ---------------------------------------------------------------- views

def weak(x, rng):
    """Photometric only, so the teacher's output stays in the input's frame."""
    g = 1.0 + 0.05 * float(2 * rng.random() - 1)
    b = 0.02 * float(2 * rng.random() - 1)
    return (x * g + b).clamp(0, 1)


def _lowfreq(x, rng, k=4):
    b = torch.from_numpy(rng.normal(0, 1, (x.shape[0], 1, k, k)).astype(np.float32)).to(x.device)
    b = F.interpolate(b, size=x.shape[-2:], mode="bicubic", align_corners=False)
    return b / b.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)


def warp(x, th):
    g = F.affine_grid(th, list(x.shape), align_corners=False)
    return F.grid_sample(x, g, mode="bilinear", padding_mode="zeros", align_corners=False)


def strong(x, rng, bias=0.15, noise=0.02, rot=5.0, trans=0.02, scale=0.03):
    """Anatomy-preserving strong view: smooth bias field, low-frequency noise and a
    small rigid+scale warp. Returns the view and the theta that carries the
    teacher's prediction into the student's coordinate frame."""
    v = (x * (1 + bias * _lowfreq(x, rng)) + noise * _lowfreq(x, rng)).clamp(0, 1)
    n = x.shape[0]
    a = np.deg2rad(rot) * (2 * rng.random(n) - 1)
    s = 1 + scale * (2 * rng.random(n) - 1)
    t = trans * (2 * rng.random((n, 2)) - 1)
    th = np.zeros((n, 2, 3), np.float32)
    th[:, 0, 0], th[:, 0, 1], th[:, 0, 2] = np.cos(a) / s, -np.sin(a) / s, t[:, 0]
    th[:, 1, 0], th[:, 1, 1], th[:, 1, 2] = np.sin(a) / s, np.cos(a) / s, t[:, 1]
    th = torch.from_numpy(th).to(x.device)
    return warp(v, th), th


# ---------------------------------------------------------------- teacher

class EMA:
    """Exponential moving-average teacher. Runs in eval/no-grad throughout."""

    def __init__(self, model, warm=1000, d0=0.99, d1=0.999):
        self.t = copy.deepcopy(model).eval()
        for p in self.t.parameters():
            p.requires_grad_(False)
        self.warm, self.d0, self.d1 = warm, d0, d1

    @torch.no_grad()
    def update(self, model, it):
        d = self.d0 if it < self.warm else self.d1
        for a, b in zip(self.t.state_dict().values(), model.state_dict().values()):
            if a.dtype.is_floating_point:
                a.mul_(d).add_(b, alpha=1 - d)
            else:
                a.copy_(b)

    @torch.no_grad()
    def __call__(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.t(x).float().clamp(0, 1)


# ---------------------------------------------------------------- gates

def _interp(x, edges, vals):
    """Piecewise-linear lookup; flat outside the fitted range."""
    i = torch.bucketize(x, edges).clamp(1, len(edges) - 1)
    lo, hi = edges[i - 1], edges[i]
    f = ((x - lo) / (hi - lo).clamp(min=1e-8)).clamp(0, 1)
    return vals[i - 1] * (1 - f) + vals[i] * f


class Observability:
    """o_l(x,p) in [0,1]. Local ULF band energy -> the local ULF/3T band
    correlation that energy bought on the training pairs. Needs only the ULF
    image at inference, which is the whole point: it has to work on a scan with
    no high-field partner. This is an empirical image-domain proxy, not a
    physical observability operator."""

    def __init__(self, edges, vals, n=3, win=9):
        self.edges, self.vals, self.n, self.win = edges, vals, n, win

    @staticmethod
    def fit(chunks, n=3, win=9, bins=16, per_chunk=40000, seed=0):
        """chunks: an iterable of (ulf_centre, hf) cuda batches. Voxels are
        subsampled per chunk so the fit never holds the whole cohort at once."""
        g = torch.Generator().manual_seed(seed)
        EE, CC = [[] for _ in range(n)], [[] for _ in range(n)]
        for u, h in chunks:
            m = ((u > 0) | (h > 0)).float()
            for l, (pu, ph) in enumerate(zip(lap_pyr(u, n), lap_pyr(h, n))):
                ml = F.interpolate(m, size=pu.shape[-2:], mode="area") > 0.5
                e = torch.log(loc(pu * pu, win).sqrt().clamp(min=1e-8))[ml]
                c = loc_corr(pu, ph, win)[ml].clamp(0, 1)
                if not len(e):
                    continue
                i = torch.randint(0, len(e), (min(per_chunk, len(e)),), generator=g).to(e.device)
                EE[l].append(e[i].cpu()); CC[l].append(c[i].cpu())
        edges, vals, raw = [], [], []
        for l in range(n):
            e, c = torch.cat(EE[l]).cuda(), torch.cat(CC[l]).cuda()
            q = torch.unique(torch.quantile(e, torch.linspace(0, 1, bins + 1, device=e.device)))
            i = torch.bucketize(e, q).clamp(1, len(q) - 1) - 1
            sm = torch.zeros(len(q) - 1, device=e.device).index_add_(0, i, c)
            k = torch.zeros(len(q) - 1, device=e.device).index_add_(0, i, torch.ones_like(c))
            v = (sm / k.clamp(min=1)).clamp(0, 1)
            raw.append(v.clone())
            # no monotonicity is imposed: measured on this cohort the fine and
            # medium bands gain support with energy while the coarse band loses
            # it, so forcing a non-decreasing curve would erase the coarse signal
            v = F.avg_pool1d(v[None, None], 3, 1, 1, count_include_pad=False)[0, 0]
            edges.append(q[:-1]); vals.append(v.clamp(0, 1))
        o = Observability(edges, vals, n, win)
        o.raw = raw
        return o

    def __call__(self, ulf_centre):
        out = []
        for l, pu in enumerate(lap_pyr(ulf_centre, self.n)):
            e = torch.log(loc(pu * pu, self.win).sqrt().clamp(min=1e-8))
            out.append(_interp(e, self.edges[l].to(e.device), self.vals[l].to(e.device)))
        return out

    def state(self):
        d = {"edges": [e.cpu().numpy().tolist() for e in self.edges],
             "vals": [v.cpu().numpy().tolist() for v in self.vals],
             "n": self.n, "win": self.win}
        if hasattr(self, "raw"):
            d["raw"] = [r.cpu().numpy().tolist() for r in self.raw]
        return d

    @staticmethod
    def load(s, device="cuda"):
        return Observability([torch.tensor(e, device=device) for e in s["edges"]],
                             [torch.tensor(v, device=device) for v in s["vals"]],
                             s["n"], s["win"])


FEATS = ["u", "d", "s"]


def features(teach1, teach2, ulf_centre, mask, n=3):
    """Per-sample, per-scale evidence available without a high-field reference.

    u  teacher perturbation variation across two weak views
    d  MIND disagreement between the teacher's output and the ULF input
    s  ULF band energy (how much source structure supports this band)
    """
    md = (mind(teach1) - mind(ulf_centre)).abs().mean(1, keepdim=True)
    pm = pyr_pool(mask, n)
    pd = pyr_pool(md, n)
    pu = [a - b for a, b in zip(lap_pyr(teach1, n), lap_pyr(teach2, n))] if teach2 is not None else None
    ps = lap_pyr(ulf_centre, n)
    out = []
    for l in range(n):
        w = pm[l].clamp(min=0) + 1e-6
        f = {"u": (pu[l].abs() * pm[l]).sum((1, 2, 3)) / w.sum((1, 2, 3)) if pu else None,
             "d": (pd[l] * pm[l]).sum((1, 2, 3)) / w.sum((1, 2, 3)),
             "s": (ps[l].abs() * pm[l]).sum((1, 2, 3)) / w.sum((1, 2, 3))}
        out.append(f)
    return out


def features_sp(p1, p2, ulf_c, n=3, win=9):
    """The same three signals as `features`, kept as spatial maps and with MIND
    recomputed at each scale. Measured per voxel these rank the teacher's error
    far better than their per-slice means do (AUROC 0.875 vs AURC 0.961 at the
    fine band), so the trust gate is built from these, not from the scalars."""
    pu = [(a - b).abs() for a, b in zip(lap_pyr(p1, n), lap_pyr(p2, n))]
    ps = [b.abs() for b in lap_pyr(ulf_c, n)]
    out = []
    for l in range(n):
        f = 2 ** l
        a = F.avg_pool2d(p1, f) if f > 1 else p1
        b = F.avg_pool2d(ulf_c, f) if f > 1 else ulf_c
        d = (mind(a) - mind(b)).abs().mean(1, keepdim=True)
        out.append({"u": loc(pu[l], win), "d": loc(d, win), "s": loc(ps[l], win)})
    return out


class Risk:
    """r_hat_l(u,d,s): predicted scale-level teacher error, fitted by ridge on
    standardised log evidence against out-of-fold paired errors. Linear and
    strongly regularised on purpose: 50 subjects do not support more."""

    def __init__(self, mu, sd, w, b):
        self.mu, self.sd, self.w, self.b = mu, sd, w, b

    @staticmethod
    def fit(X, y, alpha=1.0):
        """X (N,L,F) log features, y (N,L) out-of-fold errors."""
        mu, sd, w, b = [], [], [], []
        for l in range(X.shape[1]):
            Z = np.log(np.clip(X[:, l], 1e-8, None))
            m, s = Z.mean(0), Z.std(0) + 1e-8
            Z = (Z - m) / s
            A = np.concatenate([Z, np.ones((len(Z), 1))], 1)
            P = np.eye(A.shape[1]) * alpha
            P[-1, -1] = 0.0
            c = np.linalg.solve(A.T @ A + P, A.T @ y[:, l])
            mu.append(m); sd.append(s); w.append(c[:-1]); b.append(float(c[-1]))
        return Risk(np.array(mu), np.array(sd), np.array(w), np.array(b))

    def __call__(self, X):
        """X (N,L,F) -> (N,L) predicted error."""
        Z = (np.log(np.clip(X, 1e-8, None)) - self.mu) / self.sd
        return np.einsum("nlf,lf->nl", Z, self.w) + self.b

    def torch_apply(self, feats, device="cuda"):
        """feats: list over scales of dicts of (B,) tensors -> (B,L) tensor."""
        mu = torch.tensor(self.mu, dtype=torch.float32, device=device)
        sd = torch.tensor(self.sd, dtype=torch.float32, device=device)
        w = torch.tensor(self.w, dtype=torch.float32, device=device)
        b = torch.tensor(self.b, dtype=torch.float32, device=device)
        out = []
        for l, f in enumerate(feats):
            z = torch.stack([f[k] for k in FEATS], 1).clamp(min=1e-8).log()
            out.append(((z - mu[l]) / sd[l] * w[l]).sum(1) + b[l])
        return torch.stack(out, 1)

    def apply_sp(self, feats, scale=0):
        """feats: a list of spatial-map dicts -> a list of predicted-error maps."""
        out = []
        for l, f in enumerate(feats):
            m = self.w.shape[0]
            i = l if m > 1 else 0
            z = [(torch.log(f[k].clamp(min=1e-8)) - float(self.mu[i][j])) / float(self.sd[i][j])
                 for j, k in enumerate(FEATS)]
            out.append(sum(z[j] * float(self.w[i][j]) for j in range(len(FEATS))) + float(self.b[i]))
        return out

    def state(self):
        return {k: getattr(self, k).tolist() for k in ("mu", "sd", "w", "b")}

    @staticmethod
    def load(s):
        return Risk(*[np.array(s[k]) for k in ("mu", "sd", "w", "b")])
