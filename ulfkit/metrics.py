"""Evaluation metrics for ULF -> 3T enhancement.

Everything is reported both whole-image (what the literature quotes) and inside
the brain mask. The two disagree a lot here: the volumes are skull-stripped, so
~75% of every slice is exact zero and whole-image SSIM is dominated by matching
background.
"""
import numpy as np
import torch
import torch.nn.functional as F

_WIN = None


def _gauss(ws=11, sigma=1.5, device="cuda"):
    global _WIN
    if _WIN is None or _WIN.device != torch.device(device):
        g = torch.arange(ws, dtype=torch.float32, device=device) - ws // 2
        g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
        g = (g / g.sum())
        _WIN = (g[:, None] @ g[None, :])[None, None]
    return _WIN


def ssim_map(a, b, device="cuda"):
    """a,b: (N,1,H,W) in [0,1]."""
    w = _gauss(device=a.device.type if a.is_cuda else "cpu")
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.conv2d(a, w, padding=5)
    mu2 = F.conv2d(b, w, padding=5)
    s11 = F.conv2d(a * a, w, padding=5) - mu1 ** 2
    s22 = F.conv2d(b * b, w, padding=5) - mu2 ** 2
    s12 = F.conv2d(a * b, w, padding=5) - mu1 * mu2
    return ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))


def brain_mask(ulf, hf):
    """Union of the two skull-stripped supports, as float (N,1,H,W)."""
    return ((ulf > 0) | (hf > 0)).float()


def volume_metrics(pred, hf, ulf):
    """All metrics for one subject's stack. pred/hf/ulf: (N,1,H,W) float32 cuda."""
    m = brain_mask(ulf, hf)
    n = m.sum().clamp(min=1)
    out = {}

    mse_m = (((pred - hf) ** 2) * m).sum() / n
    out["psnr"] = (10 * torch.log10(1.0 / mse_m.clamp(min=1e-12))).item()
    out["psnr_full"] = (10 * torch.log10(1.0 / ((pred - hf) ** 2).mean().clamp(min=1e-12))).item()

    s = ssim_map(pred, hf)
    out["ssim"] = ((s * m).sum() / n).item()
    out["ssim_full"] = s.mean().item()

    out["nrmse"] = (torch.sqrt((((pred - hf) ** 2) * m).sum()) /
                    torch.sqrt(((hf ** 2) * m).sum()).clamp(min=1e-8)).item()
    out["mae"] = ((pred - hf).abs() * m).sum().div(n).item()
    return out


_DWT = None


def _dwt():
    global _DWT
    if _DWT is None:
        from pytorch_wavelets import DWTForward
        _DWT = DWTForward(J=1, wave="sym4", mode="zero").cuda()
    return _DWT


def band_metrics(pred, hf, ulf):
    """Per-subband residual ratio r_b and directional cosine.

    r_b   = ||W_b(pred) - W_b(ulf)|| / ||W_b(hf) - W_b(ulf)||   (1.0 = right size)
    cos_b = cos(W_b(pred) - W_b(ulf), W_b(hf) - W_b(ulf))       (1.0 = right way)

    These separate 'moved far enough' from 'moved the right way'. An identity map
    scores r=0 and an undefined cosine; a model that sharpens generically scores
    r~1 with a low cosine.
    """
    dwt = _dwt()
    yl_p, yh_p = dwt(pred)
    yl_h, yh_h = dwt(hf)
    yl_u, yh_u = dwt(ulf)
    bands = {"LL": (yl_p, yl_h, yl_u)}
    for i, nm in enumerate(["LH", "HL", "HH"]):
        bands[nm] = (yh_p[0][:, :, i], yh_h[0][:, :, i], yh_u[0][:, :, i])
    out = {}
    for nm, (p, h, u) in bands.items():
        dp = (p - u).flatten()
        dg = (h - u).flatten()
        out[f"r_{nm}"] = (dp.norm() / dg.norm().clamp(min=1e-8)).item()
        out[f"cos_{nm}"] = (F.cosine_similarity(dp[None], dg[None]).item())
    return out


def radial_spectrum(x, nbins=128):
    """Mean radial power spectrum of (N,1,H,W)."""
    X = torch.fft.fftshift(torch.fft.fft2(x[:, 0]), dim=(-2, -1)).abs() ** 2
    H, W = x.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                            torch.arange(W, device=x.device), indexing="ij")
    r = torch.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2).long().clamp(max=nbins - 1)
    p = X.mean(0).flatten()
    cnt = torch.bincount(r.flatten(), minlength=nbins).clamp(min=1)
    return torch.bincount(r.flatten(), weights=p, minlength=nbins) / cnt


def hf_power_error(pred, hf, r_lo=32):
    """log-ratio of high-frequency power, pred vs 3T. 0 = matched sharpness."""
    sp, sh = radial_spectrum(pred), radial_spectrum(hf)
    return (torch.log10(sp[r_lo:].sum() / sh[r_lo:].sum().clamp(min=1e-12))).item()


def aggregate(rows):
    """rows: list of per-subject dicts -> mean/std dict."""
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1)))
            for k in keys}


def paired_test(a, b):
    """Two-sided Wilcoxon signed-rank over per-subject values."""
    from scipy.stats import wilcoxon
    try:
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        return float("nan")
