"""Image-quality metrics (pure NumPy - no scikit-image needed).

IMPORTANT: PSNR and SSIM are FULL-REFERENCE metrics. They compare an image with a trusted
ground-truth image of the same size. For an arbitrary uploaded photo no ground truth exists,
so the app must NOT report these numbers in the normal workflow. They are only valid:
  * for the bundled demo images (we generated the sharp reference ourselves), and
  * in the controlled experiment below (sharp image -> shrink -> upscale -> compare with original).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image


@dataclass
class Scores:
    psnr: float   # dB, higher = closer to the reference
    ssim: float   # 0..1 (1 = identical structure); NaN if the image is too small


def _to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float64)


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    mse = float(np.mean((ref - test) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(255.0 ** 2 / mse)


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    k = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _filter(channel: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Separable 'valid' Gaussian filtering of a 2-D array."""
    tmp = sliding_window_view(channel, len(k), axis=0) @ k
    return sliding_window_view(tmp, len(k), axis=1) @ k


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    """Mean SSIM (Wang et al., 2004): Gaussian 11x11 window, sigma 1.5, averaged over RGB."""
    if min(ref.shape[0], ref.shape[1]) < 11:
        return float("nan")
    k = _gaussian_kernel()
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    scores = []
    for c in range(ref.shape[2]):
        x, y = ref[:, :, c], test[:, :, c]
        mu_x, mu_y = _filter(x, k), _filter(y, k)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sig_x = _filter(x * x, k) - mu_x2
        sig_y = _filter(y * y, k) - mu_y2
        sig_xy = _filter(x * y, k) - mu_xy
        smap = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2))
        scores.append(float(smap.mean()))
    return float(np.mean(scores))


def compute_scores(reference: Image.Image, test: Image.Image) -> Scores:
    """PSNR + SSIM of `test` against a TRUE reference of identical size."""
    if reference.size != test.size:
        raise ValueError(f"Reference {reference.size} and test image {test.size} must have the same size.")
    ref, tst = _to_array(reference), _to_array(test)
    return Scores(psnr=psnr(ref, tst), ssim=ssim(ref, tst))


def fmt_psnr(value: float) -> str:
    return "∞ (identical)" if math.isinf(value) else f"{value:.2f} dB"


def run_controlled_experiment(
    image: Image.Image,
    scale: int,
    enhance_fn: Callable[[Image.Image], Image.Image],
    max_side: int = 384,
) -> Dict:
    """Create a meaningful reference: treat `image` as ground truth, shrink it, rebuild it two ways.

    hr (ground truth) --bicubic shrink--> lr --+--> bicubic upscale --> compared with hr
                                               +--> deep-learning model --> compared with hr
    """
    hr = image.convert("RGB")
    longest = max(hr.size)
    if longest > max_side:                              # keep the experiment fast on CPU
        r = max_side / longest
        hr = hr.resize((max(1, round(hr.width * r)), max(1, round(hr.height * r))), Image.LANCZOS)
    w, h = (hr.width // scale) * scale, (hr.height // scale) * scale   # sizes divisible by the scale
    if min(w, h) < 16 * scale:
        raise ValueError(f"Image is too small for a {scale}x experiment (needs at least {16 * scale} px per side).")
    hr = hr.crop((0, 0, w, h))

    lr = hr.resize((w // scale, h // scale), Image.BICUBIC)
    bicubic = lr.resize((w, h), Image.BICUBIC)
    ai = enhance_fn(lr)
    if ai.size != hr.size:
        raise RuntimeError(f"Model output size {ai.size} does not match the reference {hr.size}.")
    return {
        "hr": hr, "lr": lr, "bicubic": bicubic, "ai": ai,
        "bicubic_scores": compute_scores(hr, bicubic),
        "ai_scores": compute_scores(hr, ai),
    }
