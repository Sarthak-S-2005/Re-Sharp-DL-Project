"""
Model handling: where the pretrained weights come from, automatic download,
and loading them into the RRDBNet architecture.

Weights: official Real-ESRGAN releases by Xintao Wang et al. (BSD-3-Clause)
    https://github.com/xinntao/Real-ESRGAN
"""

from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from models.rrdbnet import RRDBNet

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


class ModelUnavailableError(RuntimeError):
    """Raised when weights are missing / cannot be downloaded / are corrupted."""


@dataclass(frozen=True)
class ModelSpec:
    scale: int
    name: str
    filename: str
    url: str
    min_bytes: int  # sanity check: a truncated download is smaller than this


MODEL_SPECS = {
    2: ModelSpec(
        scale=2,
        name="Real-ESRGAN x2plus",
        filename="RealESRGAN_x2plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        min_bytes=60_000_000,
    ),
    4: ModelSpec(
        scale=4,
        name="Real-ESRGAN x4plus",
        filename="RealESRGAN_x4plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        min_bytes=60_000_000,
    ),
}


def weights_path(scale: int) -> Path:
    return WEIGHTS_DIR / MODEL_SPECS[scale].filename


def weights_ready(scale: int) -> bool:
    p = weights_path(scale)
    return p.exists() and p.stat().st_size >= MODEL_SPECS[scale].min_bytes


def _manual_help(spec: ModelSpec) -> str:
    return (
        f"Download this file in your browser:\n  {spec.url}\n"
        f"and save it as:\n  {weights_path(spec.scale)}"
    )


def ensure_weights(scale: int, progress_cb: Optional[Callable[[float], None]] = None) -> Path:
    """Return the path of the weights file, downloading it on first use."""
    spec = MODEL_SPECS[scale]
    target = weights_path(scale)
    if weights_ready(scale):
        return target

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    try:
        req = urllib.request.Request(spec.url, headers={"User-Agent": "re-sharp/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress_cb and total:
                    progress_cb(min(done / total, 1.0))
        if tmp.stat().st_size < spec.min_bytes:
            raise ModelUnavailableError("Downloaded file is incomplete.")
        shutil.move(str(tmp), str(target))
    except ModelUnavailableError:
        tmp.unlink(missing_ok=True)
        raise ModelUnavailableError("Download was incomplete.\n" + _manual_help(spec))
    except Exception as exc:  # network errors, DNS, firewall, ...
        tmp.unlink(missing_ok=True)
        raise ModelUnavailableError(
            f"Could not download the {spec.name} weights ({exc}).\n" + _manual_help(spec)
        ) from exc
    return target


def load_model(scale: int, device: torch.device,
               progress_cb: Optional[Callable[[float], None]] = None) -> RRDBNet:
    """Build RRDBNet and load the pretrained weights. Returns the model in eval mode."""
    path = ensure_weights(scale, progress_cb)
    try:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # very old PyTorch without weights_only
            ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("params_ema") or ckpt.get("params") or ckpt
        model = RRDBNet(scale=scale, num_feat=64, num_block=23, num_grow_ch=32)
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ModelUnavailableError(
            f"The weights file looks corrupted or incompatible ({exc}).\n"
            f"Delete it and run again so it is re-downloaded:\n  {path}"
        ) from exc
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model
