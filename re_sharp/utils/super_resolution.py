"""
Inference pipeline: runs the pretrained RRDBNet on a PIL image.

Large images are processed in small overlapping tiles so memory stays low on a
student laptop (CPU). Each tile is given a little extra context (`tile_pad`) so
that tile borders do not show visible seams; the context is cropped away again
before the tile is pasted into the output.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def get_device() -> torch.device:
    """Use the GPU if one is available, otherwise the CPU (default for most laptops)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # use all CPU cores for the convolutions
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return torch.device("cpu")


def bicubic_upscale(img: Image.Image, scale: int = None, size: tuple = None) -> Image.Image:
    """Traditional interpolation baseline (NOT deep learning)."""
    if size is None:
        size = (img.width * scale, img.height * scale)
    return img.resize(size, Image.BICUBIC)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).clamp(0, 1).mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.ascontiguousarray(arr))


def enhance_image(
    img: Image.Image,
    model: torch.nn.Module,
    scale: int,
    device: torch.device,
    tile: int = 128,
    tile_pad: int = 10,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Image.Image:
    """Super-resolve an RGB PIL image by `scale` (2 or 4) with the given model.

    tile and tile_pad must be even numbers (the 2x model uses pixel-unshuffle).
    """
    assert tile % 2 == 0 and tile_pad % 2 == 0, "tile and tile_pad must be even"
    x = _pil_to_tensor(img)
    _, _, h, w = x.shape

    # The 2x model needs even height/width -> pad by 1 px (replicated) and crop later.
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    H, W = x.shape[2], x.shape[3]

    out = torch.zeros(1, 3, H * scale, W * scale, dtype=torch.float32)
    ys_list = list(range(0, H, tile))
    xs_list = list(range(0, W, tile))
    total = len(ys_list) * len(xs_list)
    done = 0

    with torch.inference_mode():
        for y0 in ys_list:
            for x0 in xs_list:
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)          # tile core
                py0, px0 = max(y0 - tile_pad, 0), max(x0 - tile_pad, 0)  # core + context
                py1, px1 = min(y1 + tile_pad, H), min(x1 + tile_pad, W)

                region = x[:, :, py0:py1, px0:px1].to(device)
                sr = model(region).cpu()

                # crop the context away again and paste the core into the output
                cy0, cx0 = (y0 - py0) * scale, (x0 - px0) * scale
                cy1, cx1 = cy0 + (y1 - y0) * scale, cx0 + (x1 - x0) * scale
                out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = sr[:, :, cy0:cy1, cx0:cx1]

                done += 1
                if progress_cb:
                    progress_cb(done / total)

    out = out[:, :, : h * scale, : w * scale]
    return _tensor_to_pil(out)
