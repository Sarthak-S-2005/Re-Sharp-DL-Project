"""
Demo images generated locally (no licensing issues, no downloads).

For every demo we create
  * <name>_reference_hr.png  - the sharp 720x480 "ground truth"
  * <name>_lowres.jpg        - blurred + downscaled (360x240) + JPEG-compressed input

The app NEVER depends on these files: it regenerates them if they are missing,
and normal uploads work without them.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

DEMO_DIR = Path(__file__).resolve().parent.parent / "assets" / "demo_images"
W, H = 720, 480

DEMOS = {
    "City skyline": "city_skyline",
    "Text poster": "text_poster",
    "Hills & trees": "hills_trees",
}


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _vertical_gradient(top, bottom) -> Image.Image:
    t = np.linspace(0, 1, H)[:, None, None]
    arr = np.array(top)[None, None, :] * (1 - t) + np.array(bottom)[None, None, :] * t
    arr = np.repeat(arr, W, axis=1)
    return Image.fromarray(arr.astype(np.uint8))


def _skyline() -> Image.Image:
    img = _vertical_gradient((18, 22, 64), (255, 150, 90))
    d = ImageDraw.Draw(img)
    rng = np.random.RandomState(1)
    for _ in range(90):  # stars
        x, y = rng.randint(0, W), rng.randint(0, int(H * 0.4))
        d.point((x, y), fill=(255, 255, 255))
    d.ellipse([560, 60, 640, 140], fill=(255, 236, 170))  # moon / sun
    x = 0
    while x < W:
        bw, bh = rng.randint(40, 90), rng.randint(120, 330)
        shade = rng.randint(14, 40)
        d.rectangle([x, H - bh, x + bw, H], fill=(shade, shade, shade + 16))
        for wy in range(H - bh + 10, H - 10, 16):          # fine window grid = high-frequency detail
            for wx in range(x + 6, x + bw - 10, 12):
                if rng.rand() > 0.4:
                    d.rectangle([wx, wy, wx + 5, wy + 8], fill=(255, 220, 120))
        x += bw + rng.randint(0, 6)
    return img


def _text_poster() -> Image.Image:
    img = _vertical_gradient((240, 244, 255), (200, 210, 255))
    d = ImageDraw.Draw(img)
    for i in range(0, W, 12):                                # thin vertical lines
        d.line([(i, 0), (i, 60)], fill=(120, 120, 220), width=1)
    d.text((40, 80), "RE-SHARP", fill=(40, 40, 120), font=_font(96))
    d.text((44, 190), "AI-Powered Image Super-Resolution", fill=(70, 70, 150), font=_font(30))
    body = ("Deep learning estimates a higher-resolution image\n"
            "from patterns it learned during training.\n"
            "Fine text and thin edges are hard for interpolation.")
    d.multiline_text((44, 250), body, fill=(30, 30, 60), font=_font(20), spacing=8)
    for k in range(6):                                       # small concentric circles
        d.ellipse([560 + k * 8, 300 + k * 8, 680 - k * 8, 420 - k * 8], outline=(79, 70, 229), width=2)
    d.text((44, 420), "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789", fill=(90, 90, 90), font=_font(16))
    return img


def _hills() -> Image.Image:
    img = _vertical_gradient((120, 180, 255), (230, 245, 255))
    d = ImageDraw.Draw(img)
    rng = np.random.RandomState(5)
    xs = np.arange(0, W + 1, 4)
    for layer, (base, amp, col) in enumerate([(250, 40, (90, 140, 110)), (310, 50, (60, 120, 80)),
                                              (380, 40, (35, 95, 55))]):
        ys = base + amp * np.sin(xs / (70 + 25 * layer) + layer * 1.7)
        d.polygon([(0, H)] + [(int(x), int(y)) for x, y in zip(xs, ys)] + [(W, H)], fill=col)
    for _ in range(45):                                       # trees
        tx, ty = rng.randint(10, W - 10), rng.randint(340, H - 10)
        s = rng.randint(14, 30)
        d.polygon([(tx, ty - s * 2), (tx - s // 2, ty), (tx + s // 2, ty)], fill=(20, 70, 35))
        d.line([(tx, ty), (tx, ty + 5)], fill=(70, 45, 20), width=2)
    for fx in range(0, W, 10):                                # fence-like fine pattern
        d.line([(fx, 440), (fx, 470)], fill=(240, 240, 240), width=1)
    d.line([(0, 450), (W, 450)], fill=(240, 240, 240), width=1)
    arr = np.asarray(img, dtype=np.float32) + rng.normal(0, 3, (H, W, 3))  # mild grain
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


_BUILDERS = {"city_skyline": _skyline, "text_poster": _text_poster, "hills_trees": _hills}


def _degrade(hr: Image.Image) -> bytes:
    """Blur, shrink to half size and JPEG-compress -> realistic 'low quality' input."""
    lr = hr.filter(ImageFilter.GaussianBlur(0.9)).resize((W // 2, H // 2), Image.BICUBIC)
    buf = io.BytesIO()
    lr.save(buf, "JPEG", quality=65)
    return buf.getvalue()


def ensure_demo_images() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for slug, build in _BUILDERS.items():
        ref_path = DEMO_DIR / f"{slug}_reference_hr.png"
        low_path = DEMO_DIR / f"{slug}_lowres.jpg"
        if ref_path.exists() and low_path.exists():
            continue
        hr = build()
        hr.save(ref_path)
        low_path.write_bytes(_degrade(hr))


def demo_paths(label: str) -> tuple[Path, Path]:
    """(low-res input path, high-res reference path) for a demo label."""
    slug = DEMOS[label]
    return DEMO_DIR / f"{slug}_lowres.jpg", DEMO_DIR / f"{slug}_reference_hr.png"
