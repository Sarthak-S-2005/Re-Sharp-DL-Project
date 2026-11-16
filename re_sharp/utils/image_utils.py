"""Image validation, information, encoding and the HTML before/after slider.

Everything here is plain Pillow - no deep learning - so it is fast and easy to test.
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageOps

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_FILE_BYTES = 25 * 1024 * 1024      # reject uploads larger than 25 MB
MIN_SIDE = 16                          # reject images smaller than 16 px on either side
MAX_PIXELS = 50_000_000                # refuse absurdly large images before decoding them (50 MP)

# Longest input side (pixels) that we hand to the network, per scale factor.
# Chosen so a typical student laptop CPU finishes in well under ~1-2 minutes.
MAX_PROCESS_SIDE = {2: 800, 4: 320}


class ImageValidationError(ValueError):
    """Raised with a friendly message when an upload cannot be used."""


@dataclass
class SourceImage:
    """A validated input image plus everything the UI wants to display about it."""
    name: str
    fmt: str                          # "JPEG" | "PNG" | "WEBP"
    image: Image.Image                # RGB pixels (EXIF-rotated)
    file_bytes: int                   # size of the uploaded file on disk
    alpha: Optional[Image.Image] = None      # separate alpha channel ("L") for transparent PNG/WEBP
    reference: Optional[Image.Image] = None  # TRUE high-res ground truth (demo images only)


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / 1024 ** 2:.2f} MB"


def aspect_ratio(width: int, height: int) -> str:
    """'3:2' for simple ratios, otherwise a decimal such as '1.78:1'."""
    g = math.gcd(width, height)
    w, h = width // g, height // g
    if w <= 32 and h <= 32:
        return f"{w}:{h}"
    return f"{width / height:.2f}:1"


# --------------------------------------------------------------------------- #
# Loading + validation
# --------------------------------------------------------------------------- #
def validate_and_load(filename: str, data: bytes) -> SourceImage:
    """Validate raw upload bytes and return a SourceImage. Raises ImageValidationError."""
    if not data:
        raise ImageValidationError("The file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise ImageValidationError(
            f"File is {format_bytes(len(data))}; the limit is {format_bytes(MAX_FILE_BYTES)}. "
            "Please upload a smaller image."
        )
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '{ext or 'unknown'}'. Please upload a JPG, JFIF, PNG or WEBP image."
        )

    try:
        Image.open(io.BytesIO(data)).verify()        # cheap integrity check
        img = Image.open(io.BytesIO(data))            # verify() invalidates the object -> reopen
        fmt = img.format
        if fmt not in ALLOWED_FORMATS:
            raise ImageValidationError(
                f"Unsupported image content ({fmt}). Please upload a JPG, PNG or WEBP image."
            )
        if img.size[0] * img.size[1] > MAX_PIXELS:
            raise ImageValidationError(
                f"This image has more than {MAX_PIXELS // 1_000_000} megapixels. Please resize it first."
            )
        img.load()                                    # fully decode (catches truncated files)
    except ImageValidationError:
        raise
    except Exception as exc:                          # corrupted / truncated / not an image
        raise ImageValidationError(
            "This file could not be read as an image - it may be corrupted. Please try another file."
        ) from exc

    img = ImageOps.exif_transpose(img)                # respect phone-camera rotation
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    alpha = None
    if has_alpha:
        rgba = img.convert("RGBA")
        alpha = rgba.getchannel("A")
        img = rgba.convert("RGB")
    else:
        img = img.convert("RGB")

    if min(img.size) < MIN_SIDE:
        raise ImageValidationError(
            f"Image is too small ({img.width} x {img.height}). The minimum is {MIN_SIDE} x {MIN_SIDE} pixels."
        )
    return SourceImage(name=filename, fmt=fmt, image=img, file_bytes=len(data), alpha=alpha)


def cap_for_processing(img: Image.Image, scale: int) -> Tuple[Image.Image, bool]:
    """Downscale (never upscale) so the long side <= MAX_PROCESS_SIDE[scale]. Returns (image, was_capped)."""
    limit = MAX_PROCESS_SIDE[scale]
    w, h = img.size
    if max(w, h) <= limit:
        return img, False
    ratio = limit / max(w, h)
    return img.resize((max(1, round(w * ratio)), max(1, round(h * ratio))), Image.LANCZOS), True


# --------------------------------------------------------------------------- #
# Encoding for download
# --------------------------------------------------------------------------- #
def encode_image(img: Image.Image, source_fmt: str, out_mode: str = "Auto") -> Tuple[bytes, str, str]:
    """Encode the enhanced image. Returns (bytes, mime, extension).

    Auto: JPEG input -> high-quality JPEG (quality 95); PNG/WEBP input -> lossless PNG.
    PNG : always lossless PNG. Images with transparency are always PNG.
    """
    buf = io.BytesIO()
    if out_mode == "Auto" and source_fmt == "JPEG" and img.mode != "RGBA":
        img.convert("RGB").save(buf, format="JPEG", quality=95, subsampling=0)
        return buf.getvalue(), "image/jpeg", "jpg"
    img.save(buf, format="PNG", compress_level=6)
    return buf.getvalue(), "image/png", "png"


# --------------------------------------------------------------------------- #
# Before / after slider (self-contained HTML, no extra packages)
# --------------------------------------------------------------------------- #
def _data_uri(img: Image.Image, max_width: int) -> str:
    if img.width > max_width:
        img = img.resize((max_width, max(1, round(img.height * max_width / img.width))), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


_SLIDER_TEMPLATE = """
<style>
  body {margin:0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: transparent;}
  .cmp {position:relative; margin:0 auto; max-width:__MAXW__px; aspect-ratio:__W__ / __H__;
        overflow:hidden; border-radius:14px; box-shadow:0 8px 28px rgba(15,23,42,.28);
        user-select:none; background:#0F172A;}
  .cmp img {position:absolute; inset:0; width:100%; height:100%; object-fit:fill; display:block;}
  #before {clip-path: inset(0 50% 0 0);}
  .line {position:absolute; top:0; bottom:0; left:50%; width:3px; background:#fff;
         transform:translateX(-50%); box-shadow:0 0 8px rgba(0,0,0,.55); pointer-events:none;}
  .knob {position:absolute; top:50%; left:50%; width:38px; height:38px; margin:-19px 0 0 -19px;
         border-radius:50%; background:#4f46e5; border:3px solid #fff; color:#fff; font-weight:700;
         font-size:15px; display:flex; align-items:center; justify-content:center;
         pointer-events:none; box-shadow:0 2px 10px rgba(0,0,0,.4);}
  .tag {position:absolute; top:12px; padding:5px 12px; border-radius:999px; font-size:12px;
        font-weight:700; letter-spacing:.06em; color:#fff; background:rgba(15,23,42,.78);
        pointer-events:none;}
  .tl {left:12px;} .tr {right:12px; background:rgba(124,58,237,.92);}
  input[type=range] {position:absolute; inset:0; width:100%; height:100%; opacity:0;
                     cursor:ew-resize; margin:0;}
  .hint {text-align:center; color:#64748B; font-size:13px; margin-top:10px;}
</style>
<div class="cmp">
  <img id="after" src="__AFTER__" alt="AI enhanced">
  <img id="before" src="__BEFORE__" alt="Original">
  <div class="line" id="line"><div class="knob">&#8596;</div></div>
  <div class="tag tl">ORIGINAL</div><div class="tag tr">AI ENHANCED</div>
  <input type="range" min="0" max="100" value="50" id="slider" aria-label="Before after slider">
</div>
<div class="hint">Drag left / right to compare the original with the AI-enhanced image</div>
<script>
  const s = document.getElementById('slider');
  const b = document.getElementById('before');
  const l = document.getElementById('line');
  function update() { b.style.clipPath = 'inset(0 ' + (100 - s.value) + '% 0 0)'; l.style.left = s.value + '%'; }
  s.addEventListener('input', update); update();
</script>
"""


def slider_html(before: Image.Image, after: Image.Image) -> Tuple[str, int]:
    """Return (html, iframe_height) for st.components.v1.html. Both images share one box."""
    width, height = after.size
    max_display_w = int(min(800, 720 * width / height))
    iframe_height = int(min(800 * height / width, 720)) + 56
    page = (
        _SLIDER_TEMPLATE
        .replace("__MAXW__", str(max_display_w))
        .replace("__W__", str(width))
        .replace("__H__", str(height))
        .replace("__AFTER__", _data_uri(after, 1200))
        .replace("__BEFORE__", _data_uri(before, 1200))
    )
    return page, iframe_height
