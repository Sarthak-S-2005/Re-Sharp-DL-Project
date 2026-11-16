"""
RE-SHARP - AI-Powered Image Super-Resolution
Run with:  streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from models.model_manager import MODEL_SPECS, ModelUnavailableError, load_model, weights_ready
from utils.demo_images import DEMOS, demo_paths, ensure_demo_images
from utils.image_utils import (
    MAX_FILE_BYTES,
    MAX_PROCESS_SIDE,
    ImageValidationError,
    SourceImage,
    aspect_ratio,
    cap_for_processing,
    encode_image,
    format_bytes,
    slider_html,
    validate_and_load,
)
from utils.metrics import compute_scores, fmt_psnr, run_controlled_experiment
from utils.super_resolution import bicubic_upscale, enhance_image, get_device

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

st.set_page_config(page_title="RE-SHARP | AI Super-Resolution", page_icon="✨", layout="wide")

# --------------------------------------------------------------------------- styling
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, .stApp, button, input, select, textarea { font-family: 'Inter','Segoe UI',system-ui,sans-serif !important; }
  .stApp {
    background:
      radial-gradient(900px 480px at 8% -8%, #e6e0ff 0%, transparent 60%),
      radial-gradient(800px 460px at 100% 0%, #ffe1f0 0%, transparent 55%),
      #faf9ff;
  }
  .block-container { padding-top: 1.4rem; max-width: 1200px; }
  [data-testid="stSidebar"] { background: rgba(255,255,255,.72); backdrop-filter: blur(12px);
          border-right: 1px solid rgba(124,58,237,.12); }
  .hero { position:relative; overflow:hidden; text-align:center; padding: 2.4rem 1rem 2rem; border-radius: 26px;
          background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 55%,#db2777 100%); color:#fff;
          box-shadow: 0 18px 44px rgba(79,70,229,.28); margin-bottom: 1.6rem; }
  .hero::before, .hero::after { content:""; position:absolute; border-radius:50%; background:rgba(255,255,255,.10); }
  .hero::before { width:260px; height:260px; top:-110px; left:-70px; }
  .hero::after  { width:200px; height:200px; bottom:-90px; right:-40px; }
  .hero h1 { position:relative; font-size: 2.9rem; font-weight:800; margin:0; letter-spacing:.07em; color:#fff; }
  .hero .sub { position:relative; margin:.35rem 0 1rem; font-size:1.08rem; font-weight:500; opacity:.95; }
  .hero .pill { position:relative; display:inline-block; margin:0 .25rem; padding:.28rem .9rem; border-radius:999px;
                background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.35); font-size:.85rem; font-weight:600; }
  h2, h3 { color:#2b2557; font-weight:700 !important; }
  .card { border:1px solid rgba(124,58,237,.16); border-radius:16px; padding:.9rem 1.1rem;
          background: rgba(255,255,255,.78); box-shadow: 0 6px 20px rgba(79,70,229,.08); margin-bottom:.7rem; }
  .card .tag { font-size:.72rem; font-weight:800; letter-spacing:.14em; color:#7c3aed; }
  .card .big { font-size:1.25rem; font-weight:700; color:#2b2557; }
  .arrow { text-align:center; font-weight:700; letter-spacing:.18em; color:#7c3aed; margin:.6rem 0 1.1rem; }
  .small-note { font-size:.85rem; color:#6b6890; }
  [data-testid="stMetric"] { background: rgba(255,255,255,.8); border:1px solid rgba(124,58,237,.16);
          border-radius:16px; padding:.8rem 1rem; box-shadow:0 6px 18px rgba(79,70,229,.07); }
  [data-testid="stFileUploaderDropzone"] { border:2px dashed rgba(124,58,237,.38); border-radius:18px;
          background: rgba(255,255,255,.72); }
  div.stButton > button, div.stDownloadButton > button { border-radius:14px; font-weight:700; padding:.65rem 1.1rem;
          transition: transform .15s ease, box-shadow .15s ease; }
  div.stButton > button:hover, div.stDownloadButton > button:hover { transform: translateY(-2px);
          box-shadow:0 10px 22px rgba(79,70,229,.22); }
  button[kind="primary"], [data-testid="stBaseButton-primary"] {
          background: linear-gradient(135deg,#4f46e5,#7c3aed 60%,#db2777) !important; border:none !important; color:#fff !important; }
  .stTabs [data-baseweb="tab-list"] { gap:.4rem; }
  .stTabs [data-baseweb="tab"] { border-radius:12px 12px 0 0; padding:.55rem 1rem; font-weight:600; }
  .stTabs [aria-selected="true"] { color:#4f46e5 !important; }
  [data-testid="stImage"] img { border-radius:14px; box-shadow:0 8px 24px rgba(30,30,80,.14); }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>✨ RE-SHARP</h1>
  <div class="sub">AI-Powered Image Super-Resolution</div>
  <span class="pill">Restore</span><span class="pill">Enhance</span><span class="pill">Upscale</span>
</div>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner=False)
def cached_device():
    return get_device()


@st.cache_resource(show_spinner=False)
def cached_model(scale: int, _progress_cb=None):
    """Load (and on first use download) the pretrained model. Cached across reruns."""
    return load_model(scale, cached_device(), _progress_cb)


@st.cache_resource(show_spinner=False)
def prepare_demos() -> bool:
    ensure_demo_images()
    return True


# --------------------------------------------------------------------------- session state
for key, default in (("source", None), ("source_key", None), ("source_error", None),
                     ("result", None), ("experiment", None), ("uploader_n", 0)):
    st.session_state.setdefault(key, default)

device = cached_device()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    scale = st.radio(
        "Upscaling factor",
        [2, 4],
        format_func=lambda s: "2×  (recommended – faster)" if s == 2 else "4×  (slower – small images)",
    )
    st.caption(
        f"For CPU speed, inputs are limited to {MAX_PROCESS_SIDE[scale]} px (long side) at {scale}×. "
        "Larger images are resized down first, with a warning."
    )
    out_mode = st.selectbox("Download format", ["Auto (match input)", "PNG (lossless)"])
    out_mode = "Auto" if out_mode.startswith("Auto") else "PNG"

    st.divider()
    st.markdown("**Model:** " + MODEL_SPECS[scale].name)
    st.caption("Pretrained CNN (RRDBNet). No training happens in this app.")
    st.markdown("**Device:** " + ("GPU (CUDA)" if device.type == "cuda" else "CPU"))
    if weights_ready(scale):
        st.success("Model weights found ✔")
    else:
        st.info("Weights (~67 MB) download automatically on first use.")

    st.divider()
    if st.button("🔄 Process another image", use_container_width=True):
        st.session_state.uploader_n += 1
        for k in ("source", "source_key", "source_error", "result", "experiment"):
            st.session_state[k] = None
        st.rerun()

# --------------------------------------------------------------------------- input section
st.subheader("1 · Upload your low-resolution or blurry image")
up_col, demo_col = st.columns([3, 2])
with up_col:
    uploaded = st.file_uploader(
        f"Drag & drop or browse (JPG, JFIF, PNG, WEBP – max {MAX_FILE_BYTES // (1024 * 1024)} MB)",
        type=["jpg", "jpeg", "jfif", "png", "webp"],
        key=f"uploader_{st.session_state.uploader_n}",
    )
with demo_col:
    prepare_demos()
    demo_label = st.selectbox("…or try a demo image", ["— none —"] + list(DEMOS.keys()))

# Work out which source is selected and (re)load it only when it changes
if uploaded is not None:
    new_key = ("upload", uploaded.name, uploaded.size)
elif demo_label != "— none —":
    new_key = ("demo", demo_label)
else:
    new_key = None

if new_key != st.session_state.source_key:
    st.session_state.source_key = new_key
    st.session_state.source = None
    st.session_state.source_error = None
    st.session_state.result = None
    st.session_state.experiment = None
    try:
        if new_key and new_key[0] == "upload":
            st.session_state.source = validate_and_load(uploaded.name, uploaded.getvalue())
        elif new_key and new_key[0] == "demo":
            low_path, ref_path = demo_paths(demo_label)
            src = validate_and_load(low_path.name, low_path.read_bytes())
            src.reference = Image.open(ref_path).convert("RGB")
            st.session_state.source = src
    except ImageValidationError as exc:
        st.session_state.source_error = str(exc)
    except Exception as exc:  # never crash the UI because of a bad file
        st.session_state.source_error = f"Could not read this image: {exc}"

src: SourceImage | None = st.session_state.source

if st.session_state.source_error:
    st.error("⚠️ " + st.session_state.source_error)
if src is None:
    if not st.session_state.source_error:
        st.info("👆 Upload an image (or pick a demo image) to get started.")
    st.stop()


# --------------------------------------------------------------------------- helpers
def preview(img: Image.Image, max_side: int = 1400) -> Image.Image:
    """Smaller copy used ONLY to display very large uploads quickly (processing uses the full image)."""
    longest = max(img.size)
    if longest <= max_side:
        return img
    f = max_side / longest
    return img.resize((round(img.width * f), round(img.height * f)), Image.LANCZOS)


def info_card(tag: str, w: int, h: int, size_text: str, fmt: str):
    st.markdown(
        f"""<div class="card"><div class="tag">{tag}</div>
        <div class="big">{w} × {h} px</div>
        <div class="small-note">file size: {size_text} &nbsp;|&nbsp; format: {fmt}</div></div>""",
        unsafe_allow_html=True,
    )


def props_table(rows: dict[str, str], title: str):
    st.markdown(f"**{title}**")
    st.table(pd.DataFrame({"Value": list(rows.values())}, index=list(rows.keys())))


def run_enhancement(src: SourceImage, scale: int, out_mode: str):
    bar = st.progress(0.0, text="Preparing the AI model…")
    try:
        if not weights_ready(scale):
            bar.progress(0.0, text=f"Downloading {MODEL_SPECS[scale].name} weights (first run only, ~67 MB)…")
        model = cached_model(
            scale, _progress_cb=lambda p: bar.progress(min(p, 1.0), text=f"Downloading model weights… {int(p * 100)}%")
        )

        proc_img, was_capped = cap_for_processing(src.image, scale)
        bar.progress(0.0, text="Running AI super-resolution… (CPU inference can take a while)")
        t0 = time.perf_counter()
        sr = enhance_image(
            proc_img, model, scale, device,
            progress_cb=lambda p: bar.progress(min(p, 1.0), text=f"AI is reconstructing details… {int(p * 100)}%"),
        )
        elapsed = time.perf_counter() - t0

        bicubic = bicubic_upscale(proc_img, size=sr.size)

        out_img = sr
        if src.alpha is not None:  # keep transparency: upscale the alpha channel with plain resizing
            out_img = sr.copy()
            out_img.putalpha(src.alpha.resize(sr.size, Image.LANCZOS))
        data, mime, ext = encode_image(out_img, src.fmt, out_mode)

        # Reference-based metrics ONLY when a true high-res reference of identical size exists (demo images).
        ref_scores = None
        if src.reference is not None and src.reference.size == sr.size:
            ref_scores = {"ai": compute_scores(src.reference, sr), "bicubic": compute_scores(src.reference, bicubic)}

        try:  # best-effort copy to ./outputs
            OUTPUT_DIR.mkdir(exist_ok=True)
            (OUTPUT_DIR / f"{Path(src.name).stem}_resharp_{scale}x.{ext}").write_bytes(data)
        except OSError:
            pass

        st.session_state.result = {
            "scale": scale, "elapsed": elapsed, "sr": sr, "bicubic": bicubic, "proc_img": proc_img,
            "was_capped": was_capped, "bytes": data, "mime": mime, "ext": ext, "ref_scores": ref_scores,
            "download_name": f"{Path(src.name).stem}_resharp_{scale}x.{ext}",
        }
        st.session_state.experiment = None
        bar.empty()
    except ModelUnavailableError as exc:
        bar.empty()
        st.error("❌ The AI model is not available.")
        st.code(str(exc), language="text")
    except MemoryError:
        bar.empty()
        st.error("❌ Not enough memory for this image. Try a smaller image or the 2× option.")
    except RuntimeError as exc:
        bar.empty()
        if "out of memory" in str(exc).lower():
            st.error("❌ Out of memory. Try a smaller image or the 2× option.")
        else:
            st.error("❌ The model failed while processing this image.")
            with st.expander("Technical details"):
                st.code(str(exc))
    except Exception as exc:
        bar.empty()
        st.error("❌ Something went wrong while enhancing the image.")
        with st.expander("Technical details"):
            st.code(repr(exc))


# --------------------------------------------------------------------------- original preview + enhance button
st.subheader("2 · Review and enhance")
left, right = st.columns([3, 2])
with left:
    st.image(preview(src.image), caption=f"Original: {src.name}", use_container_width=True)
with right:
    props_table(
        {
            "Filename": src.name,
            "Format": src.fmt,
            "Width": f"{src.image.width} px",
            "Height": f"{src.image.height} px",
            "Aspect ratio": aspect_ratio(*src.image.size),
            "File size": format_bytes(src.file_bytes),
        },
        "Uploaded image",
    )
    limit = MAX_PROCESS_SIDE[scale]
    if max(src.image.size) > limit:
        st.warning(
            f"Large image: for CPU speed it will first be resized so its long side is {limit} px "
            f"before AI processing. Choose 2× or a smaller image to keep more of the original size."
        )
    st.caption("CPU inference can take from a few seconds to a couple of minutes depending on image size.")
    if st.button("✨ Enhance Image", type="primary", use_container_width=True):
        run_enhancement(src, scale, out_mode)

res = st.session_state.result
if res is None:
    st.stop()

# --------------------------------------------------------------------------- results
sr: Image.Image = res["sr"]
st.markdown('<div class="arrow">↓ &nbsp; AI PROCESSING COMPLETE &nbsp; ↓</div>', unsafe_allow_html=True)
st.subheader("3 · Results")

if res["was_capped"]:
    st.warning(
        f"The input was resized to {res['proc_img'].width} × {res['proc_img'].height} px before AI processing "
        f"(CPU limit), so the final size is {sr.width} × {sr.height} px."
    )

m1, m2, m3, m4 = st.columns(4)
m1.metric("Original", f"{src.image.width} × {src.image.height}")
m2.metric("Enhanced", f"{sr.width} × {sr.height}")
m3.metric("Scale", f"{res['scale']}×")
m4.metric("Processing", f"{res['elapsed']:.2f} s")

st.download_button(
    "⬇  DOWNLOAD ENHANCED IMAGE",
    data=res["bytes"],
    file_name=res["download_name"],
    mime=res["mime"],
    type="primary",
    use_container_width=True,
)

tab_side, tab_slider, tab_bicubic, tab_info, tab_exp, tab_how = st.tabs(
    ["Side by side", "Before / After slider", "Bicubic vs AI", "Details & metrics",
     "Controlled experiment", "How it works"]
)

# ---- tab 1: side by side
with tab_side:
    c1, c2 = st.columns(2)
    with c1:
        info_card("ORIGINAL", src.image.width, src.image.height, format_bytes(src.file_bytes), src.fmt)
        st.image(preview(src.image), use_container_width=True)
    with c2:
        info_card("AI ENHANCED", sr.width, sr.height, format_bytes(len(res["bytes"])), res["ext"].upper())
        st.image(sr, use_container_width=True)

# ---- tab 2: slider
with tab_slider:
    st.caption("Drag the handle left/right. The original is stretched to the enhanced size so both halves line up.")
    html, height = slider_html(src.image, sr)
    components.html(html, height=height, scrolling=False)

# ---- tab 3: bicubic vs AI
with tab_bicubic:
    st.markdown(
        "**Traditional interpolation** estimates new pixels with a fixed mathematical formula. "
        "**Deep-learning super-resolution** uses a network that learned image patterns from training data. "
        "Look at the zoomed crops and judge the actual results yourself – the AI result is not guaranteed to be better."
    )
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Result A – Bicubic upscaling**")
        st.image(res["bicubic"], use_container_width=True)
    with cb:
        st.markdown(f"**Result B – Deep learning ({MODEL_SPECS[res['scale']].name})**")
        st.image(sr, use_container_width=True)

    st.markdown("##### 🔍 Zoom into the same region")
    z1, z2, z3 = st.columns(3)
    zoom = z1.slider("Zoom", 2, 8, 4)
    cx = z2.slider("Horizontal position (%)", 0, 100, 50)
    cy = z3.slider("Vertical position (%)", 0, 100, 50)
    cw, ch = max(8, sr.width // zoom), max(8, sr.height // zoom)
    left_px = int((sr.width - cw) * cx / 100)
    top_px = int((sr.height - ch) * cy / 100)
    box = (left_px, top_px, left_px + cw, top_px + ch)
    show_w = 640
    show_size = (show_w, max(1, round(show_w * ch / cw)))
    za, zb = st.columns(2)
    za.image(res["bicubic"].crop(box).resize(show_size, Image.NEAREST), caption="Bicubic (zoomed)", use_container_width=True)
    zb.image(sr.crop(box).resize(show_size, Image.NEAREST), caption="AI (zoomed)", use_container_width=True)

# ---- tab 4: info + metrics
with tab_info:
    i1, i2 = st.columns(2)
    with i1:
        props_table(
            {
                "Filename": src.name,
                "Format": src.fmt,
                "Width": f"{src.image.width} px",
                "Height": f"{src.image.height} px",
                "Aspect ratio": aspect_ratio(*src.image.size),
                "File size": format_bytes(src.file_bytes),
            },
            "Original image",
        )
    with i2:
        props_table(
            {
                "Format": res["ext"].upper(),
                "Width": f"{sr.width} px",
                "Height": f"{sr.height} px",
                "Aspect ratio": aspect_ratio(*sr.size),
                "File size": format_bytes(len(res["bytes"])),
                "Scale factor": f"{res['scale']}×",
                "Processing time": f"{res['elapsed']:.2f} s ({'GPU' if device.type == 'cuda' else 'CPU'})",
            },
            "Enhanced image",
        )

    st.markdown("#### Image-quality metrics (PSNR / SSIM)")
    st.info(
        "**Not available for arbitrary uploaded images.** PSNR and SSIM compare an image against a TRUE "
        "high-resolution reference. For a normal uploaded photo, no such reference exists, and inventing one "
        "(or comparing the output with itself) would give meaningless numbers. "
        "Judge uploaded images visually, or use the *Controlled experiment* tab."
    )
    if res["ref_scores"]:
        st.success("This demo image comes with a TRUE high-resolution reference, so valid metrics can be computed:")
        rs = res["ref_scores"]
        st.table(
            pd.DataFrame(
                {
                    "PSNR (higher = closer)": [fmt_psnr(rs["bicubic"].psnr), fmt_psnr(rs["ai"].psnr)],
                    "SSIM (max 1.0)": [f"{rs['bicubic'].ssim:.4f}", f"{rs['ai'].ssim:.4f}"],
                },
                index=["Bicubic upscaling", "Deep learning (Real-ESRGAN)"],
            )
        )
        st.caption("Numbers measure pixel/structure similarity to the reference – not necessarily how sharp or "
                   "pleasing the result looks.")
    elif src.reference is not None:
        st.caption("A reference exists for this demo, but only at 2× without resizing – select 2× to see the metrics.")

# ---- tab 5: controlled experiment
with tab_exp:
    st.markdown(
        "A **controlled experiment** creates a meaningful reference: the image is treated as *ground truth*, "
        "shrunk by the selected factor to make a low-resolution input, then rebuilt with bicubic and with the "
        "AI model. Both results are compared with the ground truth using PSNR and SSIM."
    )
    st.caption("Fair warning: Real-ESRGAN is trained for real-world degradations and favours sharp-looking output, "
               "so its PSNR/SSIM can be lower than bicubic on this clean synthetic test even when it looks better. "
               "Numbers and eyes can disagree.")
    if st.button("🧪 Run controlled experiment"):
        bar = st.progress(0.0, text="Running experiment…")
        try:
            exp_scale = res["scale"]
            model = cached_model(exp_scale)
            exp = run_controlled_experiment(
                src.image, exp_scale,
                lambda lr: enhance_image(lr, model, exp_scale, device,
                                         progress_cb=lambda p: bar.progress(min(p, 1.0), text=f"AI is working… {int(p * 100)}%")),
            )
            exp["scale"] = exp_scale
            st.session_state.experiment = exp
        except ModelUnavailableError as exc:
            st.error("❌ The AI model is not available.")
            st.code(str(exc), language="text")
        except Exception as exc:
            st.error("❌ The experiment failed.")
            with st.expander("Technical details"):
                st.code(repr(exc))
        bar.empty()

    exp = st.session_state.experiment
    if exp:
        st.table(
            pd.DataFrame(
                {
                    "PSNR (higher = closer)": [fmt_psnr(exp["bicubic_scores"].psnr), fmt_psnr(exp["ai_scores"].psnr)],
                    "SSIM (max 1.0)": [f"{exp['bicubic_scores'].ssim:.4f}", f"{exp['ai_scores'].ssim:.4f}"],
                },
                index=["Bicubic upscaling", "Deep learning (Real-ESRGAN)"],
            )
        )
        e1, e2, e3, e4 = st.columns(4)
        e1.image(exp["hr"], caption=f"Ground truth {exp['hr'].width}×{exp['hr'].height}", use_container_width=True)
        e2.image(exp["lr"], caption=f"Low-res input {exp['lr'].width}×{exp['lr'].height}", use_container_width=True)
        e3.image(exp["bicubic"], caption="Bicubic", use_container_width=True)
        e4.image(exp["ai"], caption="AI (Real-ESRGAN)", use_container_width=True)

# ---- tab 6: how it works
with tab_how:
    st.markdown(
        """
**Pipeline**

```
Upload → Validate → Preprocess (RGB, size cap, tiling)
       → CNN feature extraction (23 RRDB blocks of dense + residual convolutions)
       → Upsampling layers + reconstruction convolutions
       → Enhanced image → Side-by-side / Slider / Download
```

* **CNN feature extraction:** stacked convolution layers turn pixels into feature maps (edges, textures, patterns).
* **Super-resolution / reconstruction:** upsampling layers enlarge those feature maps and the last convolutions
  rebuild an RGB image with twice (or four times) the width and height.
* **Pretrained:** the weights were trained by the Real-ESRGAN authors on large image datasets. This app does **not**
  train anything.

**Honest limitations**

* The model *estimates* a higher-resolution image from learned patterns – it does not recover the true lost detail.
* Extremely blurry images cannot be fully restored; results depend on image content.
* Super-resolution can introduce artifacts or invent plausible-looking but wrong detail.
* Educational prototype – not a forensic or professional restoration tool.
"""
    )

st.markdown("---")
st.caption("RE-SHARP is an educational Deep Learning prototype using pretrained Real-ESRGAN weights "
           "(xinntao/Real-ESRGAN, BSD-3-Clause). Results are AI estimates, not exact originals.")
