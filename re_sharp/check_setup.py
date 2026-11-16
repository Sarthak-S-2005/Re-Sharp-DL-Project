"""
5-minute self-test. Run:  python check_setup.py

1. checks the libraries import
2. downloads / loads the 2x model
3. super-resolves a tiny test image on your CPU and prints the timing
"""
import time

print("[1/4] Checking libraries ...")
import numpy, PIL, torch  # noqa: E401
import streamlit  # noqa: E401,F401
print(f"      torch {torch.__version__} | numpy {numpy.__version__} | Pillow {PIL.__version__} | "
      f"streamlit {streamlit.__version__}")

from models.model_manager import load_model
from utils.demo_images import demo_paths, ensure_demo_images
from utils.image_utils import validate_and_load
from utils.super_resolution import bicubic_upscale, enhance_image, get_device

print("[2/4] Preparing demo images ...")
ensure_demo_images()
low, _ = demo_paths("Text poster")
src = validate_and_load(low.name, low.read_bytes())
small = src.image.crop((0, 0, 96, 64))

print("[3/4] Loading model (first run downloads ~67 MB) ...")
device = get_device()
model = load_model(2, device, lambda p: print(f"      download {int(p * 100)}%") if int(p * 100) % 20 == 0 else None)
print(f"      device = {device}")

print("[4/4] Running AI super-resolution on a 96x64 test patch ...")
t0 = time.perf_counter()
out = enhance_image(small, model, 2, device)
dt = time.perf_counter() - t0
bic = bicubic_upscale(small, size=out.size)
assert out.size == (192, 128), out.size
print(f"      OK: {small.size} -> {out.size} in {dt:.2f}s (bicubic baseline size {bic.size})")
print("\nALL GOOD. Start the app with:  streamlit run app.py")
