"""Regenerate the demo images in assets/demo_images (the app also does this automatically)."""
from utils.demo_images import DEMO_DIR, ensure_demo_images

ensure_demo_images()
print("Demo images ready in:", DEMO_DIR)
