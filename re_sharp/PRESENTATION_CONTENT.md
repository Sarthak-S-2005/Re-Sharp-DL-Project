# RE-SHARP — Presentation Content (6 Slides)

---

## Slide 1 — Title + Problem

**Title:** RE-SHARP: AI-Powered Image Super-Resolution

**Bullet points**
- Team name, members, guide/professor, course (Deep Learning)
- Problem: low-resolution or compressed images look blocky/blurry when enlarged
- Goal: enlarge an image and reconstruct a sharper-looking version using a CNN
- Built with PyTorch + Streamlit, runs on a normal laptop (CPU)

**Show:** A low-resolution demo image enlarged by simple zoom (visibly soft/blocky). Add the RE-SHARP logo/header screenshot.

**Presenter says:**
"Everyone has seen an image turn blurry when you zoom in. Our project, RE-SHARP, takes a small or slightly blurry image
and uses a deep learning model to produce a larger, sharper-looking version. We built a complete web application around
it so anyone can upload an image, compare the result and download it."

---

## Slide 2 — Why Traditional Upscaling Is Limited

**Title:** Why Interpolation Is Not Enough

**Bullet points**
- Nearest / bilinear / bicubic = fixed mathematical formulas on neighbouring pixels
- They only *average* existing pixels — they cannot add new detail
- Result: soft edges, blur, blocky or ringing artifacts
- No knowledge of what real images (edges, textures, text) look like

**Show:** Zoomed crop of the **Bicubic** result from the app (“Bicubic vs AI” tab, zoom 4×).

**Presenter says:**
"Traditional upscaling, such as bicubic interpolation, calculates new pixels as weighted averages of nearby pixels.
It has no idea what an edge or a texture should look like, so the result is smooth but soft. It estimates pixels
mathematically; it does not learn from data."

---

## Slide 3 — Our Deep Learning Solution

**Title:** CNN-Based Super-Resolution with Real-ESRGAN

**Bullet points**
- Pretrained **Real-ESRGAN** (RRDBNet CNN generator): 2× and 4× models
- **CNN feature extraction:** 23 RRDB blocks of dense + residual convolution layers
- **Reconstruction:** upsampling layers + convolutions rebuild a high-res RGB image
- Pretrained by the model authors — we run inference, we do not train from scratch
- Honest note: it is *not* an autoencoder

**Show:** Simple block diagram: Input → conv → RRDB × 23 → upsample → conv → Output (from README architecture section).

**Presenter says:**
"Our model is a convolutional neural network called RRDBNet from Real-ESRGAN. The convolution layers extract features
such as edges and textures. Residual and dense connections help it learn the missing detail. Then upsampling layers
enlarge the feature maps and the last layers reconstruct the high-resolution image. We use official pretrained weights,
because training such a network from scratch needs huge datasets and GPUs, and we had about two hours."

---

## Slide 4 — Architecture / Workflow

**Title:** System Architecture

**Bullet points**
- Upload → Validation (type, size, corruption) → Preprocessing (RGB, size cap, tiling)
- Deep Learning model (feature extraction → super-resolution)
- Enhanced image → Visualization (side-by-side, slider, bicubic vs AI) and Download
- Tiled CPU inference so large images do not crash a laptop
- Friendly error handling at every step

**Show:** The ASCII/flow architecture diagram from the README redrawn as a clean slide graphic, plus a screenshot of the app's upload screen.

**Presenter says:**
"The user uploads an image. We validate it, convert it to RGB and, if it is very large, cap it so the CPU can handle it.
The image is processed in small tiles by the CNN and stitched back together. The enhanced image goes to the interface
for visual comparison, and to the download button."

---

## Slide 5 — Before vs After + Bicubic vs AI

**Title:** Results: Before / After and Bicubic vs AI

**Bullet points**
- Original vs enhanced: resolution (e.g., 360 × 240 → 720 × 480), file size, format, processing time
- Interactive before/after slider
- Result A (bicubic) vs Result B (deep learning), zoomed on the same region
- Let the visuals speak — AI is not guaranteed to be better on every image
- PSNR/SSIM only with a true reference (demo + controlled experiment)

**Show:** Screenshot of the slider tab; screenshot of the Bicubic vs AI tab with zoom 4× on text or windows (use the “Text poster” or “City skyline” demo).

**Presenter says:**
"Here is the original and here is the AI-enhanced result. Move the slider to compare. Now compare bicubic on the left
with the deep learning result on the right — zoom in on the fine details. We do not claim AI is always better; on some
images it can add artifacts. We also avoid fake numbers: PSNR and SSIM need a real high-resolution reference, so we only
show them for demo images and our controlled experiment."

---

## Slide 6 — Results, Limitations and Future Scope

**Title:** Results, Limitations, Future Scope

**Bullet points**
- Result: working end-to-end app — upload, enhance, compare, download, on CPU
- Limitations: the model *estimates* detail from learned patterns; very blurry images cannot be truly restored; possible artifacts; CPU is slow for large images; pretrained model, not trained by us; educational prototype
- Future scope: GPU/batch processing, face enhancement, denoising/deblurring, video SR, model comparison, LPIPS metric

**Show:** Screenshot of the “Details & metrics” tab (info tables + “Not available for arbitrary image” note) and a future-scope icon list.

**Presenter says:**
"We delivered a complete application that enlarges and enhances images with a real deep learning model. We are honest
about limits: the model estimates a higher-resolution version from patterns it learned — it does not recover the exact
original details, and it can sometimes introduce artifacts. In future we can add GPU support, face enhancement and video.
Thank you — we are happy to take questions."
