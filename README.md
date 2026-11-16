# ✨ RE-SHARP — AI-Powered Image Super-Resolution & Restoration

> Restore • Enhance • Upscale — a Deep Learning prototype built with PyTorch and Streamlit.

---

## 1. Project Title
**RE-SHARP: AI-Powered Image Super-Resolution & Restoration**

## 2. Project Overview
RE-SHARP is a web application where a user uploads a small, slightly blurry or compressed image and receives a
larger, visually enhanced version produced by a **pretrained Convolutional Neural Network (CNN)** for image
super-resolution. The app shows the original and enhanced image side by side, an interactive before/after slider,
image information (size, format, scale, processing time), a **bicubic vs. deep-learning** comparison, and a download
button.

## 3. Problem Statement
Low-resolution images (old photos, screenshots, compressed web images) look blocky or blurry when enlarged.
Traditional interpolation (nearest-neighbour, bilinear, bicubic) only averages neighbouring pixels with a fixed formula,
so it cannot create new, sharper-looking detail. The problem: *how can we enlarge an image while producing a
plausible, sharper high-resolution reconstruction?*

## 4. Motivation
* Super-resolution is a classic, visual application of CNNs that is easy to demonstrate.
* It shows the difference between *mathematical interpolation* and *learned reconstruction*.
* Useful in photography, medical/satellite imaging research, video, and old-photo recovery.

## 5. Objectives
1. Build a runnable end-to-end application around a **genuine pretrained deep learning** super-resolution model.
2. Let users compare original vs. enhanced images visually and download the result.
3. Compare the deep-learning output with **bicubic interpolation** honestly.
4. Report image-quality metrics **only when mathematically meaningful** (no fake PSNR/SSIM).
5. Run on an ordinary student laptop (CPU only).

## 6. Features
| Feature | Where |
|---|---|
| Upload JPG / PNG / WEBP with validation (type, 25 MB limit, corrupted file, tiny/huge images) | Upload section |
| Explicit **Enhance Image** button (no automatic heavy processing) | Step 2 |
| 2× and 4× upscaling (both supported by real pretrained models) | Sidebar |
| Original vs. AI-enhanced side-by-side with resolution / file size / format | Tab *Side by side* |
| Interactive **before/after slider** (pure HTML/JS, works offline) | Tab *Before / After slider* |
| Bicubic (Result A) vs. Deep Learning (Result B) + zoom into the same region | Tab *Bicubic vs AI* |
| Image info: filename, format, width, height, aspect ratio, file size, scale, processing time | Tab *Details & metrics* |
| PSNR / SSIM only with a true reference (demo images and controlled experiment) | Tabs *Details* / *Controlled experiment* |
| **Download enhanced image** (JPEG input → high-quality JPEG, otherwise PNG; or force PNG) | Under results |
| 3 built-in demo images (generated locally, no licensing issues) | Upload section |
| “Process another image” without restarting | Sidebar |
| Friendly errors (corrupted file, unsupported type, model unavailable, out-of-memory) | Everywhere |

## 7. Deep Learning Concepts Used
1. **CNN-based feature extraction** — stacked 3×3 convolution layers produce feature maps (edges → textures → patterns).
2. **Image reconstruction / super-resolution** — upsampling layers + convolutions rebuild a higher-resolution RGB image.
3. **Residual learning** — skip connections (`out = 0.2·F(x) + x`) make deep networks trainable.
4. **Dense connections** — each conv layer in a block sees all previous feature maps.
5. **Pretrained models / transfer of learned weights** — the model was trained by its authors; we only run inference.
6. **GAN-based training (background knowledge)** — Real-ESRGAN was trained with an adversarial + perceptual objective,
   which is why its outputs look sharp. (We do not train it here.)

> ⚠️ **Technical honesty:** the selected model is **not an autoencoder** and this project does **not** train any model.
> It is a feed-forward CNN generator (RRDBNet) used for inference with pretrained weights.

## 8. Architecture

```
                 USER
                   |
                   v
             Upload Image
                   |
                   v
        Validation + Preprocessing
   (type/size/corruption check, EXIF rotation, RGB,
    size cap for CPU, split into overlapping tiles)
                   |
                   v
        +-------------------------------+
        |  Deep Learning Model (RRDBNet)|
        |  conv_first                   |  <- shallow feature extraction
        |  23 x RRDB blocks             |  <- deep CNN feature extraction
        |  conv_body + skip connection  |
        |  2 x (upsample + conv)        |  <- upsampling / reconstruction
        |  conv_hr -> conv_last         |
        +-------------------------------+
                   |
                   v
            Enhanced Image
              /         \
             v           v
     Visualization     Download
   (side-by-side,     (PNG / JPEG)
    slider, bicubic
    vs AI, details)
```

### What happens inside the model (high level)
1. **conv_first** converts the RGB image into 64 feature maps.
2. **23 RRDB blocks** (Residual-in-Residual Dense Blocks) refine those features. Each RRDB = 3 *residual dense blocks*;
   each dense block = 5 convolution layers where every layer receives all earlier feature maps.
3. A **global skip connection** adds the first features back, so the network only needs to learn the *missing detail*.
4. **Upsampling**: the feature maps are enlarged ×2 (nearest-neighbour) and convolved, twice (= 4× internally).
5. **conv_hr + conv_last** convert the high-resolution feature maps back to a 3-channel RGB image.
6. For the **2× model**, the input is first rearranged by *pixel-unshuffle* (space-to-depth), so the same network
   produces a net 2× enlargement.

## 9. Technology Stack
* **Python 3.9 – 3.12**
* **PyTorch** — deep learning inference
* **Streamlit** — web UI
* **Pillow, NumPy** — image handling; PSNR / SSIM are implemented from scratch in NumPy (no extra library)
* **Model:** Real-ESRGAN `x2plus` / `x4plus` pretrained weights (RRDBNet architecture re-implemented in
  `models/rrdbnet.py`)

### Why Real-ESRGAN (and why this way)?
* It is a well-established, widely cited pretrained super-resolution model designed for real-world images
  (blur, noise, compression), which matches our use case.
* The popular `realesrgan` / `basicsr` pip packages frequently break with newer PyTorch/torchvision versions. To
  **maximise reliability**, we re-implemented only the small RRDBNet architecture (~100 lines) in plain PyTorch and load
  the **official weights** into it. Only `torch` is required.
* Weights come from the official release page: <https://github.com/xinntao/Real-ESRGAN/releases>
  (`RealESRGAN_x2plus.pth`, `RealESRGAN_x4plus.pth`, ~67 MB each, BSD-3-Clause licence).

## 10. System Workflow
1. User opens the app and uploads an image (or selects a demo).
2. The app validates the file and shows the original with its information.
3. User presses **Enhance Image**.
4. The image is capped to a CPU-friendly size if necessary (with a warning), then processed tile by tile by the CNN.
5. The enhanced image is displayed with side-by-side view, slider, bicubic comparison and details.
6. User downloads the enhanced image or processes another one.

## 11. Installation (Windows)

```bat
cd re_sharp
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Optional but recommended **before your demo** (downloads the ~67 MB weights once so nothing is downloaded live):

```bat
python download_weights.py 2
python check_setup.py
```

## 12. How to Run

```bat
venv\Scripts\activate
streamlit run app.py
```
The browser opens at <http://localhost:8501>.

If the automatic weight download is blocked (college Wi-Fi/firewall), download the file manually in your browser and put
it in the `weights/` folder:

* <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth>  → `weights/RealESRGAN_x2plus.pth`
* <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth>  → `weights/RealESRGAN_x4plus.pth`

## 13. How the Model Works (student-friendly)
* A **convolution layer** slides small 3×3 filters over the image. Early filters respond to edges, later ones to
  textures and shapes. The outputs are called **feature maps**.
* **Residual connections** let information skip layers, so the network focuses on learning *what detail is missing*.
* **Dense connections** reuse earlier features, helping the network model fine textures.
* **Upsampling layers** enlarge the feature maps, and the final convolutions turn them into an RGB image.
* During **training** (done by the model authors, not by us) the network saw many pairs of degraded/clean images and
  learned to map one to the other. Real-ESRGAN uses a GAN-style training objective, which encourages realistic textures.
* At **inference** (what our app does) the image just flows forward once through the network — no learning happens.

## 14. Traditional vs. Deep Learning Comparison
```
                 LOW-RES IMAGE
                      |
         +------------+------------+
         v                         v
  Bicubic upscaling        Deep-learning SR
  (fixed formula)          (learned patterns)
         v                         v
      Result A                  Result B
```
* **Bicubic** estimates each new pixel as a weighted average of its 16 nearest neighbours. It cannot add detail, so
  edges look soft.
* **Deep learning** estimates a higher-resolution reconstruction using patterns learned during training, so it can
  produce sharper edges and textures.
* We **do not claim** the AI result is always better. Use the *Bicubic vs AI* tab (with zoom) and judge visually;
  on some images the AI adds artifacts or over-sharpens.

## 15. Evaluation / Metrics
* **PSNR** (Peak Signal-to-Noise Ratio, dB) and **SSIM** (Structural Similarity, 0–1) are *full-reference* metrics:
  they need a **true** high-resolution image for comparison.
* For an arbitrary uploaded image no such reference exists, so the app shows **“Not available for arbitrary uploaded
  image”** instead of fake numbers.
* Valid metrics are shown in two controlled situations:
  1. **Demo images** that ship with a true high-res reference (generated locally).
  2. **Controlled experiment tab**: treat an image as ground truth → shrink → reconstruct with bicubic and AI → compare.
* Note: Real-ESRGAN optimises for perceptual sharpness, so PSNR/SSIM can even favour bicubic although the AI image may look
  better. Metrics and human perception can disagree.

## 16. Limitations
* The model **estimates** a higher-resolution reconstruction using patterns learned during training — it does **not**
  recover the exact original details.
* Extremely blurry or tiny images may not recover lost information.
* Results depend on image content; super-resolution may occasionally introduce artifacts or invent plausible but wrong detail.
* A **pretrained** model is used rather than training from scratch.
* On CPU, large images are slow; the app caps input size (2×: 800 px, 4×: 320 px on the long side) and warns the user.
* Educational prototype — **not** a professional or forensic restoration system.

## 17. Future Scope
* GPU acceleration and batch processing.
* Face-specific enhancement (e.g., GFPGAN) and denoising/deblurring options.
* Video super-resolution.
* Fine-tuning the model on a domain dataset (medical, satellite, documents).
* Additional perceptual metrics (LPIPS) and user-study evaluation.
* Model comparison (SRCNN, EDSR, SwinIR) inside the app.

## 18. Team Contribution

| Member | Roll No. | Contribution |
|---|---|---|
| _Name 1_ | | Model integration & inference pipeline |
| _Name 2_ | | Streamlit UI & comparison features |
| _Name 3_ | | Testing, documentation, README |
| _Name 4_ | | Presentation & viva preparation |

*(Replace with your actual team details.)*

## 19. Screenshots
_Add your screenshots here after running the app:_

1. `docs/screenshot_upload.png` — upload screen
2. `docs/screenshot_sidebyside.png` — original vs. enhanced
3. `docs/screenshot_slider.png` — before/after slider
4. `docs/screenshot_bicubic_vs_ai.png` — bicubic vs. AI

## 20. Disclaimer
RE-SHARP is an **educational prototype**. Outputs are AI estimates and must not be used as evidence or for forensic,
medical or legal purposes. Pretrained weights belong to the Real-ESRGAN authors (BSD-3-Clause):
<https://github.com/xinntao/Real-ESRGAN>. Please cite: *Wang et al., “Real-ESRGAN: Training Real-World Blind
Super-Resolution with Pure Synthetic Data”, 2021.*

---

## Project structure
```
re_sharp/
├── app.py                      # Streamlit UI
├── requirements.txt
├── README.md
├── PRESENTATION_CONTENT.md     # 6 slides
├── VIVA_QUESTIONS.md           # 25 questions & answers
├── DEMO_GUIDE.md               # demo script, errors & fixes, do-not-change list
├── check_setup.py              # 5-minute self-test
├── download_weights.py         # pre-download model weights
├── make_demo_images.py         # regenerate demo images
├── .gitignore
├── .streamlit/config.toml
├── models/
│   ├── rrdbnet.py              # CNN architecture (Real-ESRGAN generator)
│   └── model_manager.py        # weight URLs, auto-download, loading
├── utils/
│   ├── image_utils.py          # validation, info, encoding, slider
│   ├── super_resolution.py     # tiled CPU inference, bicubic baseline
│   ├── metrics.py              # PSNR / SSIM + controlled experiment
│   └── demo_images.py          # local demo image generator
├── assets/demo_images/         # generated demo inputs + references
├── weights/                    # pretrained weights (auto-downloaded)
└── outputs/                    # enhanced images are also saved here
```
