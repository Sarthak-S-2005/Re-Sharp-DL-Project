# RE-SHARP — Demo & Troubleshooting Guide

## A. Setup (Windows) — exact commands

```bat
cd re_sharp
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python download_weights.py 2
python check_setup.py
streamlit run app.py
```

`check_setup.py` should end with **ALL GOOD**. If `download_weights.py` fails, download the weights manually
(see README section 12) and put them in the `weights` folder.

## B. Test it in 5 minutes
1. `streamlit run app.py` → the browser opens.
2. Select the demo image **City skyline** → **Enhance Image** (about 5–30 s on CPU).
3. Check: original 360 × 240 → enhanced 720 × 480, scale 2×, processing time shown.
4. Open each tab: *Side by side*, *Before / After slider* (drag it), *Bicubic vs AI* (zoom 4×), *Details & metrics*.
5. Click **DOWNLOAD ENHANCED IMAGE** → open the file and confirm it is 720 × 480.
6. Upload a **.txt file** renamed or a random file → friendly error appears.
7. Click **Process another image** in the sidebar → upload your own photo → enhance again.
8. Optional: *Controlled experiment* tab → Run.

## C. How to demonstrate in front of your professor (≈ 5 min)
1. **(30 s) Idea.** "Interpolation guesses pixels with a formula; our CNN learned patterns from data."
2. **(30 s) Upload.** Choose demo **Text poster** (thin text and lines show the difference clearly). Point out info table.
3. **(30 s) Enhance.** Press *Enhance Image*; mention the progress bar and CPU inference.
4. **(60 s) Before/After.** Show *Side by side* (resolution, file size, format), then drag the slider.
5. **(60 s) Bicubic vs AI.** Zoom to 4× on text/windows. Say: "We don't claim AI is always better — judge the pixels."
6. **(30 s) Metrics.** Open *Details & metrics*: "PSNR/SSIM need a true reference, so for uploaded images we show *Not available*.
   For this demo we do have the true high-res reference, so these numbers are valid."
7. **(30 s) Download.** Click download, show the file.
8. **(30 s) Architecture.** Show README diagram: conv → 23 RRDB → upsample → reconstruct. Mention: pretrained, not an autoencoder.
9. **Limitations.** Heavily blurry images can't be recovered; model estimates detail; artifacts possible.

**Before the demo:** run once so weights are downloaded and the app is warmed up. Have 2–3 of your own photos
(small, ≤ 500 px) ready. Take backup screenshots in case the laptop is slow.

## D. Simple explanation of the whole project
RE-SHARP lets a user upload a small or blurry picture. The app first checks the file, then sends the picture through a
pretrained convolutional neural network (Real-ESRGAN). The first layers extract features such as edges and textures;
the later layers enlarge those features and rebuild a bigger RGB image. The app then shows the original and the AI
result together, lets you drag a slider, compares with ordinary bicubic enlargement, and lets you download the result.
We did not train the network ourselves; we use its published weights, and we are honest that the output is an
estimate, not the true original.

## E. 1-minute explanation to memorize
"Our project is RE-SHARP, an AI image super-resolution app. When you enlarge a low-resolution image normally, for example
with bicubic interpolation, the computer just averages nearby pixels, so the picture becomes soft. We use a pretrained
deep learning model, Real-ESRGAN, which is a convolutional neural network with 23 residual dense blocks. The
convolution layers extract features like edges and textures, and the upsampling layers reconstruct a higher-resolution
image from those features. The user uploads an image, presses Enhance, and sees the original and the enhanced image side by
side, with a before/after slider, resolution, file size and processing time. We also compare it with bicubic upscaling
and let the user download the result. We do not claim the AI recovers the exact original: it estimates detail from
patterns it learned, may add artifacts, and cannot fix extremely blurry images. PSNR and SSIM are shown only where a real
reference exists. It runs on a normal laptop CPU."

## F. Viva questions
See `VIVA_QUESTIONS.md` (25 questions and answers).

## G. Common errors and exact fixes

| Problem | Fix |
|---|---|
| `'streamlit' is not recognized` | The virtual environment is not active. Run `venv\Scripts\activate` then `streamlit run app.py`. |
| `python is not recognized` | Install Python 3.10–3.12 from python.org and tick **Add Python to PATH**; reopen the terminal. |
| `ModuleNotFoundError: No module named 'torch'` (or any module) | `venv\Scripts\activate` then `pip install -r requirements.txt`. |
| `pip install torch` very slow / fails | Run `pip install torch --index-url https://download.pytorch.org/whl/cpu` then `pip install -r requirements.txt`. |
| *"Could not download the Real-ESRGAN weights"* | Firewall/Wi-Fi blocks GitHub. Download the `.pth` file in a browser (links in README §12) and save it in `weights\` with the exact file name. Or use a mobile hotspot and run `python download_weights.py 2`. |
| *"weights file looks corrupted or incompatible"* | Delete the file in `weights\` and run `python download_weights.py 2` again. |
| App is very slow | Use **2×**, a smaller image (≤ 400 px), and close other programs. First run also loads the model (a few seconds). 4× is much slower. |
| *Out of memory* | Use 2× and a smaller image. |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Slider shows blank | Reload the page (F5) and press *Enhance Image* again. |
| Browser doesn't open | Open <http://localhost:8501> manually. |
| `Pillow` / `numpy` version conflict | `pip install --upgrade pillow numpy` |
| Upload rejected | Only JPG / PNG / WEBP up to 25 MB, at least 16 × 16 px, not corrupted. |

## H. What NOT to change at the last minute
- Do **not** upgrade or reinstall `torch`, `streamlit`, `numpy`, or `Pillow` on the day of the demo.
- Do **not** edit `models/rrdbnet.py` — layer names must match the pretrained weights exactly or loading fails.
- Do **not** rename or delete files in `weights\` or change `MODEL_SPECS` URLs/file names.
- Do **not** change `tile`, `tile_pad` (they must stay even numbers) or `MAX_PROCESS_SIDE` unless you test again.
- Do **not** switch Python versions or recreate the `venv` right before presenting.
- Do **not** rely on live internet in the demo: run `python download_weights.py` beforehand.
- Do **not** claim the AI "restores the exact original" or that PSNR/SSIM prove quality on uploaded images.
- Safe changes: team names in README, screenshots, colours/text in `app.py` CSS (re-run and test after editing).
