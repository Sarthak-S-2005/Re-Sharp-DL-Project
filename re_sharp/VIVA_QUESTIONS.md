# RE-SHARP — Viva Questions & Answers (25)

Answers are short and simple so you can say them in your own words.

---

**1. What is super-resolution?**
Creating a higher-resolution image from a low-resolution one. The output has more pixels, and a good model makes them look sharper and more detailed.

**2. What is image restoration?**
Improving a degraded image — removing blur, noise or compression artifacts. Super-resolution is one kind of restoration where we also increase the size.

**3. Why Deep Learning?**
Interpolation uses one fixed formula for every image. A deep network learns from many example images what edges, textures and text should look like, so it can reconstruct sharper detail.

**4. Why a CNN?**
Images have local patterns (edges, corners, textures) that repeat across the image. Convolution layers share the same small filters everywhere, so they detect these patterns efficiently with far fewer parameters than a fully connected network.

**5. What is feature extraction?**
Turning raw pixels into useful descriptions — feature maps showing edges, textures and shapes. Early layers find simple features, deeper layers find complex ones.

**6. What is upscaling?**
Increasing the width and height of an image, e.g., 720×480 → 1440×960 is 2× upscaling.

**7. Difference between interpolation and super-resolution?**
Interpolation (bicubic) calculates new pixels with a fixed formula from neighbours — it cannot add detail. Super-resolution uses a trained network that estimates plausible detail learned from data.

**8. What is a pretrained model?**
A model already trained by someone else on a large dataset. We download its learned weights and use them directly for inference.

**9. Why didn't you train from scratch?**
Training a super-resolution network needs large datasets, many hours of GPU time and tuning. We had about two hours and CPU laptops, so using official pretrained Real-ESRGAN weights was the practical and honest choice.

**10. What is a convolution layer?**
A layer that slides small filters (e.g., 3×3) over the image and computes a weighted sum at every position. The filter weights are learned during training.

**11. What are feature maps?**
The outputs of convolution layers. Each map highlights where a particular pattern (like a vertical edge) appears in the image. Our model uses 64 feature maps per layer.

**12. What is upsampling?**
Increasing the spatial size of feature maps. Our model enlarges them with nearest-neighbour upsampling followed by convolutions, which then refine the details.

**13. What is PSNR?**
Peak Signal-to-Noise Ratio, in decibels. It measures pixel-wise error between an image and a reference — higher means closer to the reference.

**14. What is SSIM?**
Structural Similarity Index, from 0 to 1. It compares luminance, contrast and structure between two images — closer to 1 means more similar.

**15. Why can PSNR/SSIM be misleading without ground truth?**
They need a true high-resolution reference. Without one, you'd be comparing against something invented (or the image against itself), producing meaningless numbers. So for uploaded images we show "Not available" and judge visually. Even with ground truth, a sharper, nicer-looking GAN output can score lower than blurry bicubic.

**16. What are the limitations of your project?**
The model estimates detail rather than recovering the true original; very blurry images cannot be fully restored; results depend on image content; artifacts may appear; CPU inference is slow for large images; we use a pretrained model; it is an educational prototype.

**17. What happens with heavily blurred images?**
The information is already lost, so the model can only guess. It may produce smooth or odd-looking textures or artifacts. Super-resolution works best for mildly degraded images.

**18. Why can AI hallucinate details?**
The network fills in missing detail from patterns it learned in training. Those details are plausible, not guaranteed to be real, so they can differ from the true original (e.g., slightly wrong texture or text shapes).

**19. What is the difference between 2× and 4×?**
2× doubles width and height (4× the pixels); 4× multiplies each by four (16× the pixels). 4× must invent far more detail, is slower, and is more likely to show artifacts. Both are separate pretrained models in our app.

**20. What would you improve in the future?**
GPU support and batch processing, face-specific enhancement, deblurring/denoising, video support, model comparison (EDSR, SwinIR), and perceptual metrics like LPIPS.

**21. Which model exactly are you using?**
Real-ESRGAN (x2plus and x4plus), whose generator is a CNN called RRDBNet with 23 Residual-in-Residual Dense Blocks. We re-implemented the architecture in PyTorch and load the official pretrained weights.

**22. Is this an autoencoder?**
No. An autoencoder compresses then reconstructs its own input. Our model is a feed-forward CNN generator that maps a low-res image to a high-res image. We don't train an autoencoder.

**23. What is a residual connection and why is it used?**
A skip connection that adds the input of a block to its output (`out = F(x) + x`). It lets the network learn only the missing detail and makes very deep networks easier to train.

**24. What is a GAN and how does it relate to your model?**
A Generative Adversarial Network trains a generator against a discriminator that judges realism. Real-ESRGAN was trained this way, so its outputs look sharp and realistic. We only use the trained generator for inference.

**25. How does your app handle large images on a laptop?**
It caps the input size (2×: 800 px, 4×: 320 px long side) and warns the user, processes the image in small overlapping tiles to save memory, and shows a progress bar.
