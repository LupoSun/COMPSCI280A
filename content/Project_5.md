---
title: "Project 5: Diffusion Models"
tags:
  - CS180
  - Proj5
  - ComputerVision
  - DiffusionModels
  - GenerativeAI
date: 2025-11-25
permalink: /cs180/proj5/
draft: false
---

> 📌 **TL;DR**  
> This project explores the power of diffusion models for image generation and manipulation. I implement diffusion sampling loops, create optical illusions with visual anagrams, generate hybrid images, and perform various image editing tasks like inpainting and SDEdit. The journey goes from understanding the forward diffusion process to implementing classifier-free guidance for high-quality generation!

---

# Part 0: Setup

## Accessing DeepFloyd

**Model.** Using DeepFloyd IF, a two-stage text-to-image diffusion model by Stability AI:
- Stage 1: Generates 64×64 images
- Stage 2: Upsamples to 256×256

**Text prompt encoding.** Using provided T5 encoder clusters to generate prompt embeddings (4096-dimensional vectors).

---

## Part 0: Playing with Pre-trained Models

**Goal.** Generate images from custom text prompts to understand the model's capabilities.

**My text prompts:**
1. A abstract painting of deconstructivism
2. a photo a building in the style of deconstructionism
3. a image of the theme LUX
4. a photo of a firework show with the text "TAO" in the sky
5. a photo of a cat wearing a cape with text "CS280A" on it
6. a picture of building by  Zaha Hadid
7. an apple hybridized with an orange
8. EUSEXUA
9. a tree full of cats
10. Saturn merging with Mercury
11. a building designed by Mies van der Rohr and Zaha Hadid
12. a planet of boba tea
13. a car with low-poly design
14. an acrylic painting of a golden toilet
15. an acrylic painting of  concert
16. an impressionist painting of a pond
17. an expressionist painting of a people eating hotpot
18. a high quality photo
19. Mariah Carey
20. Jennifer Lopez
21. Kanye West
22. Taylor Swift
23. Einstein
24. Camera
25. Icon of a picture
26. a logo with text "Cam" on it

### Generated Images
**Random seed used:** 42
I first generated 14 images using text embedding's randomly picked from my dictionary.
![[0.1.png]]

And then I did some experiment generating images using different prompt and steps.
![[0.2.png]]
The three prompts I tested demonstrate both the capabilities and limitations of the DeepFloyd diffusion model:

1. **"Fireworks display with TAO text"** - The model successfully captured the festive atmosphere with colorful bursts in the night sky, though the text rendering varies significantly across inference steps. At 10 steps, the text is blocky and barely legible. By 30 steps, it becomes more stylized and integrated with the fireworks. At 50 steps, the image achieves the highest quality with well-defined fireworks and clearer text.
2. **"A cat wearing a basketball jersey"** - This prompt shows impressive coherence across all step counts. The model consistently interprets the clothing request, generating a cat in blue athletic wear. The progressive refinement is evident: 10 steps produces a rough but recognizable result, 30 steps adds detail to the fur texture and clothing folds, and 50 steps delivers photorealistic quality with proper lighting and sharp details.
3. **"Saturn with its rings in deep space"** - Here we see a clear failure mode at low inference steps. The 10-step result is essentially a blank blue gradient, showing the model hasn't had enough iterations to form coherent structures. At 30 steps, Saturn's iconic rings begin to emerge with correct geometry. By 50 steps, the planet and rings are well-defined with appropriate cosmic coloring.

**Key observation:** The `num_inference_steps` parameter dramatically affects output quality. Complex prompts requiring precise spatial relationships (like text or planetary features) particularly benefit from higher step counts, while simpler subjects (like the cat) converge faster.

---

# Part 1: Sampling Loops

## 1.1 — Implementing the Forward Process

**Goal.** Implement the forward diffusion process that adds noise to clean images.

**The forward process equation:**

$$q(x_t | x_0) = \mathcal{N}(x_t ; \sqrt{\bar\alpha_t} x_0, (1 - \bar\alpha_t)\mathbf{I})$$

Equivalent to:

$$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon \quad \text{where}~ \epsilon \sim \mathcal{N}(0, 1)$$

### Results: Noisy Campanile at Different Timesteps
![[1.1.png]]

---

## 1.2 — Classical Denoising

**Goal.** Attempt to denoise using Gaussian blur filtering (spoiler: it doesn't work well).

**Method:** Applied `torchvision.transforms.functional.gaussian_blur` with various kernel sizes and sigmas.

### Results: Gaussian Blur Denoising
![[1.2.png]]

---

## 1.3 — One-Step Denoising

**Goal.** Use the pretrained UNet to estimate and remove noise in a single step.

**Process:**
1. Add noise to Campanile using `forward()` function
2. Pass through `stage_1.unet` to estimate noise
3. Remove estimated noise to recover clean image

**Noise removal formula:**

$$x_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t} \epsilon}{\sqrt{\bar\alpha_t}}$$

### Results: One-Step Denoising
![[1.3.png]]

---

## 1.4 — Iterative Denoising

**Goal.** Implement the full iterative denoising loop with strided timesteps.

**Denoising equation:**

$$x_{t'} = \frac{\sqrt{\bar\alpha_{t'}}\beta_t}{1 - \bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t'})}{1 - \bar\alpha_t} x_t + v_\sigma$$

**Timestep schedule:** Starting at t=990, stride of 30, ending at t=0

### Results: Iterative Denoising Process
![[1.4_steps.png]]

### Comparison: Different Denoising Methods

![[1.4_comparison.png]]
*Left to right: Original, Noisy, Iterative, One-step, Gaussian blur*

---

## 1.5 — Diffusion Model Sampling

**Goal.** Generate images from pure noise using `iterative_denoise` with `i_start=0`.

**Prompt:** "a high quality photo"

### Generated Samples
![[1.5.png]]

---

## 1.6 — Classifier-Free Guidance (CFG)

**Goal.** Improve generation quality using CFG by combining conditional and unconditional noise estimates.

**CFG formula:**

$$\epsilon = \epsilon_u + \gamma (\epsilon_c - \epsilon_u)$$

**CFG scale:** γ = 7

**Prompt:** "a high quality photo"

### Generated Samples with CFG
![[1.6.png]]

---

## 1.7 — Image-to-Image Translation (SDEdit)

**Goal.** Edit real images by adding noise and denoising with different starting points.

### 1.7.0 — Editing the Campanile

**Prompt:** "a high quality photo"

![[1.7.png]]

---

### 1.7.1 — Editing Hand-Drawn and Web Images

#### Web Image: Evangelion

![[1.7.1.1.png]]

---
#### Hand-Drawn Image 1:  Mastercard inspired color

![[1.7.1.2.png]]

---

#### Hand-Drawn Image 2: Thin Shiba

![[1.7.1.3.png]]

---

### 1.7.2 — Inpainting

**Goal.** Fill in masked regions using RePaint algorithm.

**Method:** At each denoising step, replace unmasked regions with noised original image:

$$x_t \leftarrow \mathbf{m} x_t + (1 - \mathbf{m}) \text{forward}(x_{orig}, t)$$

#### Campanile Inpainting

![[1.7.2.1.png]]

---

#### Custom Inpainting Example 1: Wurster Hall

![[1.7.2.2.png]]

---

#### Custom Inpainting Example 2: Last Supper

![[1.7.2.3.png]]

---

### 1.7.3 — Text-Conditional Image-to-Image Translation

**Goal.** Guide SDEdit with custom text prompts instead of "a high quality photo".

#### Campanile → 'a tree full of cats'

![[1.7.3.1.png]]

---

#### Last Supper → 'A abstract painting of deconstructivism'

![[1.7.3.2.png]]

---

#### Bauer Wurster Hall → 'a picture of building by Zaha Hadid'

![[1.7.3.3.png]]

---

## 1.8 — Visual Anagrams

**Goal.** Create optical illusions that reveal different images when flipped upside down.

**Method:** Average noise estimates from upright and flipped images:

$$\epsilon_1 = \text{CFG-UNet}(x_t, t, p_1)$$
$$\epsilon_2 = \text{flip}(\text{CFG-UNet}(\text{flip}(x_t), t, p_2))$$
$$\epsilon = (\epsilon_1 + \epsilon_2) / 2$$

### Visual Anagram 1
![[1.8.1.png]]

---

### Visual Anagram 2
![[1.8.2.png]]

---

## 1.9 — Hybrid Images

**Goal.** Create hybrid images by combining low and high frequencies from different prompts.

**Method:** Composite low-pass and high-pass filtered noise estimates:

$$\epsilon_1 = \text{CFG-UNet}(x_t, t, p_1)$$
$$\epsilon_2 = \text{CFG-UNet}(x_t, t, p_2)$$
$$\epsilon = f_{\text{lowpass}}(\epsilon_1) + f_{\text{highpass}}(\epsilon_2)$$

**Filter parameters:** Gaussian blur with kernel size 33, σ = 2

### Hybrid Image 1

![[1.9.3.png]]

---

### Hybrid Image 2

![[1.9.2.png]]

---
### Hybrid Image 3

![[1.9.1.png]]

---
# Part 2: Bells & Whistles

## 2.1 Triple View Anagram

**What I implemented:** Extended the visual anagram technique to create images that reveal three different scenes depending on the viewing angle. Instead of just upright/flipped, this creates a "triple anagram" that shows different content when:

1. Viewed normally (0°)
2. Rotated 90° clockwise
3. Rotated 180° (upside down)

**Method:** Average three noise estimates with different rotations:

$$\epsilon_1 = \text{CFG-UNet}(x_t, t, p_1)$$
$$\epsilon_2 = \text{rot}_{90}(\text{CFG-UNet}(\text{rot}_{-90}(x_t), t, p_2))$$
$$\epsilon_3 = \text{rot}_{180}(\text{CFG-UNet}(\text{rot}_{180}(x_t), t, p_3))$$
$$\epsilon = (\epsilon_1 + \epsilon_2 + \epsilon_3) / 3$$

**Results:**
![[BW.1.1.png]]
![[BW.1.2.png]]
![[BW.1.3.png]]

---
## 2.2 Logo Design: Image-to-image Translation
![[BW.2.1.png]]
![[BW.2.2.png]]
![[BW.2.3.png]]
The prompt I used was "Camera". Camera is important in this class and it does have two letters in common with the "Cal" logo. I first run `forward()` on the *Cal* logo to add noise, then use the `iterative_denoise_cfg()` on multiple noise levels to get a variety of options.  The ones marked with orange are interesting.


