---
title: "Project 5: Fun with Diffusion Models!"
tags:
  - CS280A
  - Proj5
  - ComputerVision
  - DiffusionModels
  - GenerativeAI
date: 2025-11-25
permalink: /cs180/proj5/
draft: false
---

> 📌 **TL;DR**  
> This project explores the full spectrum of diffusion-based generative models for image generation and manipulation. Part A demonstrates the power of pre-trained diffusion models (DeepFloyd IF) through sampling, editing techniques like SDEdit and inpainting, and creative applications including visual anagrams and hybrid images. Part B takes a deeper dive by implementing a flow matching model from scratch on MNIST, progressing from single-step denoising to time-conditioned and class-conditioned generation with classifier-free guidance. Together, these parts showcase both the practical applications of modern diffusion models and the mathematical foundations that make them work!

---
# Part A: The Power of Diffusion Models!
## Part 0: Setup

### Accessing DeepFloyd

**Model.** Using DeepFloyd IF, a two-stage text-to-image diffusion model by Stability AI:
- Stage 1: Generates 64×64 images
- Stage 2: Upsamples to 256×256

**Text prompt encoding.** Using provided T5 encoder clusters to generate prompt embeddings (4096-dimensional vectors).

> [!tip] My text prompts:
>1. A abstract painting of deconstructivism
> 2. a photo a building in the style of deconstructionism
>3. a image of the theme LUX
>4. a photo of a firework show with the text "TAO" in the sky
>5. a photo of a cat wearing a cape with text "CS280A" on it
>6. a picture of building by  Zaha Hadid
>7. an apple hybridized with an orange
>8. EUSEXUA
>9. a tree full of cats
>10. Saturn merging with Mercury
>11. a building designed by Mies van der Rohr and Zaha Hadid
>12. a planet of boba tea
>13. a car with low-poly design
>14. an acrylic painting of a golden toilet
>15. an acrylic painting of  concert
>16. an impressionist painting of a pond
>17. an expressionist painting of a people eating hotpot
>18. a high quality photo
>19. Mariah Carey
>20. Jennifer Lopez
>21. Kanye West
>22. Taylor Swift
>23. Einstein
>24. Camera
>25. Icon of a picture
>26. a logo with text "Cam" on it


---

### Part 0: Playing with Pre-trained Models

**Goal.** Generate images from custom text prompts to understand the model's capabilities.
#### Generated Images
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

## Part 1: Sampling Loops

### 1.1 — Implementing the Forward Process

**Goal.** Implement the forward diffusion process that adds noise to clean images.

**The forward process equation:**

$$q(x_t | x_0) = \mathcal{N}(x_t ; \sqrt{\bar\alpha_t} x_0, (1 - \bar\alpha_t)\mathbf{I})$$

Equivalent to:

$$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon \quad \text{where}~ \epsilon \sim \mathcal{N}(0, 1)$$

### Results: Noisy Campanile at Different Timesteps
![[1.1.png]]

---

### 1.2 — Classical Denoising

**Goal.** Attempt to denoise using Gaussian blur filtering (spoiler: it doesn't work well).

**Method:** Applied `torchvision.transforms.functional.gaussian_blur` with various kernel sizes and sigmas.

### Results: Gaussian Blur Denoising
![[1.2.png]]

---

### 1.3 — One-Step Denoising

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

### 1.4 — Iterative Denoising

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

### 1.5 — Diffusion Model Sampling

**Goal.** Generate images from pure noise using `iterative_denoise` with `i_start=0`.

**Prompt:** "a high quality photo"

### Generated Samples
![[1.5.png]]

---

### 1.6 — Classifier-Free Guidance (CFG)

**Goal.** Improve generation quality using CFG by combining conditional and unconditional noise estimates.

**CFG formula:**

$$\epsilon = \epsilon_u + \gamma (\epsilon_c - \epsilon_u)$$

**CFG scale:** γ = 7

**Prompt:** "a high quality photo"

### Generated Samples with CFG
![[1.6.png]]

---

### 1.7 — Image-to-Image Translation (SDEdit)

**Goal.** Edit real images by adding noise and denoising with different starting points.

#### 1.7.0 — Editing the Campanile

**Prompt:** "a high quality photo"

![[1.7.png]]

---

#### 1.7.1 — Editing Hand-Drawn and Web Images

##### Web Image: Evangelion

![[1.7.1.1.png]]

---
##### Hand-Drawn Image 1:  Mastercard inspired color

![[1.7.1.2.png]]

---

##### Hand-Drawn Image 2: Thin Shiba

![[1.7.1.3.png]]

---

#### 1.7.2 — Inpainting

**Goal.** Fill in masked regions using RePaint algorithm.

**Method:** At each denoising step, replace unmasked regions with noised original image:

$$x_t \leftarrow \mathbf{m} x_t + (1 - \mathbf{m}) \text{forward}(x_{orig}, t)$$

##### Campanile Inpainting

![[1.7.2.1.png]]

---

##### Custom Inpainting Example 1: Wurster Hall

![[1.7.2.2.png]]

---

##### Custom Inpainting Example 2: Last Supper

![[1.7.2.3.png]]

---

#### 1.7.3 — Text-Conditional Image-to-Image Translation

**Goal.** Guide SDEdit with custom text prompts instead of "a high quality photo".

##### Campanile → 'a tree full of cats'

![[1.7.3.1.png]]

---

##### Last Supper → 'A abstract painting of deconstructivism'

![[1.7.3.2.png]]

---

##### Bauer Wurster Hall → 'a picture of building by Zaha Hadid'

![[1.7.3.3.png]]

---

### 1.8 — Visual Anagrams

**Goal.** Create optical illusions that reveal different images when flipped upside down.

**Method:** Average noise estimates from upright and flipped images:

$$\epsilon_1 = \text{CFG-UNet}(x_t, t, p_1)$$
$$\epsilon_2 = \text{flip}(\text{CFG-UNet}(\text{flip}(x_t), t, p_2))$$
$$\epsilon = (\epsilon_1 + \epsilon_2) / 2$$

#### Visual Anagram 1
![[1.8.1.png]]

---

#### Visual Anagram 2
![[1.8.2.png]]

---

### 1.9 — Hybrid Images

**Goal.** Create hybrid images by combining low and high frequencies from different prompts.

**Method:** Composite low-pass and high-pass filtered noise estimates:

$$\epsilon_1 = \text{CFG-UNet}(x_t, t, p_1)$$
$$\epsilon_2 = \text{CFG-UNet}(x_t, t, p_2)$$
$$\epsilon = f_{\text{lowpass}}(\epsilon_1) + f_{\text{highpass}}(\epsilon_2)$$

**Filter parameters:** Gaussian blur with kernel size 33, σ = 2

#### Hybrid Image 1

![[1.9.3.png]]

---

#### Hybrid Image 2

![[1.9.2.png]]

---
#### Hybrid Image 3

![[1.9.1.png]]

---
## Part 2: Bells & Whistles

### 2.1 Triple View Anagram

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
### 2.2 Logo Design: Image-to-image Translation
![[BW.2.1.png]]
![[BW.2.2.png]]
![[BW.2.3.png]]
The prompt I used was "Camera". Camera is important in this class and it does have two letters in common with the "Cal" logo. I first run `forward()` on the *Cal* logo to add noise, then use the `iterative_denoise_cfg()` on multiple noise levels to get a variety of options.  The ones marked with orange are interesting.

---

# Part B: Flow Matching from Scratch!

## Part 1: Training a Single-Step Denoising UNet

### 1.1 - Implementing the UNet

**Architecture:** Implemented a UNet with downsampling and upsampling blocks with skip connections. The network uses:
- **Conv blocks**: Standard convolution with BatchNorm and GELU activation
- **DownConv**: Downsampling by factor of 2
- **UpConv**: Upsampling by factor of 2  
- **Skip connections**: Concatenating features from encoder to decoder

**Hidden dimension:** D = 128

---

### 1.2 - Using the UNet to Train a Denoiser

**Training objective:** Learn to denoise images corrupted with Gaussian noise

$$L = \mathbb{E}_{z,x} \|D_{\theta}(z) - x\|^2$$

where $z = x + \sigma \epsilon$ and $\epsilon \sim \mathcal{N}(0, I)$

Visualization of Noising Process

**Different noise levels:** σ = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

![[1.2_noising_visualization.png]]

---

### 1.2.1 - Training

**Hyperparameters:**
- Noise level: σ = 0.5
- Batch size: 256
- Learning rate: 1e-4
- Optimizer: Adam
- Epochs: 5

#### Training Loss Curve

![[1.2.1_training_loss_curve.png]]

#### Denoising Results After Training

**After 1st epoch:**

![[1.2.1_denoising_epoch_1.png]]

**After 5th epoch:**

![[1.2.1_denoising_epoch_5.png]]

---

### 1.2.2 - Out-of-Distribution Testing

**Test:** Apply the denoiser trained on σ=0.5 to images with different noise levels

**Noise levels tested:** σ = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

#### Out-of-Distribution Denoising Results

![[1.2.2_ood_denoising_results.png]]

---

### 1.2.3 - Denoising Pure Noise

**Goal:** Train denoiser to map pure Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$ to clean MNIST digits

#### Training Loss Curve for Pure Noise

![[1.2.3_training_loss_curve.png]]

#### Results Denoising Pure Noise

**After 1st epoch:**

![[1.2.3_denoising_epoch_1.png]]

**After 5th epoch:**

![[1.2.3_denoising_epoch_5.png]]

**Observations and explanations:**
Seeing averaged/blurry digits that represent the mean of the training data. With MSE loss, the model learns to predict the centroid/average of all possible training examples, which minimizes squared distance but doesn't generate realistic individual samples. The model is learning the expectation over the training distribution rather than sampling from it.

---

## Part 2: Training a Flow Matching Model

### 2.1 - Adding Time Conditioning to UNet

**Time embedding:** Implemented FCBlock (fully-connected blocks) to inject scalar timestep $t \in [0,1]$ into the UNet

**Architecture modification:**
- Two FCBlock modules for time conditioning
- Modulation of features at two points in the decoder path
- Time normalization to $[0,1]$ range

**Forward process:**

$$x_t = (1-t)x_0 + tx_1$$

**Flow prediction:**

$$u(x_t, t) = \frac{d}{dt} x_t = x_1 - x_0$$

---

### 2.2 - Training the Time-Conditioned UNet

**Flow matching objective:**

$$L = \mathbb{E}_{x_0 \sim p_0, x_1 \sim p_1, t \sim U[0,1]} \|(x_1-x_0) - u_\theta(x_t, t)\|^2$$

**Hyperparameters:**
- Hidden dimension: D = 64
- Batch size: 64
- Initial learning rate: 1e-2
- Learning rate schedule: Exponential decay with γ = $0.1^{1/\text{num\_epochs}}$
- Optimizer: Adam

#### Training Loss Curve

![[2.3_training_loss_curve.png]]

---

### 2.3 - Sampling from the Time-Conditioned UNet

**Sampling algorithm:** Iterative denoising starting from pure noise with strided timesteps

**Number of timesteps:** 50

#### Deliverable: Generation Results

**After 1 epoch:**
![[2.3_time_fm_samples_epoch_1.png]]

**After 5 epochs:**
![[2.3_time_fm_samples_epoch_5.png]]

**After 10 epochs:**
![[2.3_time_fm_samples_epoch_10.png]]

---

### 2.4 - Adding Class-Conditioning to UNet

**Class conditioning:** Extended the UNet to accept class labels as one-hot vectors

**Dropout:** 10% unconditional training ($p_{\text{uncond}} = 0.1$) to enable classifier-free guidance

**Architecture:**
- Four FCBlock modules (two for time, two for class)
- Class embeddings combined multiplicatively with time embeddings: `c * features + t`
- One-hot encoding for digit classes 0-9

**Modified forward:**
```python
unflatten = c1 * unflatten + t1
up1 = c2 * up1 + t2
```

---

### 2.5 - Training the Class-Conditioned UNet

**Training objective:** Same as time-only, but with class conditioning and periodic dropout

**Hyperparameters:**
- Hidden dimension: D = 64
- Batch size: 64
- Initial learning rate: 1e-2
- Learning rate schedule: Exponential decay
- Unconditional probability: p_uncond = 0.1
- Epochs: 10

#### Training Loss Curve
![[2.6_training_loss_curve.png]]

---

### 2.6 - Sampling from the Class-Conditioned UNet

**Sampling method:** Classifier-free guidance with scale γ = 5.0

**CFG formula:**

$$\epsilon = \epsilon_u + \gamma (\epsilon_c - \epsilon_u)$$

#### Deliverable: Generation Results with Class Conditioning

**After 1 epoch:**
![[2.6_time_fm_samples_epoch_1.png]]

**After 5 epochs:**
![[2.6_time_fm_samples_epoch_5.png]]

**After 10 epochs:**
![[2.6_time_fm_samples_epoch_10.png]]

---

### 2.6.1 - Training Without Learning Rate Scheduler

**Challenge:** Maintain performance without exponential learning rate decay

> [!tip] My Approach
> In the original implementation I used an exponential learning rate scheduler that decayed the learning rate from 1×10⁻² to 1×10⁻³ over training. For the challenge, I removed the scheduler and instead used a fixed learning rate.
> To compensate for the loss of the scheduler, I chose a constant learning rate of 3×10⁻³, which roughly matches the effective average learning rate of the original schedule (geometric mean of 1×10⁻² and 1×10⁻³), while increasing the dimension of the hidden layers from 64 to 128
> With this fixed learning rate setup, the final training loss and qualitative sample quality are comparable to the scheduler-based run. 
#### Results Without Scheduler
![[2.6.1_time_fm_samples_no_scheduler_1.png]]
![[2.6.1_time_fm_samples_no_scheduler_5.png]]
![[2.6.1_time_fm_samples_no_scheduler_10.png]]

---

## Part 3: Bells & Whistles

### 3.1 - Better Time-Conditioned UNet (CS280A Required)

**Goal:** Improve the time-conditioned only UNet from Part 2.3 to match the quality of the class-conditioned version

#### Improvements Made

> [!tip] My Approach
> To improve the time-conditioned-only UNet, we increased the base number of channels from 64 to 128 (making the network more expressive) and extended training from 10 to 40 epochs with an exponentially decaying learning rate (from 3×10⁻³ down to 3×10⁻⁴). Compared to the original time-only model, the improved model produces sharper digits with fewer artifacts and more consistent shapes, closing much of the gap to the class-conditioned UNet.

#### Improved Results
GIF:
![[BW.gif]]

Frames:
![[BW_time_fm_samples_epoch_1.png]]
![[BW_time_fm_samples_epoch_5.png]]
![[BW_time_fm_samples_epoch_10.png]]
![[BW_time_fm_samples_epoch_15.png]]
![[BW_time_fm_samples_epoch_20.png]]
![[BW_time_fm_samples_epoch_25.png]]
![[BW_time_fm_samples_epoch_30.png]]
![[BW_time_fm_samples_epoch_35.png]]
![[BW_time_fm_samples_epoch_40.png]]
