---
title: "Project 1: Images of the Russian Empire"
tags:
  - CS280A
  - Proj0
  - Computer Vision
  - Photography
date: 2025-09-10
permalink: /cs280A/proj1/
draft: false
---
## Part 1 — Overview

This project reconstructs color photographs from Prokudin-Gorskii glass plate scans by aligning three monochrome channels (B, G, R).  
I started with a simple brute-force search, quickly ran into speed and accuracy walls, then built a coarse-to-fine image pyramid that made alignment fast and reliable.  
Finally, I layered on a few automatic enhancements (“bells & whistles”) to automatically detect borders, clean up tonal range, neutralize color casts, and nudge the colors closer to what we expect today.

**tl;dr**: Pyramid alignment is much faster. Post-processing makes the results pop. Edge-based alignment for the win.

---

## Part 2 — Implementation

### 1. Basic Brute Force Alignment

**What I tried:**  

For each moving channel (G and R), I searched integer shifts in a fixed window around the fixed channel (B), scoring each candidate with SSD / ZNCC on an inner crop to avoid border artifacts.

The **Sum of Squared Differences (SSD)** score is:
$$
\text{SSD}(A,B) = \sum_{x,y} \left(A(x,y) - B(x,y)\right)^2
$$
The **Zero-mean Normalized Cross-Correlation (ZNCC)** score is:
$$
\text{ZNCC}(A,B) = \frac{\langle A - \bar{A},\; B - \bar{B} \rangle}{\|A - \bar{A}\|\;\|B - \bar{B}\|}
$$
![[SS_Cathedral.jpg]]

![[SS_Moastery.jpg]]

![[SS_Tobolsk.jpg]]

**What happened:**  
- It works on small images… until it doesn’t.  
- The search space grows quadratically with the shift range; it gets slow fast.  
- Borders and exposure differences can confuse raw-pixel SSD.

**A memorable failure:**  
![[emir_r59,42_g48,24_27m24s.jpg]]
This implementation worked with the small cathedral image, but the mighty Emir took **27m24s** (yes, really) and still wasn’t well aligned. That pushed me to build the pyramid.

---

### 2. Image Pyramid Alignment

**Idea in a sentence:**  
Coarsest level: do a big search cheaply. Then, as it es to finer levels, **guess** the shift (×2 from the coarser level) and only **refine** in a small local window. Same metric (ZNCC/SSD), way less work.
$$
(d_x^{(L-1)}, d_y^{(L-1)}) \approx 2 \cdot (d_x^{(L)}, d_y^{(L)})
$$

**Practical choices:**  
- **Metric:** ZNCC (more robust to exposure/contrast differences).  
- **Guard cropping:** remove estimated border + small guard to avoid `np.roll` wrap-around.  
- **Parameters:** `max_shift` (coarse), `refine_window` (fine), and a variance-based border estimate.

---

### 3. Bells & Whistles

#### 1. Automatic cropping
**Goal:** My implementation of automatic cropping is actually a detecting function that does not actually crop the image but serve as a pre-processing step to dynamically inform the alignment algorithm. 

**How:**  
- For each channel, I compute the **variance** of pixel intensities across rows and columns:
$$

\sigma^2_{\text{row}}(i) = \mathrm{Var}(I_{i,:}), \quad 

\sigma^2_{\text{col}}(j) = \mathrm{Var}(I_{:,j})

$$
- Rows or columns with variance below a small threshold are considered “flat” (likely border).  
- The first and last rows/columns that cross the threshold mark the estimated content bounds. 
- I expand slightly with a margin to avoid cutting into the image.  

#### 2. Automatic contrasting
**Goal:** expand tonal range so images don’t look flat.  
**How:** For a channel $C$, with lower percentile $p_l$ and upper percentile $p_h$:
$$
C' = \frac{C - p_l}{p_h - p_l}
$$
We can see crisper midtones, better separation in shadows/highlights on the right.
![[BnW_AutoContrast.jpg]]
(left before; right after)
#### 3. Automatic white balance
**Goal:** neutral gray overall.  
**How:** If $\mu_r, \mu_g, \mu_b$ are channel means and $\mu = (\mu_r+\mu_g+\mu_b)/3$, then:
$$

R' = R \cdot \frac{\mu}{\mu_r}, \quad

G' = G \cdot \frac{\mu}{\mu_g}, \quad

B' = B \cdot \frac{\mu}{\mu_b}

$$
On the right the white balance has been adjusted to be not too warm.
![[BnW_AutoWhiteBalance.jpg]]
(left before; right after)
#### 4. Better color mapping
**Goal:** P-G spectral bands ≠ modern sRGB.  
**How:** Adjust hues using a 3×3 linear color transform.
$$

\begin{bmatrix} R' \\ G' \\ B' \end{bmatrix}

=

\begin{bmatrix}

1.15 & -0.05 & -0.10 \\

-0.05 & 1.10 & -0.05 \\

-0.10 & -0.05 & 1.15

\end{bmatrix}

\cdot

\begin{bmatrix} R \\ G \\ B \end{bmatrix}

$$
 We can see a slightly more refined color visualization on the right.
 ![[BnW_AutoColorMapping.jpg]]
(left before; right after)
#### 5. Better Feature: Edge-based alignment
**Goal:** make alignment more robust when intensities differ across channels.  
**How:** Use features less sensitive to intensity shifts.  
The Sobel gradient magnitude is:
$$

M(x,y) = \sqrt{(I * S_x)^2 + (I * S_y)^2}

$$
where $S_x, S_y$ are the Sobel filters in horizontal and vertical directions.  
The pyramid alignment is then performed on $M$ instead of raw pixels, and the found shifts are applied to the original images.
In this implementation I didn't reinvent the wheel and `sobal` form `scipy.ndimage` instead.

**Result:** The most accurate alignment in this homework yet

---

## Part 3 — Gallery

Each example shows:

- **Before alignment** — naive stack (R,G,B) → heavy fringing and mis-registration.  
- **After** — pyramid alignment (automatic cropping used as pre-processing).  
- **Bells & whistles** — the post-processing trio (white balance → contrast → optional color matrix), using edge-based alignment.

The images that are beaming **neon green** light are the additional images of my choice from the Prokudin-Gorskii collection.


<div style="display: flex; flex-direction: column; gap: 2rem;">

  <div style="text-align: center;">
    <p><strong>Emir</strong></p>
    <img src="emir_result.jpg" alt="Emir" />
  </div>

  <div style="text-align: center;">
    <p><strong>Cathedral</strong></p>
    <img src="cathedral_result.jpg" alt="cathedral_result" />
  </div>

  <div style="text-align: center;">
    <p><strong>Church</strong></p>
    <img src="church_result.jpg" alt="church_result" />
  </div>

  <div style="text-align: center;">
    <p><strong>master-pnp-prok-00100-00110u</strong></p>
    <img src="EXTRA_master-pnp-prok-00100-00110u_result.jpg" alt="00110u"
         style="border: 4px solid #39FF14;
                box-shadow: 0 0 10px #39FF14,
                            0 0 20px #39FF14,
                            0 0 40px #39FF14;" />
  </div>

  <div style="text-align: center;">
    <p><strong>master-pnp-prok-00100-00176u</strong></p>
    <img src="EXTRA_master-pnp-prok-00100-00176u_result.jpg" alt="00176u"
         style="border: 4px solid #39FF14;
                box-shadow: 0 0 10px #39FF14,
                            0 0 20px #39FF14,
                            0 0 40px #39FF14;" />
  </div>

  <div style="text-align: center;">
    <p><strong>master-pnp-prok-00100-00189u</strong></p>
    <img src="EXTRA_master-pnp-prok-00100-00189u_result.jpg" alt="00189u"
         style="border: 4px solid #39FF14;
                box-shadow: 0 0 10px #39FF14,
                            0 0 20px #39FF14,
                            0 0 40px #39FF14;" />
  </div>

  <div style="text-align: center;">
    <p><strong>master-pnp-prok-00200-00212u</strong></p>
    <img src="EXTRA_master-pnp-prok-00200-00212u_result.jpg" alt="00212u"
         style="border: 4px solid #39FF14;
                box-shadow: 0 0 10px #39FF14,
                            0 0 20px #39FF14,
                            0 0 40px #39FF14;" />
  </div>

  <div style="text-align: center;">
    <p><strong>master-pnp-prok-00200-00227u</strong></p>
    <img src="EXTRA_master-pnp-prok-00200-00227u_result.jpg" alt="00227u"
         style="border: 4px solid #39FF14;
                box-shadow: 0 0 10px #39FF14,
                            0 0 20px #39FF14,
                            0 0 40px #39FF14;" />
  </div>

  <div style="text-align: center;">
    <p><strong>Harvesters</strong></p>
    <img src="harvesters_result.jpg" alt="harvesters_result" />
  </div>

  <div style="text-align: center;">
    <p><strong>Icon</strong></p>
    <img src="icon_result.jpg" alt="icon_result" />
  </div>

  <div style="text-align: center;">
    <p><strong>Italil</strong></p>
    <img src="italil_result.jpg" alt="italil_result" />
  </div>

  <div style="text-align: center;">
    <p><strong>Lastochikino</strong></p>
    <img src="lastochikino_result.jpg" alt="lastochikino_result" />
  </div>

  <div style="text-align: center;">
    <p><strong>Lugano</strong></p>
    <img src="lugano_result.jpg" alt="lugano_result" />
  </div>

  <div style="text-align: center;">
    <p><strong>Melons</strong></p>
    <img src="melons_result.jpg" alt="melons_result" />
  </div>

  <div style="text-align: center;">
    <p><strong>Monastery</strong></p>
    <img src="monastery_result.jpg" alt="monastery_result" />
  </div>

  <div style="text-align: center;">
    <p><strong>Siren</strong></p>
    <img src="siren_result.jpg" alt="siren_result" />
  </div>

  <div style="text-align: center;">
    <p><strong>Three Generations</strong></p>
    <img src="three_generations_result.jpg" alt="three_generations_result" />
  </div>

  <div style="text-align: center;">
    <p><strong>Tobolsk</strong></p>
    <img src="tobolsk_result.jpg" alt="tobolsk_result" />
  </div>

  <div style="text-align: center;">
    <p><strong>master-pnp-prok-00200-00228u</strong></p>
    <img src="EXTRA_master-pnp-prok-00200-00228u_result.jpg" alt="00228u"
         style="border: 4px solid #39FF14;
                box-shadow: 0 0 10px #39FF14,
                            0 0 20px #39FF14,
                            0 0 40px #39FF14;" />
  </div>

</div>


  