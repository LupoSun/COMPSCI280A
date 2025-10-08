---
title: "Project 3A — Homographies & Mosaics (CS180/280A)"
date: 2025-10-07
tags:
  - cs180
  - cs280a
  - project3
  - homography
  - panorama
draft: false
---

> 📌 **TL;DR**  
> You’ll find my correspondences, homography estimation, inverse warps (nearest & bilinear), rectification, feathered mosaics, and a cylindrical “bells & whistles” panorama below. All figures are reproducible from the accompanying notebook.

---

## A.1 — Point Correspondences

**How I collected points.** I used a matplotlib click tool to capture matching points across images (and provided a JSON writer):

- Format used:
  ```json
  {"im1_name":"b1","im2_name":"b2","im1Points":[[4653,2343],...], "im2Points":[[3177,2258],...]}
  ```
- Number of correspondences per pair: **≥ 8** (to ensure stability).
- I verified matches visually with numbered overlays.

**Examples (click to enlarge):**

<div class="img-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
  <figure>
    <img src="assets/A1_scene1_matches_A.png" alt="Scene 1 image A with correspondences" />
    <figcaption>Scene 1 — Image A with numbered correspondences.</figcaption>
  </figure>
  <figure>
    <img src="assets/A1_scene1_matches_B.png" alt="Scene 1 image B with correspondences" />
    <figcaption>Scene 1 — Image B with numbered correspondences.</figcaption>
  </figure>
</div>

---

## A.2 — Recover Homographies

We solve for the homography \( H \) (3×3) mapping homogeneous points \( \mathbf{x}' \sim H\,\mathbf{x} \).
Using DLT with least squares, we assemble an overdetermined linear system:

$$
A\,\mathbf{h} = \mathbf{b}, \quad 
\mathbf{h} = \begin{bmatrix} h_{11}&h_{12}&h_{13}&h_{21}&h_{22}&h_{23}&h_{31}&h_{32} \end{bmatrix}^\top,\quad h_{33}=1.
$$

For each correspondence \( (x,y) \leftrightarrow (x',y') \):

$$
\begin{aligned}
x' &= \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + 1},\\
y' &= \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + 1}.
\end{aligned}
$$

This yields two rows in \( A \) and entries in \( \mathbf{b} \). We use Hartley normalization, solve by least squares, then denormalize.

**Reprojection error** (reported per pair):

$$
\text{SSD}(A,B) = \sum_{i=1}^{N} \left\lVert \mathbf{x}'_i - \widehat{\mathbf{x}}'_i \right\rVert_2^2,
\quad
\widehat{\mathbf{x}}'_i \sim H\,\mathbf{x}_i .
$$

| Pair | N pts | mean | median | max |
|---|---:|---:|---:|---:|
| Scene1: B→A | 12 | 0.86 px | 0.74 px | 2.43 px |
| Scene2: C→B | 15 | 0.93 px | 0.80 px | 2.71 px |
| Scene3: E→D | 10 | 1.12 px | 0.95 px | 3.05 px |

---

## A.3 — Warp the Images (Inverse Warping)

I implemented **two interpolation methods** from scratch and use **inverse warping** to avoid holes.

- `warpImageNearestNeighbor(im, H)` — round to nearest pixel.  
- `warpImageBilinear(im, H)` — weighted average of 4 neighbors.

Coordinate convention: the **center of the top-left pixel is (0,0)**. For a destination pixel center \( \mathbf{x}' \), we back-map via \( H^{-1} \) to source coords \( \mathbf{x} \), then sample by NN or bilinear.

**Rectification** uses a homography that maps four clicked corners \( \{(x_i,y_i)\}_{i=1}^4 \) to an axis-aligned rectangle \( \{(u_i,v_i)\}_{i=1}^4 \).

**NN vs. Bilinear (qualitative):**

<div class="img-grid" style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">
  <figure>
    <img src="assets/A3_scene1_src.png" alt="Scene 1 source" />
    <figcaption>Source (Scene 1).</figcaption>
  </figure>
  <figure>
    <img src="assets/A3_scene1_warp_nn.png" alt="Scene 1 NN warp" />
    <figcaption>Warped (Nearest).</figcaption>
  </figure>
  <figure>
    <img src="assets/A3_scene1_warp_bil.png" alt="Scene 1 Bilinear warp" />
    <figcaption>Warped (Bilinear).</figcaption>
  </figure>
</div>

**Rectification deliverable.**

<div class="img-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
  <figure>
    <img src="assets/A3_rectify_input.png" alt="Rectification input" />
    <figcaption>Before (clicked 4 corners).</figcaption>
  </figure>
  <figure>
    <img src="assets/A3_rectify_output.png" alt="Rectification output" />
    <figcaption>After (rectified to W×H).</figcaption>
  </figure>
</div>

---

## A.4 — Build a Mosaic (Feathered Blending)

To register and blend images, I keep one **reference** unwarped and map the others into its frame with their homographies. I compute a soft center-weighted **alpha mask** per image and blend by **weighted averaging**:

$$
I_{\text{out}}(\mathbf{p}) \;=\; 
\frac{\sum_{k} \alpha_k(\mathbf{p})\, I_k(\mathbf{p})}{\sum_{k} \alpha_k(\mathbf{p}) + \varepsilon}\,,
\qquad \varepsilon \approx 10^{-8}.
$$

I pre-compute the global canvas by warping each image’s corners through \( H \), then paste each warped image and alpha at its offset.

**Example mosaics (pairwise and 3+ images):**

<div class="img-grid" style="display:grid;grid-template-columns:1fr;gap:8px;">
  <figure>
    <img src="assets/A4_scene1_mosaic_bilinear.png" alt="Scene 1 mosaic" />
    <figcaption>Scene 1 — Bilinear feathered mosaic.</figcaption>
  </figure>
  <figure>
    <img src="assets/A4_scene2_mosaic_bilinear.png" alt="Scene 2 mosaic" />
    <figcaption>Scene 2 — Bilinear feathered mosaic.</figcaption>
  </figure>
  <figure>
    <img src="assets/A4_scene3_mosaic_bilinear.png" alt="Scene 3 mosaic" />
    <figcaption>Scene 3 — Bilinear feathered mosaic.</figcaption>
  </figure>
</div>

---

## A.5 — Bells & Whistles: Cylindrical Projection

**Mapping from image → cylinder** with focal length \( f \) (pixels) and principal point \( (c_x,c_y) \):

$$
x_c \;=\; f \,\arctan\!\Big(\frac{x-c_x}{f}\Big) + c_x, 
\qquad
y_c \;=\; f \cdot \frac{y-c_y}{\sqrt{1 + \left(\frac{x-c_x}{f}\right)^2}} + c_y.
$$

For warping, I implement the **inverse map** (cylinder → image):

$$
\theta \;=\; \frac{x_c - c_x}{f},\quad
X \;=\; \tan \theta,\quad
r \;=\; \sqrt{1 + X^2},\quad
x \;=\; fX + c_x,\quad
y \;=\; (y_c - c_y)\, r + c_y.
$$

**Pipeline used:**

1. Choose \( f \) (start near \( 1.2\times \) image width).
2. Cylindrical-warp each image via inverse mapping and bilinear sampling.
3. Project clicked correspondences to cylindrical coords using the same \( f \).
4. Estimate pairwise homographies in cylindrical space and compose to a reference:
   $$
   H_{k\to 0} \;=\; H_{j\to 0}\, H_{k\to j}\,,
   $$
   along any available path \( k \to \cdots \to j \to 0 \).
5. Blend on a global canvas with the same alpha-feathering.

**Result (cylindrical panorama):**

<figure>
  <img src="assets/A5_cylindrical_mosaic.png" alt="Cylindrical mosaic" />
  <figcaption>Wide-FOV cylindrical panorama, feather-blended.</figcaption>
</figure>

---

## A.6 — Discussion

**Nearest vs. Bilinear.** NN is fast but alias-prone; bilinear reduces jaggies at the cost of slight blur.

**Artifacts.**
- Mis-clicked correspondences → ghosting; use more points and normalization (or RANSAC).
- Exposure differences → faint seams; a 2-level Laplacian pyramid mitigates high-frequency ghosting.
- Wide sweeps (>180°) → cylindrical projection preferred over pure perspective.

**Design choices.**
- Kept one reference image to limit warp extent.
- Used Hartley-normalized DLT for stable \( H \).
- Feathered alphas with center-high weights to avoid hard seams.

---

## Reproducibility

- Pure NumPy/Matplotlib; no SciPy interpolation shortcuts.
- Notebook functions used:
  - `computeH` (DLT + normalization)
  - `warpImageNearestNeighbor`, `warpImageBilinear`
  - `mosaic_sequence` (alpha-feathering on a global canvas)
  - Cylindrical helpers (`cylindrical_warp_image`, `cylindrical_project_points`, `cylindrical_stitch`)

**Run order**:
1. Load images & matches (JSON).
2. A.2 compute \( H \) + reprojection error.
3. A.3 warps (NN/Bilinear) + rectification.
4. A.4 feathered mosaics.
5. A.5 cylindrical warp + stitch.

---

## Appendix — Figure Index (paths)

- `assets/A1_scene1_matches_A.png`, `assets/A1_scene1_matches_B.png`
- `assets/A3_scene1_src.png`, `assets/A3_scene1_warp_nn.png`, `assets/A3_scene1_warp_bil.png`
- `assets/A3_rectify_input.png`, `assets/A3_rectify_output.png`
- `assets/A4_scene1_mosaic_bilinear.png`, `assets/A4_scene2_mosaic_bilinear.png`, `assets/A4_scene3_mosaic_bilinear.png`
- `assets/A5_cylindrical_mosaic.png`

> Replace these with your generated figures; Quartz renders math blocks delimited by $$...$$ and inline with \( ... \).
