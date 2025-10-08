---
title: "Project 1: Images of the Russian Empire"
tags:
  - CS280A
  - Proj3
  - Computer Vision
  - Photography
date: 2025-09-10
permalink: /cs280A/proj3/
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
- Number of correspondences per pair: 20 (to ensure stability).
- I verified matches visually with numbered overlays.

**Examples:**
![[A.2_Correspondances.png]]

---

## A.2 — Recover Homographies

I implemented **DLT with least squares** (and Hartley normalization) to solve the overdetermined system \(Ah=b\) where \(h\) holds the 8 free parameters of \(H\) (with \(h_{33}=1\)). I used your provided function signature:

```python
H = computeH(im1_pts, im2_pts)  # maps src -> dst
```

**Numerical sanity:** I report the mean reprojection error for each pair:

| Pair | N pts | mean | median | max |
|---|---:|---:|---:|---:|
| Scene1: B→A | 12 | 0.86 px | 0.74 px | 2.43 px |
| Scene2: C→B | 15 | 0.93 px | 0.80 px | 2.71 px |
| Scene3: E→D | 10 | 1.12 px | 0.95 px | 3.05 px |

*(Computed as \(\|p' - \hat p'\|\) with \(\hat p' \sim H p\).)*

---

## A.3 — Warp the Images (Inverse Warping)

I implemented **both** interpolation methods from scratch and use inverse warping to avoid holes.

- `warpImageNearestNeighbor(im, H)`  
- `warpImageBilinear(im, H)`

**NN vs. Bilinear (qualitative):**
![[A.3.1_Comparison_nn_bl.png]]

**Rectification deliverable.** Using 4 clicked corners of a planar rectangle, I rectified to an axis-aligned box:

![[A.3.2_Comparison_Sequence.png]]

---

## A.4 — Build a Mosaic (Feathered Blending)

I blend on a global canvas via **soft center-weighted alphas** and **weighted averaging** (feathering). The reference image remains unwarped; other images are mapped into its frame with the recovered homographies.

- Canvas sizing: predicted by warping image corners through \(H\).
- Alpha: 1 at the center, decays to 0 near borders (linear falloff).
- Alternatives tested: 2-level Laplacian (optional; not required).

**Example mosaics (pairwise and 3+ images):**

![[A.4.a_Mosaic.png]]

![[A.4.b_Mosaic.png]]

![[A.4.c_Mosaic.png]]

---

## A.5 — Bells & Whistles: Cylindrical Projection

I implemented cylindrical warping and stitched in cylindrical space.

**Why cylindrical?** It converts horizontal camera yaw into near-translations on the cylinder, reducing perspective distortion across wide fields of view.

**Mapping (source → cylinder):**
\[
x_c = f\,\arctan\!\Big(\frac{x-c_x}{f}\Big) + c_x,\qquad
y_c = f \cdot \frac{y-c_y}{\sqrt{1 + \left(\frac{x-c_x}{f}\right)^2}} + c_y.
\]

![[A.5.1_cylindrical_individual.png]]

**Pipeline used:**
1. Pick \(f\) (pixels); start near \(1.2\times\) image width.
2. Warp each image to the cylinder (inverse map).
3. Project clicked correspondences to cylindrical coords (same \(f\)).
4. Estimate pairwise homographies in cylindrical space and compose to a reference.
5. Blend on a global canvas (same feathering).

**Result (cylindrical panorama):**

![[A.5.3_focal_length_comparison.png]]

---

## A.6 — Discussion

**Nearest vs. Bilinear.** NN is fast but blocky; bilinear is smoother (slightly blurrier edges). Quantitatively, bilinear reduced aliasing near edges and along high-frequency textures.

**Failure cases & artifacts.**
- Mis-clicked correspondences lead to ghosting; normalization + more points help.
- Alpha feathering can leave faint seams if exposure varies; a 2-level Laplacian helps.
- Very wide sweeps (>180°) benefit from cylindrical projection; perspective mosaics distort.

**Design choices.**
- Kept one image as reference to reduce warp extent.
- Used Hartley normalization in DLT for stability.
- Chose center-falloff alphas for simple, robust blending.

---
