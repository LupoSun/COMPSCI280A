---
title: "Project 3: (Auto)stitching and Photo Mosaics"
tags:
  - CS280A
  - Proj3
  - Computer Vision
  - Photography
date: 2025-10-08
permalink: /cs280A/proj3/
draft: false
---

> 📌 **TL;DR**  
> You’ll find my correspondences, homography estimation, inverse warps (nearest & bilinear), rectification, feathered mosaics, and a cylindrical “bells & whistles” panorama below. All figures are reproducible from the accompanying notebook.

---
# Part A
## A.1—Shoot and digitize pictures
### A.1.1 Choice of Images
I took three sets of images of the scenes in Wurster Hall for the testing materials of the project 3.
![[A.1_formation.png]]
### A.1.2 Point Correspondences

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

I implemented **DLT with least squares** to solve the overdetermined system \(Ah=b\) where \(h\) holds the 8 free parameters of \(H\) (with \(h_{33}=1\)). I used your provided function signature:

```python
H = computeH(im1_pts, im2_pts)  # maps src -> dst
```

We solve for the homography \( H \) (3×3) mapping homogeneous points $$ \mathbf{x}' \sim H\,\mathbf{x} $$
Using DLT with least squares, we assemble an overdetermined linear system:

$$
A\,\mathbf{h} = \mathbf{b}, \quad 
\mathbf{h} = \begin{bmatrix} h_{11}&h_{12}&h_{13}&h_{21}&h_{22}&h_{23}&h_{31}&h_{32} \end{bmatrix}^\top,\quad h_{33}=1.
$$

For each correspondence $$( (x,y) \leftrightarrow (x',y') $$ :

$$
\begin{aligned}
x' &= \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + 1},\\
y' &= \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + 1}.
\end{aligned}
$$

---

## A.3 — Warp the Images (Inverse Warping)

I implemented **both** interpolation methods from scratch and use inverse warping to avoid holes.

- `warpImageNearestNeighbor(im, H)`  
- `warpImageBilinear(im, H)`

**NN vs. Bilinear**:
![[A.3.1_Comparison_nn_bl.png]]
As my observation, both nearest neighbor and bilinear warping achieved quite good quality on my 24 megapixel images.

![[A.3.2_Comparison_Sequence.png]]
From this image, we can see the image 1 being warped via the homography to match image 2.

---

## A.4 — Build a Mosaic (Feathered Blending)

I blend on a global canvas via **soft center-weighted alphas** and **weighted averaging** (feathering). The reference image remains unwarped; other images are mapped into its frame with the recovered homographies.

- Canvas sizing: predicted by warping image corners through \(H\).
- Alpha: 1 at the center, decays to 0 near borders (linear falloff).

Here are the results:
![[A.4.a_Mosaic.png]]

**Other Eexample mosaics:**
![[A.4.b_Mosaic.png]]

![[A.4.c_Mosaic.png]]

---

## A.5 — Bells & Whistles: Cylindrical Projection

I implemented cylindrical warping and stitched in cylindrical space.

**Why cylindrical?** It converts horizontal camera yaw into near-translations on the cylinder, reducing perspective distortion across wide fields of view.

**Mapping from image → cylinder** with focal length \( f \) (pixels) and principal point \( (c_x,c_y) \):

$$
x_c \;=\; f \,\arctan\!\Big(\frac{x-c_x}{f}\Big) + c_x, 
\qquad
y_c \;=\; f \cdot \frac{y-c_y}{\sqrt{1 + \left(\frac{x-c_x}{f}\right)^2}} + c_y.
$$

The images look like this when warped onto cylinders:

![[A.5.1_cylindrical_individual.png]]

**Pipeline used:**
1. Pick \(f\) (pixels); start near \(1.2\times\) image width.
2. Warp each image to the cylinder (inverse map).
3. Project clicked correspondences to cylindrical coords (same \(f\)).
4. Estimate pairwise homographies in cylindrical space and compose to a reference.
5. Blend on a global canvas (same feathering).

**Result (cylindrical projection with difference focal lengths):**

![[A.5.3_focal_length_comparison.png]]
The balanced perspektive looks the best!

---
