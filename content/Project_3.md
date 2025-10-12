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

# Part B Feature Matching for Autostitching

## B.1 — Harris Corner Detection

**Goal.** Detect many good corners, then keep a *spatially well-distributed* subset.

**Pipeline:**
1) **Harris response** $H(x,y)$   
2) **Peak finding** with a loose threshold  
3) **ANMS:** each corner $i$ gets a radius to the nearest **stronger** corner
$$
r_i \;=\; \min_{j:\,s_j > c\, s_i} \left\| \mathbf{x}_i - \mathbf{x}_j \right\|_2,\quad c\in[0.9,1.0]
$$
Sort by $r_i$ and keep the top $N$ (I kept 250 points).

**Implementation notes:**
- Harris via the given `harris.py` (`get_harris_corners`), then ANMS.
- I discard a $\sim 20$ px border so B.2 can take $40\times 40$ patches.

Dense Harris (the whole image was completely covered by the corners) vs. evenly spread ANMS:  
![[B.1_Harris_ANMS.png]]

---

## B.2 — Feature Descriptor Extraction

**Goal.** A simple, robust descriptor per keypoint (no rotation/scale invariance).

**Steps:**
- Sample a $40\times 40$ window centered on each ANMS point (bilinear; pre-blur $\sigma\approx 1$ to reduce aliasing).  
- **Downsample** to $8\times 8$.  
- **Bias/Gain normalize** per patch:
$$
\tilde{\mathbf{d}} \;=\; \frac{\mathbf{d} - \mu(\mathbf{d})}{\sigma(\mathbf{d}) + \varepsilon},\qquad \varepsilon=10^{-6}
$$

**Implementation.** `extract_axis_aligned_descriptors(...)` returns $(M,64)$ descriptors and the kept keypoints.

**Examples (only first 32 normalized 8×8 patches are shown):**  
![[B.2_8x8_descriptors.png]]

---

## B.3 — Feature Matching

For each descriptor $\mathbf{d}$ in image A, find **nearest** and **second-nearest** in image B by $\ell_2$ distance, keep if
$$
\frac{d_1}{d_2} \;<\; \tau \qquad \text{with } \tau \in [0.65,\,0.8]\ \text{(I used } 0.7\text{)}
$$
I also use **mutual check (cross-check)** A$\leftrightarrow$B.
**Output:** A set of tentative matches (index pairs) that are already quite clean.

**Visualization:**  
![[B.3_descriptor_matching.png]]

---

## B.4 — RANSAC for Robust Homography

**Solver:** I reuse my solver `computeH` (Part A):
1) Normalize points to mean $0$ and mean distance $\sqrt{2}$.  
2) Solve the $2N\times 8$ least-squares system (fix $h_{33}=1$).  
3) Denormalize $H$ and scale s.t. $H_{33}=1$.

**RANSAC loop.**
- Sample $4$ matches $\Rightarrow$ estimate $H$.  
- Score with **forward reprojection error**:
$$
e_k \;=\; \left\| \pi\!\big(H\,\mathbf{x}_k\big) - \mathbf{x}'_k \right\|_2,
\qquad
\pi\!\left(\begin{bmatrix}u\\v\\w\end{bmatrix}\right)=
\begin{bmatrix}\frac{u}{w}\\[2pt]\frac{v}{w}\end{bmatrix}
$$
- Inlier if $e_k < \tau_{\text{px}}$ (I used $2.0$ px).  
- **Refit on inliers** for the final $H$.

**Inlier matches (drawn):**  
![[B.4.1_ransac_inliers.png]]

**Automatic mosaics (reusing Part A warper/blend).**  
I warp all images into a common canvas using inverse mapping (Part A’s `warpImageBilinear`) and feather-blend. We can see the automatically stitched images did not only way **faster** but also more **accurate** in area like the ceiling.
![[B.4.2.Comparison_1.png]]

Here are more examples:
![[B.4.2.Comparison_2.png]]  
![[B.4.2.Comparison_3.png]]

---

## B.5 — Bells & Whistles: Panorama Recognition from an **Unordered** Set ✅

Given a folder of images in arbitrary order, **group** those that belong to the same panorama, **order** each group, and **stitch** automatically.

**How I did it (glue on top of B.1–B.4 + Part A).**
1) **Features per image:** ANMS keypoints + $8\times 8$ descriptors (cached).  
2) **All-pairs match + RANSAC:** keep edge $i\leftrightarrow j$ if inliers $\ge T$ (I used $T=50$).  
3) **Graph view:** images are nodes; edges weighted by inliers.  
   - **Connected components** $\Rightarrow$ panorama groups.  
   - For each group, build a **maximum spanning tree** and **walk leaf-to-leaf** to get a linear order.  
4) **Reference & composition:** choose **middle** image as reference; compose homographies so every image maps **into the reference frame** (fewer distortions than anchoring at an end).  
5) **Warp & feather** (reusing Part A warp/blend).

**Group detection & ordering (schematic):**  
![[B.5.PanoramaRecognition.jpg]]

---

## What I Learned

- **ANMS** is crucial: same number of points, but *far* better coverage than naive top-K Harris.  
- The simple **8×8 patch** with bias/gain normalization is surprisingly strong when combined with **ratio test + cross-check**. 


