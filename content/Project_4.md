---
title: "Project 4: Neural Radiance Field (NeRF)"
tags:
  - CS280A
  - Proj4
  - ComputerVision
  - NeuralNetworks
  - 3DReconstruction
date: 2025-11-13
permalink: /cs280A/proj4/
draft: false
---

> 📌 **TL;DR**  
> This project explores Neural Radiance Fields (NeRF) for 3D scene reconstruction. I implement camera calibration with ArUco tags, create a custom 3D dataset, build a 2D neural field as a warmup, and finally train a full NeRF from multi-view images. The journey goes from 2D image fitting to full 3D novel view synthesis!

---

# Part 0: Calibrating Your Camera and Capturing a 3D Scan

## 0.1 — Camera Calibration with ArUco Tags

**Goal.** Estimate the intrinsic parameters of my phone camera using ArUco tag detection and OpenCV's calibration pipeline.

**Pipeline:**
1. Captured 30-50 images of the calibration grid from various angles and distances using **iPhone 15 Pro Max's main camera**
2. Detected ArUco markers using OpenCV's `cv2.aruco.detectMarkers()`
3. Extracted corner coordinates for all detected tags
4. Defined 3D world coordinates for tag corners (e.g., for 0.02m × 0.02m tags: `[(0,0,0), (0.02,0,0), (0.02,0.02,0), (0,0.02,0)]`)
5. Used `cv2.calibrateCamera()` to solve for camera matrix $K$ and distortion coefficients

**Camera intrinsics recovered:**
```python
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

**Implementation notes:**
- Handled cases where tags weren't detected (skipped those images)
- Used 4×4 ArUco dictionary (`cv2.aruco.DICT_4X4_50`)
- Validated detection by visualizing corners on images

![[0.1_cam_calibration.png]]

>[!tip]
>[Download the camera calibration file for the main camera of iPhone 15 Pro Max here](/content/static/attachments/proj_4/0.4_my_data.npz)
---

## 0.2 — Capturing My Object Scan

**Object chosen:** A hanging pottery

**Capture strategy:**
- Printed a single ArUco tag with white border
- Placed object adjacent to tag on tabletop
- Captured 50 images maintaining:
  - Consistent camera/zoom (same as calibration)
  - Uniform distance (~10-20cm from object)
  - Varying horizontal and vertical angles
  - Stable lighting (no brightness changes)
  - Sharp focus (minimal motion blur)

![[0.4_example.jpeg]]
![[0.2_3d_object_images_grid_3.png]]

---

## 0.3 — Estimating Camera Pose (Perspective-n-Point)

**Goal.** For each captured image, estimate the camera's position and orientation relative to the ArUco tag coordinate system.

**Pipeline using solvePnP:**

Given the camera intrinsics $K$ from Part 0.1, I solved the classic **PnP problem** for each image:

**Inputs:**
- **objectPoints**: 3D coordinates of tag corners in world space (e.g., `[(0,0,0), (0.02,0,0), (0.02,0.02,0), (0,0.02,0)]`)
- **imagePoints**: Detected 2D pixel coordinates from `detectMarkers()`
- **cameraMatrix**: Intrinsic matrix $K$
- **distCoeffs**: Distortion coefficients

**Output from `cv2.solvePnP()`:**
- **rvec**: Axis-angle rotation vector → converted to rotation matrix $R$ via `cv2.Rodrigues()`
- **tvec**: Translation vector $\mathbf{t}$

These form the **world-to-camera** transformation. I inverted this to get **camera-to-world** ($c2w$) for NeRF:

$$
\begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}^{-1} 
= \begin{bmatrix} R^\top & -R^\top\mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}
$$

**Visualization with Viser:**
![[0.3.1_camera_pose3.png]]
![[0.3.2_camera_pose3.png]]

---

## 0.4 — Creating the Dataset

**Goal.** Package undistorted images and camera poses into `.npz` format for NeRF training.

**Pipeline:**
1. Undistorted all images using `cv2.undistort(img, K, dist_coeffs)`
2. Split into train/val/test sets (e.g., 80/10/10)
3. Saved in required format:

```python
np.savez('my_data.npz',
    images_train=images_train,    # (N_train, H, W, 3) in [0,255]
    c2ws_train=c2ws_train,        # (N_train, 4, 4)
    images_val=images_val,        # (N_val, H, W, 3)
    c2ws_val=c2ws_val,            # (N_val, 4, 4)
    c2ws_test=c2ws_test,          # (N_test, 4, 4)
    focal=focal                   # float
)
```

**Validation:**
- Tested loading and rendering with the same pipeline as `lego_200x200.npz`
- Verified camera poses align with captured scene

>[!tip]
>[Download the Hanging Pottery Dataset here](/content/static/attachments/proj_4/0.4_my_data.npz)


---

# Part 1: Fit a Neural Field to a 2D Image

**Warmup task.** Before tackling 3D, I implemented a simpler **2D neural field** that maps pixel coordinates to RGB colors: $F: (u, v) \rightarrow (r, g, b)$.

## 1.1 — Network Architecture

**MLP with Sinusoidal Positional Encoding (PE):**

The network takes normalized 2D coordinates $(u, v) \in [0,1]^2$ and outputs RGB $\in [0,1]^3$.

**Positional Encoding:**

To capture high-frequency details, I applied sinusoidal PE with frequency levels $L=10$:

$$
\text{PE}(x) = \left[x, \sin(2^0\pi x), \cos(2^0\pi x), \ldots, \sin(2^{L-1}\pi x), \cos(2^{L-1}\pi x)\right]
$$

This maps 2D input to 42D ($2 + 2 \times 2 \times L = 42$).

**MLP Structure:**
![[1_architecture.jpg]]
**Model details:**
- Width: 256 channels
- Depth: 3 hidden layers
- Activation: ReLU (hidden), Sigmoid (output)
- Output: RGB in $[0,1]$

---

## 1.2 — Dataloader Implementation

**Stochastic sampling.** To handle high-resolution images efficiently, I sampled $N=10{,}000$ random pixels per iteration.

**Pipeline:**
1. Generate pixel grid: $(u, v) \in [0, W) \times [0, H)$
2. Normalize: $u' = u/W$, $v' = v/H$
3. Sample $N$ random indices
4. Return: coordinates $(N \times 2)$ and colors $(N \times 3)$ both normalized to $[0,1]$

---

## 1.3 — Training Setup

**Loss function:** Mean Squared Error (MSE) between predicted and ground truth colors

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^N \|\text{RGB}_\text{pred}^{(i)} - \text{RGB}_\text{gt}^{(i)}\|^2
$$

**Optimizer:** Adam with learning rate $\eta = 10^{-2}$

**Metric:** Peak Signal-to-Noise Ratio (PSNR)

$$
\text{PSNR} = 10 \cdot \log_{10}\left(\frac{1}{\text{MSE}}\right)
$$

**Training Setup:** 
- 3000 iterations with batch size 10k on the aforementioned network architecture
- LR=1e-2
- L=10
- batch size=256

---

## 1.4 — Results 

### Training progression:
#### Training on example image
##### The example image is a fox
![[1.fox.jpg]]
During the training, these images have been saved
![[1.1_progress_image_fox.png]]
#### Training on my own image and its PSNR Curve
##### This is the image of my choice:
![[1.6_mariah.jpg]]
##### Here are the training process using the same hyperparameters as the fox:
![[1.6_progress_image_mariah.png]]
##### The PSNR curve:
![[1.6_training_psnr_mariah.png]]
### Hyperparameter ablation (2×2 grid):

#### Testing different positional encoding frequencies $L$ and network widths:
![[1.5_fox_comparison.png]]


**Observations:**
- Low $L$: The higher frequency details are lost
- Low width: The model is underfitted
- Optimal: Reasonably high $L$ and Width

#### Isolated Experiment with fixed network width and varying $L$
The Positional Encoding does make a huge difference in preserving high frequency details
##### 1. $L=2$
![[1.2_progress_image_fox.png]]
![[1.2_training_psnr_fox.png]]

##### 2. $L=5$
![[1.3_progress_image_fox.png]]
![[1.3_training_psnr_fox.png]]

##### 3. $L=10$
![[1.4_progress_image_fox.png]]
![[1.4_training_psnr_fox.png]]

---

# Part 2: Fit a Neural Radiance Field from Multi-view Images

Now we move to 3D! A NeRF represents a scene as a continuous function:

$$
F: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)
$$

where $\mathbf{x} = (x,y,z)$ is a 3D position, $\mathbf{d} = (\theta, \phi)$ is viewing direction, $\mathbf{c} = (r,g,b)$ is emitted color, and $\sigma$ is volume density.

---

## 2.1 — Create Rays from Cameras

### Camera-to-World Transformation

The transformation between world coordinates $\mathbf{X}_w$ and camera coordinates $\mathbf{X}_c$ is:

$$
\begin{bmatrix} \mathbf{X}_c \\ 1 \end{bmatrix} = 
\begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}
\begin{bmatrix} \mathbf{X}_w \\ 1 \end{bmatrix}
$$

**Implemented** `transform(c2w, x_c)`: Applies camera-to-world transformation

$$
\mathbf{X}_w = R^\top(\mathbf{X}_c - \mathbf{t})
$$

**Verification:** Ensured `x == transform(c2w_inv, transform(c2w, x))` holds.

---

### Pixel-to-Camera Conversion

Given camera intrinsics:

$$
K = \begin{bmatrix} f_x & 0 & o_x \\ 0 & f_y & o_y \\ 0 & 0 & 1 \end{bmatrix}
$$

A 3D point projects to pixel $(u,v)$ via:

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} x_c \\ y_c \\ z_c \end{bmatrix}
$$

**Implemented** `pixel_to_camera(K, uv, s)`: Inverts projection to recover camera-space point

$$
\begin{bmatrix} x_c \\ y_c \\ z_c \end{bmatrix} = s \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

---

### Pixel-to-Ray Conversion

For a pinhole camera, each pixel $(u,v)$ defines a ray:

**Ray origin:** Camera position in world space

$$
\mathbf{r}_o = \mathbf{t} \quad \text{(translation from } c2w\text{)}
$$

**Ray direction:** Normalized vector from camera to pixel

1. Choose depth $s=1$
2. Get camera-space point: $\mathbf{X}_c = \text{pixel\_to\_camera}(K, (u,v), 1)$
3. Transform to world: $\mathbf{X}_w = \text{transform}(c2w, \mathbf{X}_c)$
4. Normalize direction:

$$
\mathbf{r}_d = \frac{\mathbf{X}_w - \mathbf{r}_o}{\|\mathbf{X}_w - \mathbf{r}_o\|_2}
$$

**Implemented** `pixel_to_ray(K, c2w, uv)`: Returns ray origin and direction

**Important detail:** Added 0.5 offset to pixel coordinates to sample at pixel centers!

---

## 2.2 — Sampling

### Sampling Rays from Images

**Strategy:** Stochastic mini-batch sampling of $N$ rays per iteration.

**Two approaches:**
1. Sample $M$ images, then $N/M$ rays per image
2. Global sampling across all images

I chose approach 2, as my macbook has 48gb unified memory that happens to be able to fit all the rays.

**Dataloader returns:**
- Ray origins $(N \times 3)$
- Ray directions $(N \times 3)$  
- Ground truth colors $(N \times 3)$

---

### Sampling Points Along Rays

**Goal.** Discretize each ray into 3D sample points.

For the Lego scene, I used:
- `near = 2.0`, `far = 6.0`
- `n_samples = 64`

**Uniform sampling with perturbation:**

$$
t_i = t_{\min} + \frac{i}{N}(t_{\max} - t_{\min}) + \text{noise}
$$

**3D point coordinates:**

$$
\mathbf{x}_i = \mathbf{r}_o + t_i \cdot \mathbf{r}_d
$$

**Perturbation (training only):** Added random jitter to avoid overfitting to fixed sample locations.

---

## 2.3 — Visualization of Data Pipeline
**Used Viser to verify correct implementation**
### 2.3.1. One random ray from each camera with sampled points

![[2.3.1.png]]
### 2.3.2. 100 random rays from one camera with sampled points
![[2.3.2_rays_single_cam.png]]

### 2.3.3. Random rays from the top left corner of one image
![[2.3.3_top_left_corner.png]]

**Validation checks performed:**
- Rays originate from camera positions
- Rays stay within camera frustums
- Sample points lie along rays
- UV coordinates correctly index into images
---

## 2.4 — Neural Radiance Field Architecture

**Network design.** The NeRF MLP takes 3D position $\mathbf{x}$ and view direction $\mathbf{d}$, outputs RGB $\mathbf{c}$ and density $\sigma$.

**Key differences from 2D:**
1. **Inputs:** 3D coordinates (PE with $L=10$) + 3D direction (PE with $L=4$)
2. **Outputs:** RGB color + density $\sigma$
3. **Deeper network:** More capacity for complex 3D structure
4. **Skip connection:** Concatenate input PE at middle layer

**Architecture:**
![[2.4_architecture.jpg]]
**Details:**
- Position PE: $L=10$ → 63D encoding
- Direction PE: $L=4$ → 27D encoding
- Width: 256 channels
- Activations: ReLU (hidden), Sigmoid (RGB), ReLU (density)

---

## 2.5 — Volume Rendering

**The rendering equation.** To render a pixel, we integrate along its ray:

$$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \cdot \sigma(\mathbf{r}(t)) \cdot \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt
$$

where transmittance is:

$$
T(t) = \exp\left(-\int_{t_n}^t \sigma(\mathbf{r}(s)) \, ds\right)
$$

**Discrete approximation** (what I actually compute):

$$
\hat{C}(\mathbf{r}) = \sum_{i=1}^N T_i \cdot \alpha_i \cdot \mathbf{c}_i
$$

with:
- **Alpha (opacity):** $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$
- **Transmittance:** $T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right) = \prod_{j=1}^{i-1}(1-\alpha_j)$
- **Step size:** $\delta_i = t_{i+1} - t_i$

**Physical interpretation:**
- $T_i$: Probability ray survives to sample $i$ (doesn't terminate earlier)
- $\alpha_i$: Probability ray terminates at sample $i$
- Product $T_i \alpha_i$: Probability of first hit at $i$

**Implemented** `volrend(sigmas, rgbs, step_size)`:

Used `torch.cumprod` to efficiently compute transmittance:

```python
def volrend(sigmas, rgbs, step_size):
    # sigmas: (B, N, 1), rgbs: (B, N, 3)
    alphas = 1 - torch.exp(-sigmas * step_size)  # (B, N, 1)
    transmittance = torch.cumprod(1 - alphas + 1e-10, dim=1)  # (B, N, 1)
    transmittance = torch.cat([torch.ones_like(transmittance[:, :1]), 
                               transmittance[:, :-1]], dim=1)
    weights = transmittance * alphas  # (B, N, 1)
    colors = (weights * rgbs).sum(dim=1)  # (B, 3)
    return colors
```

**Validation:** Passed the provided test with `torch.allclose()`.

---

## 2.6a — Training on Lego Dataset

**Training setup:**
- Loss: MSE between rendered and ground truth RGB
- Optimizer: Adam with $\eta = 5 \times 10^{-4}$
- Batch size: 10,000 rays per iteration
- Iterations: 3000 steps

**Training visualization:**

![[2.5.1.png]]

**Training and PSNR curves:**
![[2.5.3_training_curves.png]]
**Novel view synthesis GIF:**
Rendered using provided test camera poses (`c2ws_test`):

![[2.5.2_novel_views.gif]]
>[!tip]
>[Download or view the GIF here](/content/static/images/proj_4/2.5.2_novel_views.gif)


---

## 2.6b — Training with My Own Data

**Goal.** Train NeRF on my custom captured object from Part 0.

### Dataset Adaptations

**Hyperparameter adjustments made:**
- Near/far planes: Changed from (2.0, 6.0) to (0.27, 1.63)
	- Distance estimation
	  ```python
	  # Camera positions are in c2w[:3, 3]
		camera_positions = c2ws_train[:, :3, 3]
		distances = np.linalg.norm(camera_positions, axis=1)
		mean_dist = distances.mean()

		near = mean_dist * 0.3 # or mean_dist - 2*std
		far = mean_dist * 1.8 # or mean_dist + 2*std 
		```
- Samples per ray: 128
- Image resolution: Down scaled to 300x225
- Training iterations: $10000$
- Learning rate: $5e-4$

**Reasoning:**
As the image was downsampled down to similar size as the lego dataset, I keep the model architecture the same. The camera is calibrated the images from the same downsampling process to ensure the match.

---
### Training
![[2.6_training_curves.png]]
![[2.6_training_progress.png]]
### Novel View Gif of the Hang Pottery
**Camera trajectory.** I generated a circular path around the object using the given code:
![[2.6.gif]]
>[!tip]
>[Download or view the GIF here](/content/static/images/proj_4/2.6.gif)
---
### Discussion: Challenges Along the Way
#### 1. Coordinate System Convention Mismatch

During implementation, I encountered a challenging debugging issue where training on my own dataset resulted in **large negative PSNR values** (indicating the network was producing nearly black images), while the provided LEGO dataset worked perfectly. After systematic debugging, I discovered the root cause: **coordinate system convention mismatch between OpenCV and NeRF**.

##### The Problem

OpenCV's `solvePnP` returns camera poses in OpenCV's coordinate system convention:

- **X-axis**: Right
- **Y-axis**: Down ↓
- **Z-axis**: Forward (into the scene)

However, NeRF (and most computer graphics frameworks) use a different convention:

- **X-axis**: Right
- **Y-axis**: Up ↑
- **Z-axis**: Backward (towards the viewer)

Without conversion, the camera poses had all negative Z values, meaning the cameras were positioned "behind" the origin in NeRF's coordinate space. This caused rays to miss the object entirely, resulting in black rendered images and negative PSNR values.

##### The Solution

I applied a coordinate transformation matrix to convert from OpenCV's convention to NeRF's convention:

```python
# Convert from OpenCV coordinate system to NeRF coordinate system
# This transformation is applied AFTER solving PnP and converting to c2w
flip_yz = np.array([
    [1,  0,  0, 0],   # X: remains the same (right)
    [0, -1,  0, 0],   # Y: flip (down → up)
    [0,  0, -1, 0],   # Z: flip (forward → backward)
    [0,  0,  0, 1]    # Homogeneous coordinate
])

# Apply to all camera poses
c2ws = [flip_yz @ c2w for c2w in c2ws]
```



---

# Bells & Whistles

**Depth Map Rendering**

**Goal.** Render depth instead of RGB by compositing per-point depths.

**Volume rendering for depth:**

$$
\hat{D}(\mathbf{r}) = \sum_{i=1}^N T_i \cdot \alpha_i \cdot t_i
$$

where $t_i$ is the depth along the ray to sample $i$.

**Implemented** Modified volume renderer to output depth maps:

```python
def volrend_depth(sigmas, depths, step_size):
    alphas = 1 - torch.exp(-sigmas * step_size)
    transmittance = torch.cumprod(1 - alphas + 1e-10, dim=1)
    transmittance = torch.cat([torch.ones_like(transmittance[:, :1]),
                               transmittance[:, :-1]], dim=1)
    weights = transmittance * alphas
    depth_map = (weights * depths).sum(dim=1)
    return depth_map
```

**Rendered Depth GIF:**
![[Bells_Whistles.gif]]
>[!tip]
>[Download or view the GIF here](/content/static/images/proj_4/Bells_Whistles.gif)

---

