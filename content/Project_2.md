---
title: "Project 2: Filters, Hybrids, and Blending"
tags:
  - CS280A
  - proj2
  - image_processing
  - hybrid_images
  - blending
date: 2025-09-19
permalink: /cs280A/proj2/
draft: false
---
# Overview

If Project 1 was about seeing the world in grayscale filters, Project 2 was about teaching my eyes to separate **high-frequency gossip** (edges, details) from **low-frequency vibes** (smooth stuff).  

This time, the journey took me through:
1. Building convolutions by hand (a rite of passage).  
2. Extracting edges with finite differences and the derivative of Gaussian (DoG).  
3. Sharpening and hybrid illusions that break your brain.  
4. Gaussian & Laplacian stacks.  
5. Multiresolution blending, aka turning apples into oranges and Mariah into JLo.  

Along the way, I discovered two truths:  
- Sigma is the puppet master of all things blurry.  
- Masks are surprisingly political (hard vs. soft edges).  

---

# Part 1: Fun with Filters

## 1.1 Convolutions from Scratch

Four nested loops → two loops → finally checked against SciPy.  
It worked, it always feels nIcE to reinvent the wheel;)


![[1.1_comparison.png]]

---

## 1.2 Finite Difference Operator

First taste of edges:  

$$
I_x = I * D_x, \quad I_y = I * D_y
$$

Gradient magnitude:

$$
\| \nabla I \| = \sqrt{I_x^2 + I_y^2}
$$

Then apply thresholding to keep only the strongest edges.  
The result? Looks like the photo was struck by lightning — only the outlines remain.  

![[1.2_comparison.png]]

---

## 1.3 Derivative of Gaussian (DoG)

![[1.3_comparison.png]]
![[1.3.2_comparison.png]]
As a result the DoG filter does pick up less noise and return more defined edges.

Two options:
- Blur then differentiate, or  
- Differentiate the Gaussian itself and convolve once.  
Both give nearly **identical looking** results:  

$$
I * (D_x * G) \;\;\approx\;\; (I * G) * D_x
$$

![[1.3.3_DoG_vs_two_pass.png]]

---

### Bells & Whistles (Part 1)

Edges also have *direction*.  

Gradient orientation:

$$
\theta(x,y) = \arctan\left(\frac{I_y(x,y)}{I_x(x,y)}\right)
$$

I mapped $\theta$ to hue in HSV. 
Result: The rainbow edges.  

![[1.X_comparison.png]]

---

# Part 2: Fun with Frequencies

## 2.1 Image Sharpening

Unsharp masking:

1. Blur the image:  
   $$
   I_b = I * G
   $$

2. Extract detail (high-pass):  
   $$
   H = I - I_b
   $$

3. Add scaled detail:  
   $$
   I_{\text{sharp}} = I + \alpha H
   $$

With the right $\alpha$, the Taj Mahal looks majestic. With the wrong $\alpha$, the image is channelling the cheapest cellphone apparently (according to Alyosha)

![[2.1_unsharp_masking.png]]
![[2.1_blurNresharp.png]]


---

## 2.2 Hybrid Images

Hybrid images are visual illusions:  
- From close, you see one face.  
- From far, you see another.  
- From medium distance, your brain glitches.
Up close you see one face, far away you see another.  

![[2.2_hybrid_image_JLo_Mariah.png]]
I also checked the Fourier transforms:  
- **Row 2:** The Fourier transforms show us the frequency content. You can see that Mariah’s image has strong low-frequency energy concentrated tightly in the center (smooth shading, broad features), while JLo’s has more spread-out energy toward the edges, indicating richer high-frequency details (wrinkles, edges, fine textures). The hybrid’s spectrum combines both patterns.
- **Row 3:** This makes the decomposition crystal clear. On the left, JLo’s **high-pass image** isolates her sharp features—the crisp edges around the eyes, nose, and facial lines. In the middle, Mariah’s **low-pass image** keeps her soft skin tones and broader facial structure. On the right, adding them back together reconstructs the hybrid, where depending on your viewing distance, you either see Mariah’s smooth features or JLo’s sharp details pop out.
![[2.2_hybrid_image_Ye_Taylor.png]]


---

### Bells & Whistles for Hybrids

I didn’t just stop at grayscale — I tried different **color strategies** for the high- and low-frequency bands:
- **Gray + Gray:** Both inputs converted to grayscale. Classic hybrid illusion, but no color cues.
- **Low-pass Color + High-pass Gray:** Mariah contributes the colorful smooth base, while JLo adds grayscale details. The hybrid feels tinted but still dominated by Mariah’s tones.
- **High-pass Color + Low-pass Gray:** JLo’s sharp details (in full color) layered onto a grayscale Mariah. The wrinkles and edges pop out more strongly, now with color.
- **Color + Color:** Both frequency bands keep their color. This creates the most vivid, but also the most chaotic, hybrid.


![[2.2_hybrid_image_JLo_Mariah_BellsnWhistles.png]]

---

## 2.3 Gaussian & Laplacian Stacks

Stacks are pyramids’ lazier cousins: same image size, just blurrier and blurrier.  
Laplacian stacks show “edges at different scales.”  
It’s like peeling the image layer by layer, from smooth skin to sharp wrinkles.  
I built stacks for the apple and orange. Already, the “Oraple” was peeking through.

![[2.3.1_oraple.png]]

---

## 2.4 Multiresolution Blending

Here’s the fun part: combine two images with a mask, but do it at multiple scales.  
Steps:  
1. Build Gaussian stacks of the mask.  
2. Build Laplacian stacks of both images.  
3. Blend at every level.  
4. Reconstruct.  
Hard-edged masks → ugly seam.  
Soft Gaussian masks → magic.

![[2.4.1_oraple.png]]  
Then I tried the images I hybrided before, apparently Mariah still doesn't know her.
![[2.4.2_jlo_mariah_blend.png]]
This is where the story gets spicy.  
I tried blending **Miley Cyrus and the Wrecking Ball** using a irregular mask.  
Problem: the hue difference between **Miley’s skin** and **cold gray steel** was massive.  
Result: instead of Miley on a wrecking ball, it looked like she was glued to a moon rock.  
Solution: I warmed up the wrecking ball’s color tones to match her skin. After that, the illusion clicked.  

![[2.4.3_tongue_blend.png]]
Lesson learned: sometimes the biggest challenge isn’t the math, it’s color harmony.  


---

### Bells & Whistles (Part 2)
I tested four strategies:  
- GG = Gray + Gray  
- LG = Low-pass Color + High-pass Gray  
- HG = High-pass Color + Low-pass Gray  
- CC = Color + Color  
Using color in the low-pass image produced the most natural results, while color in the high-pass image appeared noisy. Both-gray lost too much realism, and both-color gave the most compelling blends. This suggests that color information is better carried in the low-frequency channels.
![[2.4.X_color_blend.png]]

---

# Reflection

This project taught me:  
- High frequencies = crispy details. Low frequencies = the big picture.  
- Masks are powerful: soft ones hide your crimes, hard ones expose them.  
- Alignment and color balance matter as much as math.  
- And most importantly: hybrid images are a great way to gaslight your friends.  

---