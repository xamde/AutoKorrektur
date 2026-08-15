# 🧪 Testing Insights & Methodological Blueprint from Schellscheidt (2024) & Beckers (2025)

This document synthesizes the empirical evaluations, failure modes, and testing methodologies established in the bachelor theses of **Till Schellscheidt** (*"Autokorrektur – Automatisierte Objektersetzung in Fotos"*, 2024) and **Ben Beckers** (*"Autokorrektur – Inpainting auf mobilen Endgeräten"*, 2025), and translates them into an advanced, automated testing and quality-assurance strategy for AutoKorrektur.

---

## 1. Core Testing Paradigms from the Theses

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       ACADEMIC TESTING TAXONOMY                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [1] Two-Pass Verification (Schellscheidt)                                   │
│      Run YOLO over the inpainting output to detect vehicle hallucinations.   │
│                                                                              │
│  [2] 5-Criteria Evaluation Framework (Beckers & Schellscheidt)               │
│      Quantify: Segmentation, Realism, Seam Consistency, Naturalness, Latency.│
│                                                                              │
│  [3] Seam Boundary Variance (Schellscheidt Abb. 6.2 & Beckers Abb. 9)        │
│      Measure Laplacian / Sobel gradient delta across mask boundary contour.  │
│                                                                              │
│  [4] Human & Active Mobility Protection Collision (Schellscheidt Abb. A.3)   │
│      Ensure mask erosion protects pedestrian & cyclist silhouettes.          │
│                                                                              │
│  [5] Environmental Stress Matrix (Schellscheidt Abb. A.6, A.7, A.9, A.19)    │
│      Night/snow, puddle reflections, fence occlusions, extreme vehicle scale.│
│                                                                              │
│  [6] Memory & Resolution Boundary Clamping (Beckers Kap. 6.2)                │
│      Assert sub-100MB peak heap allocation on 48MP/108MP camera inputs.       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Deep Dive: Key Failure Modes & How to Test Them

### A. Two-Pass Verification & Vehicle Hallucinations (`anyCarsLeft`)
- **Thesis Insight (Schellscheidt p. 36)**: In single-pass latent diffusion, the inpainter generated new vehicle artifacts in **14% of multi-sample images** and **9.7% of one-shot images** because the surrounding street context strongly suggested traffic.
- **Automated Test Implementation**:
  - Feed the post-inpainted bitmap back into `YoloService`.
  - Assert that detected vehicle count is strictly **0** (or bounding box area confidence is below detection threshold).

### B. Seam & Boundary Transition Artifacts (`Konsistenz`)
- **Thesis Insight (Schellscheidt p. 31, Beckers p. 30)**: The most noticeable visual defect in inpainting is the sharp color/sharpness transition or colored halo along the mask perimeter.
- **Automated Test Implementation**:
  - Extract a 10px trimap band along the mask boundary contour.
  - Calculate the **Sobel Edge Variance** and **Color Discontinuity Delta** between the untouched outer boundary and the inpainted inner boundary.
  - Assert that boundary transition variance remains below the acceptable seam threshold ($\Delta E_{94} < 8.0$).

### C. Human & Cyclist Occlusion Collisions (`Menschen im Vordergrund`)
- **Thesis Insight (Schellscheidt p. 36–37, Abb. A.3 Place de la Concorde)**: When vehicles are parked near pedestrians or cyclists, indiscriminate mask dilation eats into the human silhouette, resulting in grotesque inpainting artifacts.
- **Automated Test Implementation**:
  - In a synthetic test scene containing both a car and an overlapping pedestrian/bicycle, assert that `MaskBrushView` or automatic mask assembly preserves non-vehicle pixels ($I_{\text{person}} \cap I_{\text{mask}} = \emptyset$).

### D. Environmental Stress Matrix
From the empirical evaluations of 500 Mapillary Vistas images and 51 real-world photo scenarios:

| Challenge Scenario | Example in Thesis | Test Strategy |
|---|---|---|
| **Low Contrast / Snow / Night** | *Wülfrath snow street (Schellscheidt Abb. A.7)* | Test mask generation under low-light gamma ($\gamma = 0.4$) and high-key snow exposure. |
| **Puddle & Wet Street Reflections** | *Amsterdam bridge canal (Schellscheidt Abb. A.9)* | Test that downward vertical shadow erosion swallows street reflections without lateral ballooning. |
| **Dense Traffic Clusters** | *Zugspitze parking lot (Schellscheidt Abb. A.6)* | Test bounding box merging when $\ge 5$ overlapping vehicles are present. |
| **Extreme Vehicle Scale (>50% ROI)** | *Land Rover close-up (Schellscheidt Abb. A.19)* | Test fallback progressive tile tiling when a vehicle dominates the camera view. |
| **Thin Railing & Fence Occlusion** | *Amsterdam bridge railing (Schellscheidt Abb. A.9)* | Test that instance segmentation captures car body through vertical fence slats. |

---

## 3. Ground-Truth Metric Alignment

To benchmark model quality on desktop and CI (via `backend/benchmark_ml.py`):

1. **IoU & Boundary-IoU**: Validates mask edge snapping against human-annotated vehicle ground truth.
2. **Dice Similarity (F1)**: Quantifies contour accuracy.
3. **Non-Car Background Over-Masking (FPR)**: Ensures trees, sidewalks, and buildings are not needlessly erased.
4. **Peak Signal-to-Noise Ratio (PSNR) & SSIM**: Measures background texture preservation outside the inpainting hole.
5. **Class-Wise Removal Comparison (Oh et al. 2024 / Beckers p. 36)**: Comparing generated street textures against genuine car-free reference images.
