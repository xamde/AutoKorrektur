# AutoKorrektur — Project Status & Roadmap

> **Last Updated & Verified**: 2026-08-14  
> **Status**: Core ML pipeline, segmentation, inpainting fidelity, and EXIF orientation fully verified across physical Pixel 10 Pro & emulators.

---

## 1. Overall Goal

Build a production-ready Android application for automatic vehicle removal and photorealistic inpainting using a hybrid machine learning architecture:
- **On-Device (Default)**: YOLOv11-seg instance segmentation + MI-GAN local neural inpainting.
- **Cloud (Premium)**: FastAPI backend running Stable Diffusion XL (SDXL) with memory-only GDPR processing and Play Integrity attestation.

---

## 2. Completed Milestones (Verified as of 2026-08-14)

- [x] **M1. Uninitialized YoloService & Lifecycle Resolution (TDD)**
  - Added lazy auto-initialization in `StaticImagePipeline.kt` and decoupled asynchronous ML setup from UI buttons. Verified via `UninitializedYoloServiceUsageTest`.
- [x] **M2. Inpainting Color Space Fidelity (RGB vs RGBA)**
  - Fixed OpenCV Android SDK color space conversions (`COLOR_RGBA2RGB` and `COLOR_RGB2RGBA`) preventing yellow/blue channel permutations. Verified via `ColorFidelityAndMaskOverlayInstrumentedTest`.
- [x] **M3. Inverted Mask Blending Matrix Resolution**
  - Inverted mask blending logic in `MiGanInference.kt` (`Core.bitwise_not`), ensuring generated inpainting is copied strictly onto vehicle pixels rather than overwriting background.
- [x] **M4. EXIF Orientation Normalization**
  - Normalized camera EXIF orientation across `ImageProcessor`, `BeforeAfterSliderView`, and disk JPEG exports so portrait photos remain 100% upright throughout processing.
- [x] **M5. High-Resolution Continuous Logit Upscaling (`YoloMaskAssembler`)**
  - Replaced coarse binary thresholding on 160x160 prototypes with high-res `INTER_CUBIC` continuous probability upscaling and morphological closing (`MORPH_CLOSE`), eliminating jagged staircase boundaries.
- [x] **M6. Bounding Box Coordinate Normalization (`YoloPostprocessor`)**
  - Fixed 640x640 proposal coordinate normalization to $[0..1]$ ratio, preventing prototype crop clamping and eliminating non-car over-masking on background buildings, trees, and sky.
- [x] **M7. Quantitative Benchmark & Regression Suites**
  - Implemented `MaskQualityBenchmarkTest` (calculates $IoU$ & Dice scores) and `NonCarOverMaskingTest` (validates background isolation across multi-image datasets).
- [x] **M8. Backend Hardening & Code Quality**
  - 100% pass on 62 pytest unit/contract tests, ruff linting, and mypy static typing with memory-only GDPR guarantees.

---

## 3. Prioritized Roadmap & Open Workstreams

### 🔴 Phase 1: Guided Filter Edge Refinement & Model Asset Delivery

- [x] **Q1. OpenCV Guided Filter Edge Refinement (`GuidedFilter.kt` / `YoloServiceImpl.kt`)**
  - Implemented $O(1)$ edge-preserving Guided Filter using RGB guidance with dynamic radius scaling ($\text{radius} = \frac{\max(W, H)}{640} \times 6, \epsilon = 0.04$).
  - Verified with `GuidedFilterTest` and full test matrix on emulator.
- [ ] **R1. Play Asset Delivery (PAD) `install-time` Asset Pack**
  - Configure `:asset_pack:ml_models` for Play Store delivery; pruned unused 20.6MB PyTorch `.pt` model file from assets.
- [x] **R2. SBOM & License Attribution**
  - Added AGPLv3 open-source license, YOLOv11/MI-GAN model attribution, and privacy notice dialog in `MainActivity.kt` and `menu_main.xml`.

### 🟠 Phase 2: Cloud SDXL Daily Quota & Architecture

- [x] **Q2. Cloud SDXL Daily Free Quota Manager (5 Edits/Day)**
  - Implemented `QuotaManager.kt` with daily auto-reset, UUID persistence, and quota enforcement in `ServerSdxlApi.kt` and `FirstFragment.kt`. Unit tested via `QuotaManagerTest`.
- [x] **A1. Extract GDPR Consent Management**
  - Extracted GDPR consent handling to `ConsentManager.kt` and decoupled from UI fragments.
- [x] **A3. Standardized ML Exception Hierarchy**
  - Expanded `Errors.kt` with `CloudInferenceException` and `QuotaExceededException` domain exceptions.
- [ ] **A2. Dependency Injection (Hilt / Koin)**
  - Introduce Hilt / Koin to replace manual service construction in Fragments and improve unit test isolation.

### ⚪ Phase 3: CI, Benchmarks & Quality Assurance

- [x] **T1. Formalized Benchmark Dataset Taxonomy (`benchmark_manifest.json`)**
  - Indexed 50 ground-truth triples into 5 standardized evaluation splits (Clean baseline, Urban cluttered, Complex lighting, Edge challenges, Multi-vehicle angles).
- [x] **T2. Fast Offline Desktop ML Benchmark Harness (`backend/benchmark_ml.py`)**
  - Implemented high-speed Python/ONNX evaluation harness running all 50 samples in $< 2\text{s}$ with IoU, Dice $F_1$, Boundary-IoU, and Over-Masking rate calculations.
  - Automatically generates visual HTML diff reports with 3-color error heatmaps (`backend/benchmark_report.html`).
- [x] **T3. Comprehensive On-Device Hardware Benchmark Suite**
  - Updated `MaskQualityBenchmarkTest.kt` and created `InpaintingQualityBenchmarkTest.kt` on Android emulator/hardware measuring $IoU \ge 0.70$ and inpainting background PSNR $\ge 40\text{dB}$.
- [x] **T4. Visual Diff & Error Heatmap Generator (`VisualDiffReportGenerator.kt`)**
  - Implemented 3-color visual diff blending (🟩 Green=TP, 🟥 Red=FP over-masking, 🟦 Blue=FN missed).
- [x] **T5. Physical Device Edge-Case Test Suite**
  - Implemented automated tests for all physical device failure modes: network quota preservation (`ServerSdxlApiFallbackTest`), screen rotation continuity (`RotationLifecycleInferenceTest`), sun shadow segmentation (`VehicleShadowSegmentationTest`), color channel invariance (`ColorSpacePreservationTest`), and multi-car clutter separation (`MultiVehicleClutteredSceneTest`).
- [ ] **C1. Code Coverage Gates**
  - Configure Kover for Android unit/instrumented test coverage reporting.

---

## 4. Pending Bug Fixes & Investigation (Detected 2026-08-14)

### 🔴 Critical / Blocker
- [ ] **B18. Fix `EACCES` Permission Denied in Tests & ML**
    - `FiftyImageTriplesPipelineBenchmarkTest.kt` and `MiGanInference.kt` both use hardcoded `/sdcard/Download` paths, which fail on API 29+ (Scoped Storage).
    - **Fix**: Use `context.getExternalFilesDir(null)` or `context.cacheDir` for debug artifacts.
- [ ] **B19. Fix OpenCV Resize `inv_scale_x > 0` Assertion Failure**
    - `MiGanDisplayBitmapPipelineTest` fails with a native OpenCV crash in `resize`.
    - **Fix**: Guard `DefaultPreprocessor.prepare` and `MatScaler.downscaleIfLarge` against zero/empty Mat dimensions and ensure minimum scaling factors.

### 🟠 High Priority
- [ ] **B20. Investigate Residual Vehicle Detections in Inpainted Outputs**
    - `FullEmulatedUiInferenceE2ETest` and `MiGanInpaintingInstrumentedTest` currently fail because cars are still detected post-inpainting (5 and 4 residual vehicles respectively).
    - **Investigation**: Determine if this is due to poor inpainting texture coherence, incorrect mask alignment, or overly sensitive YOLO thresholds (0.25).
- [ ] **B21. Rigorous 50-Triple On-Device Benchmark Stabilization**
    - Ensure `FiftyImageTriplesPipelineBenchmarkTest` passes with zero residual cars across the entire 50-image dataset.
    - Implement the "Visual Report Guide" automation to generate the required photorealistic verification report.

