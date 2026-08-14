# AutoKorrektur — Project Status & Roadmap

> **Last Updated & Verified**: 2026-08-15
> **Status**: Core ML pipeline, segmentation, inpainting fidelity, EXIF orientation, memory safety, test coverage, Live Camera AR engine, Multi-Model LaMa architecture, ProGuard/R8 release signing, and CI/CD fully completed and verified.

---

## 1. Overall Goal

Build a production-ready Android application for automatic vehicle removal and photorealistic inpainting using a hybrid machine learning architecture:
- **On-Device (Default)**: YOLOv11-seg instance segmentation + MI-GAN & LaMa local neural inpainting.
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

## 3. Completed Phases (Verified as of 2026-08-14)

### ✅ Phase 1: Guided Filter Edge Refinement & Model Asset Delivery

- [x] **Q1. OpenCV Guided Filter Edge Refinement**
- [x] **R1. Play Asset Delivery (PAD) & Asset Optimization**
- [x] **R2. SBOM & License Attribution**

### ✅ Phase 2: Cloud SDXL Daily Quota & Architecture

- [x] **Q2. Cloud SDXL Daily Free Quota Manager (5 Edits/Day)**
- [x] **A1. Extract GDPR Consent Management**
- [x] **A2. Dependency Injection Architecture & ViewModel Factories**
- [x] **A3. Standardized ML Exception Hierarchy**

### ✅ Phase 3: CI, Benchmarks & Quality Assurance

- [x] **T1–T5. Benchmark suites (Desktop, On-Device, Visual Diff, Physical Device Edge-Cases)**
- [x] **C1. Code Coverage Gates**

### ✅ Phase 4: Code Review Sweep (CR-01 through CR-38)

- [x] All 38 memory safety, backend async, architecture, test coverage, code cleanup, performance, and documentation items completed.

---

## 4. Product Roadmap & Advanced Features

---

### 🧭 Phase 6A: Product Strategy, Customer Personas & UX/UI Alignment

- [x] **UX-01. Customer Persona & Target Audience Definition**
    - Established `docs/PERSONAS.md` defining the 4 core customer archetypes (Dealership Lot Managers, Real Estate Photographers, Street/Social Creators, and Live AR users).
- [x] **UX-02. Workflow Archetype & Job-to-be-Done (JTBD) Mapping**
    - Mapped Batch Queue, Studio Precision, Social Export, and Live AR modes to customer workflows.
- [x] **UX-03. Comprehensive UI/UX Heuristic Audit**
    - Audited layout against Material 3 standards and eliminated friction points.
- [x] **UX-04. Methodical Guided Discovery Framework**
    - Formulated the 3-step persona-to-wireframe alignment framework.
- [x] **UX-05. Tailored UI Architecture Foundation**
    - Laid out multi-mode navigation architecture across Studio, Batch, Live AR, and Settings.

---

### 🎥 Phase 6B: Live Camera Real-Time AR Inference & Viewfinder Overlay (Item 1)

- [x] **AR-01. CameraX ImageAnalysis Zero-Copy YUV-to-OpenCV Converter**
    - Implemented `ArFrameConverter.kt` converting `ImageProxy` YUV420 planes to OpenCV RGBA matrices with sensor rotation normalization (0°, 90°, 180°, 270°). Verified via `ArFrameConverterTest`.
- [x] **AR-02. Asynchronous Frame-Skipping ML Inference Loop**
    - Implemented `RealtimeArPipeline.kt` with non-blocking atomic frame skipping and rolling FPS calculation. Verified via `RealtimeArPipelineTest`.
- [x] **AR-03. Live Temporal Background Buffer Blending in Viewfinder**
    - Enhanced `TemporalBackgroundAccumulator.kt` with `hasAccumulatedBackground` state tracking and synthetic multi-frame motion tests. Verified via `TemporalBackgroundAccumulatorTest`.
- [x] **AR-04. Custom Hardware-Accelerated AR Viewfinder Renderer**
    - Integrated `arOverlayView` with live FPS indicator badge in `activity_ar_camera.xml` and `ArCameraActivity.kt`.
- [x] **AR-05. High-Resolution Still Photo Capture with AR Composite Stitching**
    - Added instant AR photo export to gallery via `ImageExportManager` upon tapping shutter button. Verified via `ArCameraActivityInstrumentedTest`.

---

### 🚀 Phase 6C: Production Release Signing & Play Store Deployment (Item 2)

- [x] **REL-01. Android Keystore Management & Secret Vault Configuration**
    - Configured release signing config with `keystore.properties` support in `app/build.gradle.kts`.
- [x] **REL-02. Production ProGuard / R8 Optimization & JNI Reflection Guards**
    - Hardened `proguard-rules.pro` with reflection rules for WorkManager, OpenCV, ONNX Runtime, and JSON data classes.
- [x] **REL-03. Android App Bundle (AAB) Generation**
    - Added `:app:bundleRelease` task verification in CI workflow.
- [x] **REL-04. Automated Google Play Store CI/CD Release Track**
    - Updated `.github/workflows/ci.yml` with full release packaging steps.

---

### 🧠 Phase 6D: Multi-Model Local Inpainting Support — LaMa Integration (Item 4)

- [x] **ML-01. Inpainting Model Type Selection Enum**
    - Implemented `InpaintingModelType` supporting MI-GAN, LaMa, and Cloud SDXL.
- [x] **ML-02. Implement `LamaInference.kt` Engine**
    - Created `LamaInference.kt` supporting dynamic spatial dimension padding (multiples of 8) and structural inpainting fallback. Verified via `LamaInferenceUnitTest` & `LamaInferenceInstrumentedTest`.
- [x] **ML-03. Inpainting Engine Factory**
    - Implemented `InpaintingEngineFactory.kt` for dynamic inpainting model instantiation.
- [x] **ML-04. Inpainting Benchmark & Unit Tests**
    - Added comprehensive unit and instrumented tests for LaMa padding and model factory.

---

## 5. Summary Statistics

| Category | Completed | Remaining |
|---|---|---|
| Historical Milestones & Backlog (M1–M8, Phases 1–4) | 55 | 0 |
| **Phase 6A: Product Strategy & UX Alignment** | 5 | 0 |
| **Phase 6B: Live Camera Real-Time AR (Item 1)** | 5 | 0 |
| **Phase 6C: Production Release & Play Store (Item 2)** | 4 | 0 |
| **Phase 6D: Multi-Model Inpainting / LaMa (Item 4)** | 4 | 0 |
| **Total Project Tasks** | **73** | **0** |
