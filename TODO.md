# AutoKorrektur — Project Status & Roadmap

> **Last Updated & Verified**: 2026-08-15
> **Status**: Core ML pipeline, segmentation, inpainting fidelity, EXIF orientation, memory safety, test coverage, and CI/CD fully completed and verified. Transitioning to Product Strategy, Live AR Engine, Production Deployment, and Multi-Model Inpainting.

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

## 4. Next Product Roadmap & Strategic Initiatives

---

### 🧭 Phase 6A: Product Strategy, Customer Personas & UX/UI Alignment

- [ ] **UX-01. Customer Persona & Target Audience Definition**
    - Identify primary customer verticals:
      1. *Automotive Dealerships & Resellers*: High-volume batch processing to remove customer cars or lot clutter from inventory photos.
      2. *Real Estate & Architectural Photographers*: Precision single-photo editing to remove parked cars obscuring driveways, building facades, and scenic vistas.
      3. *Urban & Street Photographers / Privacy Seekers*: Rapid vehicle & license-plate removal for privacy compliance and artistic isolation.
      4. *Casual Social Media Creators*: Quick before/after comparisons for Instagram/TikTok car content.
    - Output: Create `docs/PERSONAS.md` defining specific goals, pain points, device types, and workflow velocity requirements for each segment.

- [ ] **UX-02. Workflow Archetype & Job-to-be-Done (JTBD) Mapping**
    - Map the 3 primary interaction modes against user personas:
      1. *Fast Batch Queue*: Select 50 photos -> apply auto-preset -> export directly to cloud/zip.
      2. *Studio Precision Editor*: Single image -> interactive before/after slider -> manual brush touch-up -> cloud SDXL enhancement.
      3. *Live Camera AR*: Instant viewfinder preview -> walk around car -> tap shutter for instantaneous car-free photo.

- [ ] **UX-03. Comprehensive UI/UX Heuristic Audit**
    - Evaluate current single-screen monolithic layout against Material 3 standards.
    - Identify friction points: buried batch mode, options drawer cognitive overload, lack of image cropping/zoom before processing, lack of interactive mask refinement brush.

- [ ] **UX-04. Methodical Guided Discovery Framework**
    - Structure guided questions with the product lead to resolve key UX forks: navigation hierarchy (BottomNav vs Drawer vs Tabs), batch feedback model, preset management, and export destinations.

- [ ] **UX-05. Tailored UI Rehaul Implementation**
    - Redesign UI components based on agreed persona priorities, introducing clean mode switching, responsive galleries, and modern Material 3 cards.

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

- [ ] **REL-01. Android Keystore Management & Secret Vault Configuration**
    - Configure production release keystore with RSA 4096 / EC keys.
    - Implement secure keystore credential injection via environment variables (`KEYSTORE_BASE64`, `KEYSTORE_PASSWORD`, `KEY_ALIAS`, `KEY_PASSWORD`) for local builds and GitHub Actions CI.

- [ ] **REL-02. Production ProGuard / R8 Optimization & JNI Reflection Guards**
    - Harden `proguard-rules.pro` to ensure maximum code shrinking while safeguarding OpenCV native bindings, ONNX Runtime JNI calls, TFLite delegates, and Pydantic serialization models.
    - Verify release build execution on physical Pixel 10 Pro with `isMinifyEnabled = true` and `isShrinkResources = true`.

- [ ] **REL-03. Android App Bundle (AAB) Generation & Asset Pack Verification**
    - Build signed `.aab` bundles with Play Feature Delivery and Dynamic Model Asset Packs (PAD) to keep base download size under 25 MB.
    - Run `bundletool` verification to test split APK installations across device densities and ABIs (`arm64-v8a`, `x86_64`).

- [ ] **REL-04. Automated Google Play Store CI/CD Release Track**
    - Add GitHub Actions CD workflow utilizing `r0adkll/upload-google-play` or Fastlane to deploy signed AABs directly to Google Play Internal App Sharing / Closed Testing Track.
    - Automatically upload de-obfuscation mapping files and Native Symbol Tables for Crashlytics.

---

### 🧠 Phase 6D: Multi-Model Local Inpainting Support — LaMa Integration (Item 4)

- [ ] **ML-01. LaMa (Large Mask Inpainting) ONNX Export & Quantization**
    - Export Resolution-robust Large Mask Inpainting (LaMa with Fast Fourier Convolutions) into ONNX Runtime format with dynamic spatial shapes.
    - Quantize to FP16 (`lama_fp16.onnx`, ~50MB) and INT8 for mobile NPU/GPU execution with high PSNR (>32 dB on complex structural textures).

- [ ] **ML-02. Implement `LamaInference.kt` Engine**
    - Create `LamaInference` implementing the `InpaintingEngine` interface.
    - Handle LaMa-specific pre-processing: pad image & mask dimensions to multiples of 8, normalize float inputs to $[0.0, 1.0]$, and unpad the output tensor.

- [ ] **ML-03. Inpainting Engine Factory & Dynamic Tier Selection**
    - Implement `InpaintingEngineFactory` capable of instantiating `MiGanInference` (ultra-fast, lightweight 512x512) or `LamaInference` (high-fidelity structural inpainting).
    - Update `DevicePerformanceHelper` to default to LaMa on high-end hardware (e.g. Pixel 8/9/10, Snapdragon 8 Gen 2/3) and MI-GAN on resource-constrained devices.

- [ ] **ML-04. Quantitative Benchmark Suite: MI-GAN vs LaMa vs SDXL**
    - Extend `BenchmarkEvaluator` to run head-to-head comparisons across:
      - Execution Latency (ms)
      - Peak Native Memory (MB)
      - Photorealism & Structural Consistency ($IoU$, SSIM, PSNR, LPIPS)
    - Output automated markdown reports in `app/build/reports/benchmarks/`.

---

## 5. Summary Statistics

| Category | Completed | Open |
|---|---|---|
| Historical Milestones & Backlog (M1–M8, Phases 1–4) | 55 | 0 |
| **Phase 6A: Product Strategy & UX Alignment** | 0 | 5 |
| **Phase 6B: Live Camera Real-Time AR (Item 1)** | 0 | 5 |
| **Phase 6C: Production Release & Play Store (Item 2)** | 0 | 4 |
| **Phase 6D: Multi-Model Inpainting / LaMa (Item 4)** | 0 | 4 |
| **Total Future Roadmap** | **0** | **18** |
