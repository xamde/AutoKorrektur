# AutoKorrektur — Project Status & Roadmap

> **Last Updated & Verified**: 2026-08-14 (post code review execution)
> **Status**: Core ML pipeline, segmentation, inpainting fidelity, EXIF orientation, memory safety, test coverage, and CI/CD fully completed and verified across physical Pixel 10 Pro & emulators.

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

---

## 4. Backlog Items (All Completed & Verified 2026-08-14)

---

### 🔴 4A — Critical: Memory Safety & Resource Leaks

- [x] **CR-01. Fix Bitmap leak in `BatchProcessingWorker` batch loop**
    - Recycled `result.originalBitmap`, `result.maskBitmap`, and `result.inpaintedBitmap` in `finally` block of batch iteration loop.
- [x] **CR-02. Fix Bitmap leak on `CancellationException` in `ImageProcessor.kt`**
    - Guaranteed `.recycle()` on `originalBitmap` and `transformedBitmap` prior to re-throwing `CancellationException`.
- [x] **CR-03. Fix remaining `!!` operators in `YoloTFLiteEngine.kt`**
    - Replaced all `pixelBuffer!!` usages with local immutable references.
- [x] **CR-04. Fix `TemporalBackgroundAccumulator` native memory leak on missed `close()`**
    - Added `finalize()` safety net that warns and releases unclosed `backgroundMat`.

---

### 🔴 4B — Critical: Backend Async Correctness

- [x] **CR-05. Fix synchronous gRPC call blocking the asyncio event loop in `verify_token()`**
    - Wrapped `verify_token(...)` in `await asyncio.to_thread(...)` in `backend/server.py`.
- [x] **CR-06. Fix unbounded memory read before upload size enforcement in `inpaint_image()`**
    - Implemented chunked 64KB streaming read with strict cumulative size enforcement aborting with 413.

---

### 🟠 4C — High: Architecture & Design

- [x] **CR-07. Fix WorkManager `Data` 10KB size limit for batch URI lists**
    - Implemented temp JSON file queue in `cacheDir` via `KEY_IMAGE_URIS_FILE` with automated cleanup in worker.
- [x] **CR-08. Add coroutine cancellation checkpoints to ML pipeline**
    - Inserted `currentCoroutineContext().ensureActive()` across preprocessing, YOLO segmentation, mask generation, inpainting, and blending stages in `StaticImagePipeline.kt`.
- [x] **CR-09. Move hardcoded SDXL inpainting prompt to `BackendSettings`**
    - Added `inpainting_prompt` setting in `backend/config.py` with custom prompt support.
- [x] **CR-10. Fix model ID naming mismatch in `backend/config.py`**
    - Standardized configuration setting name to `sd_model_id`.

---

### 🟠 4D — High: Test Quality & Coverage

- [x] **CR-11. Add `@After unmockkAll()` to `BatchProcessingWorkerInstrumentedTest`**
    - Added `@After fun tearDown() { unmockkAll() }` to eliminate mock state leakage.
- [x] **CR-12. Add `kotlinx.coroutines.test` to androidTestImplementation**
    - Added `androidTestImplementation(libs.kotlinx.coroutines.test)` to `app/build.gradle.kts`.
- [x] **CR-13. Write unit/instrumented tests for `ImageProcessor` core logic**
    - Created `ImageProcessorInstrumentedTest.kt` testing URI loading, scaling, and error handling.
- [x] **CR-14. Write unit/instrumented tests for `YoloTFLiteEngine` inference**
    - Created `YoloTFLiteEngineInstrumentedTest.kt` validating model initialization and tensor output shapes.
- [x] **CR-15. Write tests for `ServerInpainter` network API contract**
    - Created `ServerInpainterTest.kt` verifying multipart payload parsing and HTTP 200/400 handling.
- [x] **CR-16. Add `MainViewModel` coroutine state flow tests**
    - Expanded `MainViewModelTest.kt` with `runTest` state machine tests (`Success`, `Error`, `Idle`).
- [x] **CR-17. Add dedicated `ConsentManager` unit tests**
    - Created `ConsentManagerTest.kt` testing GDPR consent preference persistence.
- [x] **CR-18. Add `QuotaManager` date transition/reset test**
    - Injected customizable date provider and verified 24h quota reset in `QuotaManagerTest.kt`.
- [x] **CR-19. Remove test execution order dependency in `MainActivityGuiRigorousTest`**
    - Cleaned up numbered method prefixes and decoupled individual test cases.
- [x] **CR-20. Add UI/integration tests for `BatchUiDelegate` and `InstagramExportDelegate`**
    - Created `UiDelegateInstrumentedTest.kt` verifying dialog interactions.

---

### 🟡 4E — Medium: Code Quality & Cleanup

- [x] **CR-21. Remove deprecated `inferMiGan()` default method from `InpaintingEngine.kt`**
    - Removed unused deprecated alias.
- [x] **CR-22. Remove unused `import org.opencv.core.Mat` from `FirstFragment.kt`**
    - Cleaned up unused import.
- [x] **CR-23. Document `YoloPostprocessor.postprocess()`**
    - Added full KDoc and removed `@Suppress("unused")`.
- [x] **CR-24. Clean up inline FQCNs in `MainViewModel.scheduleBatchWork()`**
    - Replaced with clean top-level imports.
- [x] **CR-25. Replace `ExampleUnitTest.kt` boilerplate file**
    - Renamed to `ImageProcessingUtilsUnitTest.kt`.
- [x] **CR-26. Add docstrings to backend domain exception classes**
    - Added comprehensive docstrings to `InvalidImagePayloadError`, `ImageDimensionExceededError`, and `IntegrityVerificationError`.
- [x] **CR-27. Reduce `@Suppress` annotations on `ImageQualityMetrics.kt`**
    - Removed broad `MagicNumber` suppression.

---

### 🟡 4F — Medium: Performance

- [x] **CR-28. Use `Matrix` scaling instead of `Canvas.drawBitmap()` in `BitmapMemoryUtils.kt`**
    - Replaced with hardware-accelerated `Bitmap.createBitmap(src, 0, 0, w, h, matrix, true)`.
- [x] **CR-29. Pre-allocate `ThreadLocal` channel buffers in `YoloMaskAssembler.kt`**
    - Eliminated per-frame float array allocations.

---

### 🔵 4G — Build, CI & DevOps

- [x] **CR-30. Add GitHub Actions CI workflow file**
    - Created `.github/workflows/ci.yml` running backend pytest, Android lint, unit tests, JaCoCo, and emulator tests.
- [x] **CR-31. Add pre-commit hooks for lint and formatting**
    - Created `.pre-commit-config.yaml` with Ruff and yaml/json validators.
- [x] **CR-32. Generate and publish code coverage reports**
    - Added `jacoco` plugin and configured `jacocoTestReport` task.
- [x] **CR-33. Add `CHANGELOG.md`**
    - Created `CHANGELOG.md` following Keep a Changelog standard.

---

### 🟢 4H — Documentation Gaps

- [x] **CR-34. Add KDoc to `MainActivity.kt` public methods**
- [x] **CR-35. Add KDoc to `ArCameraActivity.kt`**
- [x] **CR-36. Add KDoc to `BatchProcessingWorker.kt`**
- [x] **CR-37. Add KDoc to `DevicePerformanceHelper.kt`**
- [x] **CR-38. Document mask polarity convention in `ARCHITECTURE.md`**
    - Created comprehensive system design reference.

---

## 5. Summary Statistics

| Category | Completed | Remaining |
|---|---|---|
| 4A — Memory Safety | 4 | 0 |
| 4B — Backend Async | 2 | 0 |
| 4C — Architecture | 4 | 0 |
| 4D — Test Coverage | 10 | 0 |
| 4E — Code Cleanup | 7 | 0 |
| 4F — Performance | 2 | 0 |
| 4G — Build/CI | 4 | 0 |
| 4H — Documentation | 5 | 0 |
| **Total** | **38** | **0** |
