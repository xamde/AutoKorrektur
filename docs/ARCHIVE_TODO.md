# 🏛️ AutoKorrektur: Completed Development Milestones Archive

This document archives all historically completed and verified development milestones, bug fixes, and architectural implementations for AutoKorrektur up to Version 1.0.0.

---

## 1. Core ML & Pipeline Milestones (Verified in v1.0.0)

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
  - 100% pass on 71 pytest unit/contract tests, ruff linting, and mypy static typing with memory-only GDPR guarantees.

---

## 2. Hardening & Performance Phases (Completed & Pushed)

### ✅ Phase 1: Ship-Blocking Stability & Memory
- [x] **C1 (Video PTS Timestamps)**: Switched `VideoEncoder.kt` to buffer mode with exact `presentationTimeUs` to guarantee constant 30 FPS playback.
- [x] **C4 (OpenCV Mat Leak)**: Wrapped all intermediate native Mat allocations in `MiGanInference.kt` in `try/finally` blocks.
- [x] **C5 (Native Pointer Safety)**: Replaced dangerous `finalize()` in `TemporalBackgroundAccumulator.kt` with deterministic `AutoCloseable.close()`.
- [x] **C6 (GDPR Zero-Storage)**: Set `spool_max_size = 15MB` in `backend/server.py` so uploads stay in volatile RAM and never touch `/tmp` disk.
- [x] **C7 (AR GC Thrashing)**: Pre-allocated reusable bitmap buffer via `AtomicReference` in `ArCameraActivity.kt` to eliminate 30 FPS allocations.
- [x] **H4 (Bitmap Leaks)**: Added explicit `recycle()` calls across `FirstFragment.kt` and `StaticImagePipeline.kt`.
- [x] **H5 (ANR Protection)**: Offloaded batch queue file I/O to `Dispatchers.IO` in `MainViewModel.kt`.
- [x] **H6 (Thread Safety)**: Synchronized camera frame access using `AtomicReference<Bitmap?>`.
- [x] **H8 & H9**: Added `MediaCodec` resource leak guards and fixed Redis authentication in `docker-compose.yml`.
- [x] **M2 & M3**: Enforced Material Design 48dp minimum touch targets across all layouts and fixed Android 12+ exported activity intent rules.

### ✅ Phase 2: Quality, I18n & Performance
- [x] **M1 (I18n / Localized Strings)**: Fully externalized all UI strings into default German and English.
- [x] **C2 (Video Decoding)**: Replaced slow `MediaMetadataRetriever` extraction with high-speed `MediaExtractor` + `MediaCodec` sequential decoding in `VideoInpaintProcessor.kt`.
- [x] **C3 (Temporal AR Consistency)**: Implemented persistent temporal background plate accumulation in `TemporalBackgroundAccumulator.kt`.
- [x] **H1–H3 (Backend Security)**: Fixed rate-limiting bypass (`X-Forwarded-For`), added Play Integrity `/v1/nonce` validation, and installed early ASGI `413 Payload Too Large` DoS protection.
- [x] **H7 & M6 (ML Acceleration)**: Memory-mapped TFLite model loading via `AssetFileDescriptor` and replaced per-pixel normalization with native OpenCV C++ SIMD `convertTo(CV_32FC3, 1.0/255.0)`.
- [x] **M11 (APK Footprint)**: Removed ~150 MB of dead/unused ONNX weights from `assets/model/`.

### ✅ Phase 3: Architecture, Testing & CI/CD
- [x] **L1 (Dynamic Versioning)**: Integrated configuration-cache-safe Git commit count and tag resolution providers in `app/build.gradle.kts`.
- [x] **L4 (CI/CD Pipeline)**: Added caching (`astral-sh/setup-uv`, `gradle/actions/setup-gradle`) and automated test/coverage/AAB artifact uploads in `.github/workflows/ci.yml`.
- [x] **L6 & L7 (Backend Production Readiness)**: Configured FastAPI `CORSMiddleware` and added healthcheck service blocks for Redis and backend in `backend/docker-compose.yml`.
- [x] **L8 (Privacy by Design)**: Cleaned `data_extraction_rules.xml` to explicitly exclude private cache and ML weights from cloud backup.
- [x] **M7 (Tile Inpainting Optimization)**: Cleaned `ProgressiveTileInpainter.kt` `createFeatheredMask` by removing dead Mat allocations.
- [x] **L3 (Unit Test Suite Expansion)**: Added academic validation tests verifying resolution modes, shadow expansion, pedestrian protection, and boundary continuity.
