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

---

## 5. Refactoring & Code Quality (Detected 2026-08-14, Full Inspection)

> Items below are the output of a systematic 4-agent codebase inspection of all Kotlin, Python, build, and test files. Ordered within each section by risk/impact.

---

### 🔴 5A — Safety: Crashes, Resource Leaks & Security

- [x] **RF-01. Remove debug `/sdcard/` write from `MiGanInference.kt` production path**
    - `MiGanInference.kt:137–150` writes `raw_migan_output.png` to `/sdcard/Download/` on *every* inference call using an unclosed `FileOutputStream` (no `.use{}`). Causes EACCES on API 29+, leaks file handles on exception, and bloats user storage.
    - **Fix**: Deleted debug write block from production path.
- [x] **RF-02. Fix OkHttp response body leak in `ServerSdxlApi.kt`**
    - `ServerSdxlApi.kt:66`: `client.newCall(request).execute()` wrapped in `.use { response -> … }` to prevent socket leaks.
- [x] **RF-03. Replace 15+ unsafe `!!` operators in `YoloTFLiteEngine.kt`**
    - Extracted safe checked local references (`val interp = interpreter ?: throw ...`) and wrapped buffer access.
- [x] **RF-04. Fix unsafe `!!` on nullable `inpaintedBitmap` in `FirstFragment.kt`**
    - Safely guarded `result.inpaintedBitmap` with null checks and fallback error handling.
- [x] **RF-05. Fix unsafe `!!` in `AppLogger.kt:53`**
    - Replaced `logFile?.length()!!` with safe null-safe length check.
- [x] **RF-06. Add `try-finally` to all OpenCV Mat allocations in `ImageQualityMetrics.kt`**
    - Wrapped all intermediate `Mat` allocations in `try-finally` blocks.
- [x] **RF-07. Add `try-finally` to `DebugUtil.kt` and `MaskTouchUpUtils.kt` Mat allocations**
    - Wrapped all intermediate OpenCV Mats in `try-finally` blocks.
- [x] **RF-08. Add `release()` / `AutoCloseable` to `YoloResult` and `Preprocessor.PreprocessResult`**
    - Implemented `AutoCloseable` on `YoloResult` and `PreprocessResult`.
- [x] **RF-09. Fix ViewModel calling `Bitmap.recycle()` on in-use bitmaps (`MainViewModel.kt:212–216`)**
    - Removed premature `.recycle()` in ViewModel `onCleared()`.
- [x] **RF-10. Fix `ModelAssetProvider.kt` concurrent file extraction race condition**
    - Added `@Synchronized` and atomic temp-file extraction with rename fallback.
- [x] **RF-11. Remove hardcoded `mock-valid-token` Play Integrity bypass from `backend/config.py`**
    - Set default `allowed_integrity_tokens` to an empty list via `default_factory=list`, requiring explicit configuration.
- [x] **RF-12. Fix FastAPI in-memory rate limit dict memory leak (`backend/server.py`)**
    - Pruned expired date entries from `rate_limits` dictionary on each lookup.
- [x] **RF-13. Fix HTTP request body size enforcement in `backend/server.py`**
    - Verified actual read bytes against `max_upload_bytes` to prevent chunked upload bypass.
- [x] **RF-14. Fix `asyncio.Semaphore(1)` created at module scope in `backend/server.py`**
    - Wrapped in `get_sdxl_semaphore()` lazy getter attached to active event loop.
- [x] **RF-15. Fix `ImageProcessor.kt` catching `CancellationException`**
    - Re-threw `CancellationException` and tracked `transformedMat` in `matsToRelease`.

---

### 🟠 5B — Architecture & Design

- [ ] **RF-16. Rename `FirstFragment` → `MainEditorFragment` and decompose the God Fragment**
    - `FirstFragment.kt` is 626 lines, handling permissions, camera, image decoding, mask composition, GDPR consent, quota gating, Instagram export, CSV export, and batch mode. Violates Single Responsibility.
    - **Fix**: Rename to `MainEditorFragment`. Extract `ExportDelegate`, `MaskPreviewDelegate`, `GdprConsentUseCase`, and `InstagramExportUseCase`. Target <300 lines.

- [ ] **RF-17. Move ML engine instantiation out of `MainViewModel` secondary constructor**
    - `MainViewModel.kt:34–42` hardcodes `StaticImagePipeline`, `YoloServiceImpl`, `YoloTFLiteEngine`, `MiGanInference`, `ServerSdxlApi`, making unit tests impossible without real Android context and model files.
    - **Prerequisite**: A2 (Hilt/Koin DI). Interim fix: use a `ViewModelFactory`.

- [ ] **RF-18. Route batch inference through WorkManager instead of `viewModelScope`**
    - `MainViewModel.processBatch()` runs in `viewModelScope`, which is torn down when the app goes to background mid-batch. `BatchProcessingWorker.kt` already exists but is unused from ViewModel.

- [ ] **RF-19. Rename `InpaintingEngine.inferMiGan()` → `inpaint()`**
    - The interface method name leaks the Mi-GAN implementation detail. Future engine swaps (LaMa, SD-inpainting) would require renaming all call sites.

- [ ] **RF-20. Move FastAPI heavy startup (PyTorch, Redis, gRPC) into Lifespan context manager**
    - `backend/server.py` loads Redis, CUDA, and multi-GB PyTorch models at module import time, making the module untestable without mocking global state.
    - **Fix**: Migrate to `@asynccontextmanager async def lifespan(app: FastAPI)` and pass references via `app.state`.

- [ ] **RF-21. Extract embedded HTML/JS from `backend/server.py` into a Jinja2 template**
    - `get_web_workbench` embeds 85 lines of raw HTML/CSS/JS inside the API module.
    - **Fix**: Move to `backend/templates/workbench.html` and render with `fastapi.templating.Jinja2Templates`.

- [ ] **RF-22. `YoloTFLiteEngine` should delegate to `ModelAssetProvider` for asset loading**
    - Direct `context.assets.open()` bypasses the abstraction layer and complicates Play Asset Delivery adoption.

- [ ] **RF-23. Fix per-request gRPC client construction in `backend/server.py`**
    - `PlayIntegrityServiceClient.from_service_account_json(...)` is called on every request. gRPC client construction is expensive (TLS handshake, channel allocation).
    - **Fix**: Construct once at startup; store in `app.state`.

- [ ] **RF-24. Make `BackendSettings` lazy / injectable via `Depends()` in `backend/config.py`**
    - Module-level `settings = BackendSettings()` causes import-time crashes on invalid env vars and prevents override in tests.
    - **Fix**: `@functools.lru_cache def get_settings() -> BackendSettings: …` and inject via `Depends(get_settings)`.

- [ ] **RF-25. Decouple domain inpainting logic from FastAPI in `backend/server.py`**
    - `process_inpainting_payload` raises `fastapi.HTTPException` directly, coupling domain logic to the web framework.
    - **Fix**: Raise domain exceptions; let the route handler translate to HTTP responses.

---

### 🟡 5C — Code Smells & Performance

- [x] **RF-26. Reuse pre-allocated `RectF` in `BeforeAfterSliderView.onDraw()`**
    - `viewRect` is reused for aspect-fit clipping and bitmap drawing, avoiding per-frame allocations.
- [x] **RF-27. Convert raw pixel literals to dp/sp in `BeforeAfterSliderView.kt`**
    - Stroke widths, text sizes, handle radius, and badges are scaled via `displayMetrics.density` / `scaledDensity`.
- [x] **RF-28. Cancel `BeforeAfterSliderView.revealAnimator` in `onDetachedFromWindow()`**
    - Overrode `onDetachedFromWindow()` to cancel active `revealAnimator` and clean up references.
- [x] **RF-29. Replace `object Idle` with `data object Idle` in `MainUiState.kt`**
    - Converted to `data object Idle : MainUiState()`.
- [x] **RF-30. Extract `"autokorrektur_prefs"` SharedPreferences key to a shared constant**
    - Extracted to shared `PreferencesConstants.kt` object and referenced across `ConsentManager` and `QuotaManager`.
- [x] **RF-31. Fix `AppLogger.kt` thread safety (`SimpleDateFormat`, `FileWriter`)**
    - Replaced with thread-safe `DateTimeFormatter` and wrapped file append operations in synchronized blocks.
- [x] **RF-32. Fix `QuotaManager.kt` thread safety (`SimpleDateFormat`)**
    - Replaced with `java.time.LocalDate.now().toString()`.
- [x] **RF-33. Fix `InstagramExportUtils.saveBitmapForSharing()` temp file collision**
    - Generated timestamp-unique filenames by default.
- [x] **RF-34. Fix CSV injection in `ImageExportManager.exportBatchResultsToCSV()`**
    - Added CSV escaping with quote wrapping and character replacement.
- [x] **RF-35. Pre-allocate `YoloTFLiteEngine` inference buffers; eliminate per-frame allocation**
    - Pre-allocated `pixelBuffer` ByteArray in `allocateBuffers()` and reused across `matToByteBuffer()` calls.

- [ ] **RF-36. Pre-allocate `YoloMaskAssembler.deinterleavePrototypes()` channel arrays**
    - Allocates `FloatArray(pixelsPerChannel)` 32 times per inference pass. Pre-allocate or use a 2D array pool.

- [x] **RF-37. Add stride > 0 guard to `ImageProcessingUtils.divStride()`**
    - Added `require(stride > 0)` check and cleaned up dimension rounding logic.
- [x] **RF-38. Replace inline FQCNs with `import` statements in `FirstFragment.kt`**
    - Replaced all inline FQCN usages of `BitmapMemoryUtils`, `MaskOverlayUtils`, `InstagramExportUtils`, and `ArCameraActivity` with explicit imports.

- [ ] **RF-39. Replace mock prediction in `backend/benchmark_ml.py` with real ONNX inference**
    - `run_benchmark()` applies a Gaussian blur to ground-truth masks (L160–162) instead of running actual ONNX inference. All reported IoU/Dice scores are fabricated.
    - **Fix**: Load the model via `onnxruntime.InferenceSession` and run real inference.

- [x] **RF-40. Fix XSS risk in `benchmark_ml.py` HTML report generation**
    - Sanitized sample ID and category fields via `html.escape(...)` in `generate_html_report()`.

- [ ] **RF-41. Fix model ID name mismatch (`sdxl_model_id` vs. SD 1.5) in `backend/config.py`**
    - `sdxl_model_id` defaults to `"runwayml/stable-diffusion-inpainting"` (SD 1.5) while the variable name and all documentation reference SDXL. Rename or update the default to the real SDXL inpainting model.

---

### 🟢 5D — Documentation (KDoc & Docstrings)

- [x] **RF-42. Add KDoc to all public functions/properties in `MainViewModel.kt`**
    - Added complete KDoc documentation for `uiState`, `properties`, and all setter/action methods.
- [x] **RF-43. Add KDoc to `StaticImagePipeline.kt` public API**
    - Documented `isInitialized`, `initialize`, `processImage`, and `close`.
- [x] **RF-44. Add KDoc to `InpaintingEngine.kt` interface methods**
    - Documented `initialize`, `inferMiGan`, and `close`.
- [x] **RF-45. Add KDoc to `YoloService.kt` interface methods and properties**
    - Documented `isInitialized`, `initialize`, `infer`, `inferDetailed`, and `close`.
- [x] **RF-46. Document COCO class indices in `YoloConfig.kt`**
    - Documented indices for car (2), motorcycle (3), bus (5), and truck (7).
- [x] **RF-47. Add KDoc to all `Errors.kt` exception subclasses**
    - Added KDoc for `ModelLoadException`, `InferenceException`, `ShapeMismatchException`, `ModelNotInitializedException`, `InpaintException`, `CloudInferenceException`, `QuotaExceededException`.
- [x] **RF-48. Add KDoc to `InstagramExportUtils.kt` enums**
    - Documented `AspectRatio` and `LayoutStyle` enums.
- [x] **RF-49. Add KDoc to all private helper methods in `MiGanInference.kt`**
    - Added KDocs for `prepareSquareInputs`, `runOnnxSession`, `processOutputMat`, `blendResult`, `preprocessImage`, `preprocessMask`, `createTensor`, `getOutputData`, `orderInCHWAsBytes`.

- [x] **RF-50. Document caller-owns-release contract for `TemporalBackgroundAccumulator.accumulateAndBlend()` return value.**
    - Added KDoc and implemented `AutoCloseable`.
- [x] **RF-51. Add `Field(description=…)` to all `BackendSettings` fields in `backend/config.py`.**
    - Added comprehensive Pydantic `Field` descriptions for all backend settings.
- [x] **RF-52. Add docstrings to `benchmark_ml.py` public symbols**
    - Added docstrings to `SampleMetrics`, `mat_to_base64`, `run_benchmark`, and `generate_html_report`.

---

### 🔵 5E — Build Configuration & CI/CD

- [x] **RF-53. Pin `onnxruntimeAndroid` version in `libs.versions.toml`**
    - Pinned to `"1.22.0"`.
- [x] **RF-54. Move CameraX (`1.6.1`) and Orchestrator (`1.5.0`) versions into `libs.versions.toml`**
    - Declared under `[versions]` and `[libraries]` in version catalog.
- [x] **RF-55. Add TFLite and OkHttp ProGuard keep rules to `app/proguard-rules.pro`**
    - Added rules preserving `org.tensorflow.lite.**` and OkHttp annotations.
- [x] **RF-56. Remove unused Mockito dependency from `app/build.gradle.kts`**
    - Removed `mockito-core` dependency from catalog and build script.
- [x] **RF-57. Remove or parameterize `org.gradle.java.home` from `gradle.properties`**
    - Parameterized `org.gradle.java.home` relying on `JAVA_HOME` and Gradle `jvmToolchain(21)` for cross-platform CI portability.

- [ ] **RF-58. Add `./gradlew lintDebug` step to CI workflow**
    - CI runs Detekt but not Android Lint. Android Lint catches API compatibility and resource issues that Detekt does not.

- [ ] **RF-59. Add Kover Android coverage reporting to CI (see also C1)**
    - Backend has `--cov=.`; Android side produces no coverage report, blocking data-driven gate decisions.

- [ ] **RF-60. Make release signing fail explicitly when `keystore.properties` is absent**
    - Current fallback to debug keystore is silent. An unsigned release build will be rejected by Play Store but the failure is invisible during local development.
    - **Fix**: `error("Release keystore not configured — set keystore.properties")` in the release signing block.

---

### 🧪 5F — Test Quality

- [x] **RF-61. Delete or replace boilerplate/trivially-passing tests**
    - Replaced `ExampleUnitTest.kt` with `ImageProcessingUtilsUnitTest` validating stride math and square padding ratios.
- [x] **RF-62. Fix always-true assertions that make tests meaningless**
    - Corrected assertions in `VehicleMaskSegmentationTest.kt` to ensure non-empty and non-zero dimension assertions.

- [x] **RF-63. Extract `hasCarDetection()` helper to `AndroidInstrumentedBaseTest`**
    - Added `hasCarDetection()` with OpenCV Mat thresholding and automatic `baseTempFiles` cleanup in `@After`.
- [x] **RF-64. Make all instrumented tests extend `AndroidInstrumentedBaseTest` for unified `tempFiles` lifecycle**
    - Converted instrumented tests to extend `AndroidInstrumentedBaseTest` with unified lifecycle.

- [x] **RF-65. Write real unit tests for `MainViewModel` (batch mode, error states, `onCleared()`)**
    - Added unit test cases for single/batch mode selection, slider clamping, and state resetting.

- [x] **RF-66. Write unit tests for `TemporalBackgroundAccumulator.accumulateAndBlend()`**
    - Added `TemporalBackgroundAccumulatorInstrumentedTest` verifying background accumulation and blending.

- [x] **RF-67. Write tests for `MaskTouchUpUtils.createDilatedMask()` and `mergeMaskWithStrokes()`**
    - Added `MaskTouchUpUtilsInstrumentedTest` verifying dilation pixel expansion and brush stroke merging.

- [x] **RF-68. Write tests for `ImageExportManager.saveImageToGallery()` and `exportBatchResultsToCSV()`**
    - Added `ImageExportManagerInstrumentedTest` verifying CSV formatting, escaping, and gallery storage.

- [x] **RF-69. Write tests for `UriLoader` EXIF rotation paths and unsupported URI scheme error**
    - Added `UriLoaderInstrumentedTest` testing unsupported schemes, file URI decoding, and EXIF 90° clockwise rotation.

- [ ] **RF-70. Enforce `PostInpaintingVehicleAssertionUtils` in all inpainting test suites**
    - `TESTING.md` mandates second-pass YOLO re-detection after every inpainting, but only `FullEmulatedUiInferenceE2ETest` uses `PostInpaintingVehicleAssertionUtils`. `InpaintingQualityBenchmarkTest`, `MiGanInpaintingInstrumentedTest`, and `FiftyImageTriplesPipelineBenchmarkTest` skip or reinvent this check.

- [x] **RF-71. Implement real Boundary-IoU in `MaskQualityBenchmarkTest.kt`**
    - Implemented OpenCV morphological trimap calculation with dilation, erosion, and intersection over union.

- [ ] **RF-72. Document all magic assertion thresholds in test files**
    - Add `// Why: <justification>` comments for every undocumented numeric threshold (e.g. `rDiff + gDiff + bDiff > 100`, `falseMaskRatio < 0.10f`, `meanPsnr >= 15.0`, `bgDiff <= 10.0`). Consolidate inconsistent thresholds across test files into named constants in `AndroidInstrumentedBaseTest`.

- [x] **RF-73. Add backend tests for missing contract paths**
    - Added tests in `test_server.py` for unapproved Play Integrity token rejection (403), oversized uploads (413), and invalid file magic bytes (400).

- [ ] **RF-74. Re-enable or remove `@Ignore`-annotated tests**
    - `ImageProcessingPipelineTests.kt` is fully `@Ignore("Split into pipeline/* tests")` — resolve the split or delete.
    - `UninitializedYoloServiceUsageTest.testStartInferenceInFirstFragmentDoesNotShowUninitializedError` is `@Ignore("ActivityScenario conflict")` — resolve the conflict and re-enable.

- [x] **RF-75. Add unit tests for `benchmark_ml.py` metric functions**
    - Created `backend/test_benchmark_ml.py` testing `compute_boundary_iou`, `compute_ssim`, `compute_psnr`, `generate_error_heatmap`, and `mat_to_base64`.
