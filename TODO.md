# AutoKorrektur — Project Status & Roadmap

> **Last Updated & Verified**: 2026-08-14 (post code review)
> **Status**: Core ML pipeline, segmentation, inpainting fidelity, and EXIF orientation fully verified across physical Pixel 10 Pro & emulators. 90/90 instrumented tests passing, 71/71 backend pytest passing.

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

### ✅ Prior Refactoring (RF-01 through RF-75)

All items from the previous refactoring sweep are complete, with the following corrections and exceptions noted in the new backlog below.

---

## 4. New Backlog — Code Review Findings (2026-08-14)

> Items below are the output of a systematic 4-agent code review covering all Kotlin source, Python backend, test suites, build configuration, documentation, and architecture. Ordered by severity within each section.

---

### 🔴 4A — Critical: Memory Safety & Resource Leaks

- [ ] **CR-01. Fix Bitmap leak in `BatchProcessingWorker` batch loop**
    - `BatchProcessingWorker.kt:66–77`: `pipeline.processImage(uri)` returns a `PipelineResult` containing multi-megapixel `originalBitmap`, `maskBitmap`, and `inpaintedBitmap`. None are recycled inside the loop body. Processing a batch of 20+ high-res photos will cause OOM.
    - **Fix**: Call `.recycle()` on all three bitmaps after extracting needed state (success/failure) in each loop iteration.

- [ ] **CR-02. Fix Bitmap leak on `CancellationException` in `ImageProcessor.kt`**
    - `ImageProcessor.kt:84`: `if (e is CancellationException) throw e` skips the bitmap recycle calls on lines 85–86. If the user cancels during heavy OpenCV processing, `originalBitmap` and `transformedBitmap` leak.
    - **Fix**: Move bitmap recycling into the `finally` block (alongside `matsToRelease.forEach { it.release() }`), or recycle before the re-throw.

- [ ] **CR-03. Fix remaining `!!` operators in `YoloTFLiteEngine.kt`**
    - `YoloTFLiteEngine.kt:134,293,294`: Three `pixelBuffer!!` usages remain despite RF-03 being marked complete. After the null-check on line 134, smart-cast is defeated by the mutable `var pixelBuffer` property, so `!!` is still needed.
    - **Fix**: Capture into a local `val` after the null check: `val buf = pixelBuffer ?: ...` and use `buf` thereafter. Eliminates all remaining `!!`.

- [ ] **CR-04. Fix `TemporalBackgroundAccumulator` native memory leak on missed `close()`**
    - `TemporalBackgroundAccumulator.kt:44–48`: If the host Fragment/Activity fails to call `close()`, the OpenCV `Mat` held in `backgroundMat` permanently leaks native memory outside GC reach. No `Cleaner` or defensive guard exists.
    - **Fix**: Register a `sun.misc.Cleaner` / `java.lang.ref.Cleaner` reference or add a `finalize()` safety net that logs a warning and releases the Mat.

---

### 🔴 4B — Critical: Backend Async Correctness

- [ ] **CR-05. Fix synchronous gRPC call blocking the asyncio event loop in `verify_token()`**
    - `server.py:142–159`: `verify_token()` is a synchronous function that calls `client.decode_integrity_token()`, a blocking gRPC call. It is invoked directly from the async `inpaint_image` route (line 316), stalling the entire event loop for all concurrent requests.
    - **Fix**: Wrap with `await asyncio.to_thread(verify_token, device_uuid, play_integrity_token)` in the caller, or convert `verify_token` to async using an async gRPC client.

- [ ] **CR-06. Fix unbounded memory read before upload size enforcement in `inpaint_image()`**
    - `server.py:323–337`: The pre-check on line 323 uses `image.size` which can be `None` or spoofed by the client. The actual enforcement happens only *after* `await image.read()` (line 328) has already loaded the entire file into RAM. A multi-gigabyte upload would OOM the server before the size check triggers.
    - **Fix**: Read uploads in chunks with a running byte counter and abort early when `max_upload_bytes` is exceeded, or configure an ASGI middleware (e.g., `starlette.middleware.trustedhost`) to enforce request body size limits before the handler.

---

### 🟠 4C — High: Architecture & Design

- [ ] **CR-07. Fix WorkManager `Data` 10KB size limit for batch URI lists**
    - `MainViewModel.kt:228–231`: `putStringArray(uriStrings)` passes all batch image URIs as a string array in WorkManager `Data`, which has a hard 10KB serialization limit. A batch of ~40+ images with long content URIs will throw `IllegalStateException` at enqueue time.
    - **Fix**: Store the URI list in a Room database or write to a temp JSON file, and pass only the DB row ID or file path in the WorkManager `Data`.

- [ ] **CR-08. Add coroutine cancellation checkpoints to ML pipeline**
    - `StaticImagePipeline.kt`, `ImageProcessor.kt`, `MiGanInference.kt`: No `ensureActive()` or `yield()` calls exist anywhere in the ML pipeline. When `inferenceJob?.cancel()` is called from the ViewModel, the heavy synchronous OpenCV/ONNX JNI operations continue executing until the next coroutine suspension point (which may be far away or never).
    - **Fix**: Add `currentCoroutineContext().ensureActive()` calls between major pipeline stages (after YOLO inference, before inpainting, before blending) to allow prompt cancellation.

- [ ] **CR-09. Move hardcoded SDXL inpainting prompt to `BackendSettings`**
    - `server.py:228`: The inpainting prompt `"seamless background, clean street, photorealistic"` is hardcoded. It cannot be changed without editing source code.
    - **Fix**: Add `inpainting_prompt: str = Field(default="seamless background, clean street, photorealistic", description="...")` to `BackendSettings` in `config.py`.

- [ ] **CR-10. Fix model ID naming mismatch in `backend/config.py`**
    - `config.py:30–31`: `sdxl_model_id` defaults to `"runwayml/stable-diffusion-inpainting"` (SD 1.5 model) while the variable name and all documentation reference SDXL. This was flagged as RF-41 and remains unchecked.
    - **Fix**: Either rename the field to `sd_inpainting_model_id` to match reality, or update the default to the actual SDXL inpainting model (`"diffusers/stable-diffusion-xl-1.0-inpainting-0.1"` or similar). Update all references in docstrings and `TODO.md`.

---

### 🟠 4D — High: Test Quality & Coverage

- [ ] **CR-11. Add `@After unmockkAll()` to `BatchProcessingWorkerInstrumentedTest`**
    - `BatchProcessingWorkerInstrumentedTest.kt:46`: Uses `mockkConstructor(StaticImagePipeline::class)` but has no `@After` teardown calling `unmockkAll()`. This leaks mock state into subsequent test classes in the same instrumented test run, causing cascading failures.
    - **Fix**: Add `@After fun teardown() { unmockkAll() }`.

- [ ] **CR-12. Migrate instrumented tests from `runBlocking` to `runTest`**
    - 40+ usages of `runBlocking` across instrumented tests (e.g., `VehicleMaskSegmentationTest`, `MlComponentTests`, `InpaintingQualityBenchmarkTest`, `ServerSdxlApiTest`, etc.). `runBlocking` on the instrumentation thread can deadlock if coroutines dispatch to `Dispatchers.Main`. `runTest` from `kotlinx-coroutines-test` provides proper virtual time control and deadlock prevention.
    - **Fix**: Replace all `= runBlocking { ... }` with `= runTest { ... }` and add `kotlinx-coroutines-test` to the androidTest dependencies if not already present.

- [ ] **CR-13. Write unit/instrumented tests for `ImageProcessor` core logic**
    - `ImageProcessor.kt` is the central ML preprocessing coordinator (loading URIs, scaling, color conversion, EXIF handling) but has zero direct test coverage. All testing is indirect through pipeline-level tests.
    - **Fix**: Create `ImageProcessorInstrumentedTest.kt` testing: (a) EXIF rotation normalization, (b) downscale/upscale fidelity, (c) cancellation behavior (CancellationException re-throw), (d) error handling for corrupt/missing URIs.

- [ ] **CR-14. Write unit/instrumented tests for `YoloTFLiteEngine` inference**
    - `YoloTFLiteEngine.kt` is the core TFLite inference engine but has no direct test file. Inference correctness is only validated indirectly via pipeline tests.
    - **Fix**: Create `YoloTFLiteEngineInstrumentedTest.kt` testing: (a) initialization and model loading, (b) inference on known input producing expected output shape, (c) buffer allocation and reuse, (d) `close()` resource release.

- [ ] **CR-15. Write tests for `ServerInpainter` network API contract**
    - `ServerInpainter.kt` (the HTTP-based cloud inpainting client) has no test coverage at all.
    - **Fix**: Create `ServerInpainterTest.kt` using a mock HTTP server (MockWebServer or MockK) to verify: (a) correct multipart request format, (b) error response handling (4xx, 5xx), (c) timeout behavior, (d) retry logic if any.

- [ ] **CR-16. Add `MainViewModel` coroutine state flow tests**
    - `MainViewModelTest.kt` only tests getters/setters and property clamping. The `processImage()` coroutine logic — including state transitions (`Idle → Processing → Success/Error`), cancellation, and error propagation — is completely untested.
    - **Fix**: Add unit tests using `runTest` and a mock `StaticImagePipeline` to verify: (a) state flow transitions, (b) cancellation via `clearState()`, (c) error state propagation from pipeline failures.

- [ ] **CR-17. Add dedicated `ConsentManager` unit tests**
    - `ConsentManager.kt` is only tested incidentally inside `QuotaManagerTest.kt`, violating test cohesion.
    - **Fix**: Create `ConsentManagerTest.kt` testing: (a) consent granting/revoking, (b) persistence across restarts, (c) state initialization for new installations.

- [ ] **CR-18. Add `QuotaManager` date transition/reset test**
    - `QuotaManagerTest.kt` does not verify that the daily quota actually resets at the date boundary. It only tests within a single "day".
    - **Fix**: Inject a `Clock`/`TimeProvider` into `QuotaManager` and write a test that advances time by 24 hours, asserting the quota resets from 0 to 5.

- [ ] **CR-19. Remove test execution order dependency in `MainActivityGuiRigorousTest`**
    - Test methods are prefixed sequentially (`test1_`, `test2_`, ...), indicating implicit dependency on execution order. Tests should be fully independent.
    - **Fix**: Refactor each test method to include its own setup/teardown. Remove numbered prefixes and rename to behavior-descriptive names.

- [ ] **CR-20. Add UI/integration tests for `BatchUiDelegate` and `InstagramExportDelegate`**
    - Both newly extracted delegates have zero test coverage.
    - **Fix**: Add Espresso-based UI tests or unit tests (mocking Fragment/Context dependencies) to verify: (a) dialog display and dismissal, (b) export format selection, (c) CSV generation, (d) share intent creation.

---

### 🟡 4E — Medium: Code Quality & Cleanup

- [ ] **CR-21. Remove deprecated `inferMiGan()` default method from `InpaintingEngine.kt`**
    - `InpaintingEngine.kt:31`: The `@Deprecated("Use inpaint() instead")` method `inferMiGan()` has zero callers anywhere in production or test code (verified via grep). It is dead code.
    - **Fix**: Remove the `inferMiGan` default method entirely from the interface.

- [ ] **CR-22. Remove unused `import org.opencv.core.Mat` from `FirstFragment.kt`**
    - `FirstFragment.kt:50`: Unused import detected by the architecture reviewer.
    - **Fix**: Delete the unused import line.

- [ ] **CR-23. Remove or justify `@Suppress("unused")` on `YoloPostprocessor.postprocess()`**
    - `YoloPostprocessor.kt:185`: The convenience `postprocess` wrapper is annotated `@Suppress("unused")`. If it's truly unused, remove it. If it's part of the public API, remove the suppression and document it.
    - **Fix**: Determine usage. Delete if unused; document if retained.

- [ ] **CR-24. Clean up inline FQCNs in `MainViewModel.scheduleBatchWork()`**
    - `MainViewModel.kt:230–241`: Multiple fully-qualified class names (`androidx.work.Data.Builder`, `de.konradvoelkel.android.autokorrektur.pipeline.BatchProcessingWorker.KEY_*`, `androidx.work.OneTimeWorkRequestBuilder`, `androidx.work.WorkManager`) used inline instead of imports.
    - **Fix**: Add top-level imports for `Data`, `OneTimeWorkRequestBuilder`, `WorkManager`, and `BatchProcessingWorker` and reference via basenames.

- [ ] **CR-25. Replace `ExampleUnitTest.kt` boilerplate file**
    - `ExampleUnitTest.kt` still exists in the unit test directory. According to RF-61 it was supposed to be replaced, but the file name remains as the Android Studio template default.
    - **Fix**: Rename to `ImageProcessingUtilsUnitTest.kt` or whatever its actual test content covers, to avoid confusion.

- [ ] **CR-26. Add docstrings to backend domain exception classes**
    - `server.py`: Custom exceptions `InvalidImagePayloadError`, `ImageDimensionExceededError`, `IntegrityVerificationError`, and `InpaintingDomainError` lack docstrings.
    - **Fix**: Add one-line docstrings explaining what condition each exception represents.

- [ ] **CR-27. Reduce `@Suppress` annotations on `ImageQualityMetrics.kt`**
    - `ImageQualityMetrics.kt:16`: `@Suppress("MagicNumber", "MaxLineLength", "LongMethod")` blankets the entire file. The magic numbers in SSIM/PSNR formulas are already documented with comments.
    - **Fix**: Remove `"MagicNumber"` suppression (the named constants like `SSIM_C1` already address it). Break the method if needed to remove `"LongMethod"`. Address `"MaxLineLength"` by reformatting long lines.

---

### 🟡 4F — Medium: Performance

- [ ] **CR-28. Use `Matrix` scaling instead of `Canvas.drawBitmap()` in `BitmapMemoryUtils.kt`**
    - `BitmapMemoryUtils.kt:46–56`: `createScaledBitmapForDisplay()` creates a `Canvas` and uses `drawBitmap()` for scaling, which is not hardware-accelerated during background loading. `Bitmap.createBitmap(src, 0, 0, w, h, matrix, true)` is more efficient.
    - **Fix**: Replace with `Matrix`-based `Bitmap.createBitmap()` scaling.

- [ ] **CR-29. Pre-allocate `ArrayList<Mat>` in `YoloMaskAssembler.deinterleavePrototypes()`**
    - `YoloMaskAssembler.kt:60–70`: While `channelBuffer` is reused, `ArrayList<Mat>(numPrototypesChannels)` and multiple `Mat` instances are still re-allocated per frame. In real-time AR contexts this creates GC pressure.
    - **Fix**: Pre-allocate the Mat list once in the constructor and overwrite contents on each call using `Mat.setTo()` and `Mat.copyTo()`.

---

### 🔵 4G — Build, CI & DevOps

- [ ] **CR-30. Add GitHub Actions CI workflow file**
    - No `.github/workflows/` directory or CI pipeline definition exists. All testing is manual.
    - **Fix**: Create `.github/workflows/ci.yml` with: (a) `uv run pytest` for backend, (b) `./gradlew lintDebug testDebugUnitTest` for Android, (c) optional emulator-based `connectedDebugAndroidTest` step using `reactivecircus/android-emulator-runner`.

- [ ] **CR-31. Add pre-commit hooks for lint and formatting**
    - No `.pre-commit-config.yaml` or git hooks exist.
    - **Fix**: Add pre-commit config with: (a) `ruff check` and `ruff format` for Python, (b) `ktlint` for Kotlin, (c) trailing whitespace and end-of-file fixers.

- [ ] **CR-32. Generate and publish code coverage reports**
    - Code coverage is *enabled* in Gradle (RF-59/C1), but no task or CI step exists to actually generate or aggregate the JaCoCo reports.
    - **Fix**: Add a `jacocoTestReport` task to `app/build.gradle.kts` and include it in the CI pipeline. Optionally upload to Codecov or similar.

- [ ] **CR-33. Add `CHANGELOG.md`**
    - No changelog exists. Version history is only tracked in `TODO.md` milestones.
    - **Fix**: Create `CHANGELOG.md` following Keep a Changelog format, retroactively documenting at least the major milestones (M1–M8) and the inpainting bug fix.

---

### 🟢 4H — Documentation Gaps

- [ ] **CR-34. Add KDoc to `MainActivity.kt` public methods**
    - `MainActivity.kt` entry point lacks KDoc on `onCreate()` configuration, menu handling, and navigation setup.
    - **Fix**: Add KDoc documenting the activity's role, navigation graph setup, and OpenCV initialization.

- [ ] **CR-35. Add KDoc to `ArCameraActivity.kt`**
    - The AR camera activity has no documentation.
    - **Fix**: Document the activity's purpose, CameraX lifecycle binding, and real-time inference flow.

- [ ] **CR-36. Add KDoc to `BatchProcessingWorker.kt`**
    - The WorkManager worker lacks class-level and method-level documentation.
    - **Fix**: Add KDoc documenting: (a) the worker's purpose, (b) expected input data keys, (c) progress reporting contract, (d) output data keys.

- [ ] **CR-37. Add KDoc to `DevicePerformanceHelper.kt`**
    - No documentation on what performance classification is used for or how it affects ML pipeline configuration.
    - **Fix**: Document the class purpose, performance tiers, and how results influence model selection.

- [ ] **CR-38. Document mask polarity convention in a central `ARCHITECTURE.md`**
    - The critical mask polarity convention (0 = car/hole, 255 = background) is documented only in scattered KDoc. A wrong assumption here caused the major inpainting bug (M3/B20).
    - **Fix**: Create `ARCHITECTURE.md` documenting: (a) mask polarity convention with diagram, (b) data flow from YOLO → MaskAssembler → GuidedFilter → MiGAN → Blending, (c) coordinate system conventions, (d) color space conventions (RGB vs RGBA).

---

## 5. Summary Statistics

| Category | Open | Severity |
|---|---|---|
| 4A — Memory Safety | 4 | 🔴 Critical |
| 4B — Backend Async | 2 | 🔴 Critical |
| 4C — Architecture | 4 | 🟠 High |
| 4D — Test Coverage | 10 | 🟠 High |
| 4E — Code Cleanup | 7 | 🟡 Medium |
| 4F — Performance | 2 | 🟡 Medium |
| 4G — Build/CI | 4 | 🔵 Low |
| 4H — Documentation | 5 | 🟢 Low |
| **Total** | **38** | |

### Recommended Execution Order

1. **First**: CR-01 through CR-06 (Critical memory/async bugs — risk of crashes and data loss)
2. **Second**: CR-07, CR-08, CR-11 (High-impact architecture and test isolation fixes)
3. **Third**: CR-12 through CR-20 (Test coverage expansion)
4. **Fourth**: CR-21 through CR-29 (Code quality and performance polish)
5. **Last**: CR-30 through CR-38 (CI/DevOps and documentation — no runtime impact)
