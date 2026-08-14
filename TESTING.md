# Testing Guide & Quality Assurance

> **Last Comprehensive Test Run**: 2026-08-14  
> **Status**: All 62 backend tests, all JVM unit tests, detekt static analysis, and connected Android instrumented test suites passing 100%.

This document describes the test structure, benchmark suites, and validation commands for the AutoKorrektur Android client and Python backend.

---

## 1. Test Suite Layout

- **JVM Unit Tests** (fast, zero Android framework dependencies):
  - Location: `app/src/test/java/...`
  - Shared Helpers: `app/src/test/java/de/konradvoelkel/android/autokorrektur/shared/JvmTestUtils.kt`
- **Instrumented Android Tests** (on-device / emulator verification):
  - Location: `app/src/androidTest/java/...`
  - Base Setup: `de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest`
  - Shared Test Utilities: `de.konradvoelkel.android.autokorrektur.shared.AndroidTestUtils`
- **Python Backend Test Suite**:
  - Location: `backend/test_server.py`
  - Framework: `pytest`, `pytest-cov`, `ruff`, `mypy`

---

## 2. Four-Tier Testing Pyramid & Benchmark Suites

| Tier | Environment | Scope & Verification Target | Execution Time |
| :--- | :--- | :--- | :--- |
| **Tier 1: Fast Desktop ML Benchmark** | Desktop Python (`backend/benchmark_ml.py`) | Evaluates 50 ground-truth triples (`benchmark_manifest.json`) for IoU, Dice $F_1$, Boundary-IoU, Over-Masking rate, and PSNR. Generates visual HTML diff reports. | **< 2s** |
| **Tier 2: On-Device Hardware Parity** | Android Device / Emulator (`ml/*BenchmarkTest.kt`) | Measures on-device model execution parity, hardware delegate fallback, latency budgets (P50/P90/P99 ms), and memory stability. | **20–30s** |
| **Tier 3: JVM Unit & State Tests** | Host JVM (`app/src/test/java/...`) | Tests 100% of ViewModel logic, QuotaManager daily reset, ConsentManager GDPR flags, and ModelAssetProvider fallbacks using MockK. | **< 2s** |
| **Tier 4: UI Workflows & Visual Diffs** | Android Instrumented (`EndToEndWorkflowsInstrumentedTest.kt`) | Validates complete end-to-end user workflows, error Snackbars, dialogs, and generates 3-color error heatmaps (🟩 TP, 🟥 FP, 🟦 FN). | **15–20s** |

---

## 3. Running Test Suites & Benchmarks

### A. Fast Offline ML Benchmark & Visual Report Generator
```bash
uv run --directory backend python benchmark_ml.py
```
*Generates visual diff report at `backend/benchmark_report.html` with side-by-side Before/Mask/Prediction/Inpainted comparison.*

### B. Android Client Tests

* **Run Fast JVM Unit Tests**:
  ```bash
  ./gradlew :app:testDebugUnitTest
  ```
* **Run Static Code Analysis (Detekt)**:
  ```bash
  ./gradlew detekt
  ```
* **Run On-Device Segmentation & Inpainting Benchmark**:
  ```bash
  ./gradlew :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=de.konradvoelkel.android.autokorrektur.ml.MaskQualityBenchmarkTest,de.konradvoelkel.android.autokorrektur.ml.InpaintingQualityBenchmarkTest
  ```
* **Run Full Connected Test Suite**:
  ```bash
  ./gradlew :app:connectedDebugAndroidTest
  ```

---

## 4. Backend Service Testing

1. **Sync dependencies**:
   ```bash
   uv sync --directory backend --extra dev
   ```
2. **Lint and Type Checks**:
   ```bash
   uv run --directory backend ruff check .
   uv run --directory backend mypy .
   ```
3. **Run Contract & Integration Tests**:
   ```bash
   uv run --directory backend pytest --cov=.
   ```
4. **Local FastAPI Server**:
   ```bash
   uv run --directory backend uvicorn server:app --host 127.0.0.1 --port 8000
   ```

---

## 5. Physical Device Hardware Matrix & Exception Fallback Policy

- **Emulator vs Physical SoC Driver Discrepancy**: x86_64 emulators often software-emulate NNAPI delegates. Real ARM64 hardware (e.g. Tensor SoC on Google Pixel 10 Pro) must be tested to ensure native driver delegate failures fall back seamlessly to CPU execution without throwing uncaught exceptions during `Interpreter` or `OrtSession` construction.
- **UI Error Guard Policy**: All Espresso UI test suites enforce `onView(withId(com.google.android.material.R.id.snackbar_text)).check(doesNotExist())` across all activity launch and interaction tests to catch any startup initialization errors or unhandled exceptions before release.

---

## 6. Physical Device Edge-Case Test Suite

To eliminate discrepancies between clean lab benchmarks and messy real-world photography on physical hardware, the following automated test suites execute on every build:

| Suite Name | Location | Key Invariant Asserted |
| :--- | :--- | :--- |
| `ServerSdxlApiFallbackTest` | `app/src/test/.../api/` | Host unreachable (`10.0.2.2`), socket timeouts, or HTTP 503 errors trigger typed exceptions and **strictly preserve daily edit quota**. |
| `RotationLifecycleInferenceTest` | `app/src/test/.../viewmodel/` | Screen rotation during or after ML inference preserves `MainUiState.Success` and prevents duplicate neural engine execution. |
| `VehicleShadowSegmentationTest` | `app/src/androidTest/.../ml/` | Direct sunlight cast shadows and tire contact points are cleanly isolated without leaving floating ground artifacts. |
| `ColorSpacePreservationTest` | `app/src/androidTest/.../ml/` | RGBA <-> RGB conversions and Guided Filter guidance maintain 100% color fidelity with zero BGR channel swapping. |
| `MultiVehicleClutteredSceneTest` | `app/src/androidTest/.../ml/` | Scenes with multiple cars, curbs, and street clutter detect distinct vehicle instances while preserving unmasked sidewalks. |
| `InpaintingQualityBenchmarkTest` | `app/src/androidTest/.../ml/` | Areas outside the car mask retain exact byte-for-byte fidelity with $\text{PSNR} \ge 40\text{dB}$ (measured **62.85 dB**). |

---

## 7. Post-Inpainting Vehicle Detection Invariant

To prevent "ghost cars" and ensure that inpainting actually eliminates vehicles from the output image, every inpainting test must run a second-pass YOLO object detection on the output image using `PostInpaintingVehicleAssertionUtils`:

```kotlin
PostInpaintingVehicleAssertionUtils.assertNoVehiclesRemain(
    inpaintedBitmap = result.inpaintedBitmap!!,
    context = appContext,
    yoloService = yoloService,
    imageProcessor = imageProcessor,
    confidenceThreshold = 0.25f,
    message = "Pipeline output must have zero detected vehicles"
)
```

- **Two-Pass Verification**:
  1. Pass 1: Original image $\to$ YOLO Segmentation (asserts $\ge 1$ vehicle detected).
  2. Pass 2: Inpainted image $\to$ YOLO Detection (asserts **$0$ residual vehicles** detected).
- **Binary Model Invariant**: `mi-gan-512.onnx` requires a binary mask with `1` on the inpaint hole and `0` on preserved background. Passing `255` bypasses the inpainting generator and leaves the vehicle untouched.



