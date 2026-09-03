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

> As of `docs/MVP_FEATURE_FLAG_PLAN.md`, the app builds as four product flavors
> (`core`/`plus`/`beta`/`full`), so `test`/`connected`/`lint` Gradle tasks need a flavor prefix —
> bare `testDebugUnitTest`/`connectedDebugAndroidTest`/`lintDebug` no longer resolve. The
> commands below target `full` (all features on, all ABIs) since it's the flavor that exercises
> every code path; swap in `core`/`plus`/`beta` to test a specific tier, or drop the flavor
> (`./gradlew test`, `./gradlew connectedAndroidTest`) to run all four.

* **Run Fast JVM Unit Tests**:
  ```bash
  ./gradlew :app:testFullDebugUnitTest
  ```
* **Run Static Code Analysis (Detekt)**:
  ```bash
  ./gradlew detekt
  ```
* **Run On-Device Segmentation & Inpainting Benchmark**:
  ```bash
  ./gradlew :app:connectedFullDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=de.konradvoelkel.android.autokorrektur.ml.MaskQualityBenchmarkTest,de.konradvoelkel.android.autokorrektur.ml.InpaintingQualityBenchmarkTest
  ```
* **Run Full Connected Test Suite**:
  ```bash
  ./gradlew :app:connectedFullDebugAndroidTest
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

## 8. Tier 5 (Proposed): Automated UI Quality & Localization Sweep

> **Why this tier exists**: field testing on the physical Pixel 10 Pro (2026-09) surfaced "lots
> of minor issues, too much to report" — the classic sign that the bugs aren't logic bugs (Tiers
> 1–4 already catch those extremely well) but *presentation* bugs: layout, locale, and
> accessibility regressions that only show up in specific configurations a human tester won't
> systematically sweep by hand. This tier automates that sweep instead of relying on a person to
> stumble into every locale × theme × screen-width combination.

### A. String Resource Localization Test (✅ Implemented)
`app/src/test/java/de/konradvoelkel/android/autokorrektur/StringResourceLocalizationTest.kt`

A pure-JVM test (no emulator, runs in `./gradlew :app:testFullDebugUnitTest`) that parses
`values/strings.xml`, `values-de/strings.xml`, and `values-en/strings.xml` directly and asserts
resource-resolution safety. It caught a real, pre-existing bug while being written: **51 of the
121 keys in the default (locale-neutral) `strings.xml` are German text that disagrees with the
deliberate English translation in `values-en`**, and 49 of those aren't declared in `values-de`
at all. The app currently "works" only because `values-de` and `values-en` happen to cover
complementary halves of the key space — but Google Play serves every locale by default, and any
device set to French, Spanish, Polish, etc. has no override to fall back to, so it renders a UI
that silently mixes English and German on the same screen. The test ships as a ratchet (today's
51 known offenders are allow-listed so CI stays green) so it fails the moment a *new* violation
is introduced, without demanding an immediate fix of the existing ones. See the file's doc
comment for the two remediation options (restrict store listing to de/en markets, or make
`values/strings.xml` purely English).

### B. Screenshot / Visual Regression Testing Across the Config Matrix (Proposed)
This app already ships `values-de`, `values-en`, `values-night`, `values-land`, `values-w600dp`,
and `values-w1240dp` — five independent axes a human tester samples maybe one or two
combinations of. [Paparazzi](https://github.com/cashapp/paparazzi) renders Android views/layouts
to PNGs on the JVM with no emulator (seconds, not the 20-30s of `connectedAndroidTest`), which
makes an exhaustive sweep affordable in CI.

```kotlin
// app/build.gradle.kts — add alongside the existing plugins block
plugins {
    id("app.cash.paparazzi") version "1.3.5" // check for a newer release before pinning
}
```

```kotlin
// app/src/test/java/.../screenshot/ConfigMatrixScreenshotTest.kt (sketch — verify against the
// current Paparazzi API before committing; DeviceConfig/environment parameters change between
// releases)
class ConfigMatrixScreenshotTest {
    @get:Rule
    val paparazzi = Paparazzi()

    @Test fun firstFragment_de_light_phone() { /* inflate fragment_first.xml, locale=de, night=no */ }
    @Test fun firstFragment_de_dark_phone() { /* locale=de, night=yes */ }
    @Test fun firstFragment_en_light_tablet() { /* locale=en, width=w1240dp */ }
    @Test fun arCameraActivity_de_landscape() { /* locale=de, orientation=land */ }
    // ... one test per (screen × locale × night × width-bucket) combination that matters
}
```
Golden images are checked in and diffed on every PR — a translator lengthening a German label,
someone flipping a color in `values-night`, or a landscape layout clipping a button all become a
failing CI check with an image diff attached, instead of a bug report from the field.

### C. Espresso Accessibility Checks (Proposed — one line, big payoff)
AndroidX Test ships a built-in accessibility scanner that plugs directly into existing Espresso
interactions with a single line, flagging touch targets under 48dp, missing content descriptions,
and insufficient contrast — exactly the kind of "minor issue" a sighted manual tester tends to
walk right past:
```kotlin
// in a JUnit ClassRule or @Before, shared across the existing Espresso suite:
androidx.test.espresso.accessibility.AccessibilityChecks.enable()
    .setRunChecksFromRootView(true)
```
This requires `androidTestImplementation("androidx.test.espresso:espresso-accessibility:<version>")`
and will very likely turn up findings in the existing Espresso suite (`MainActivityEspressoTest`,
`RigorousGuiFlowInstrumentedTest`, `EndToEndWorkflowsInstrumentedTest`, etc.) the first time it's
enabled — that's expected and is the point.

### D. Nightly Monkey / Fuzz Stability Pass (Proposed)
A scheduled (not per-PR — too slow and flaky for that) CI job running
`adb shell monkey -p de.konradvoelkel.android.autokorrektur -v 500 --throttle 50` against the AR
camera and Studio screens for a few minutes catches crashes from erratic/rapid tapping (e.g.
double-tapping the shutter during a 5s recording, backgrounding mid-inference) that scripted
Espresso flows, which only ever do exactly what they're told, structurally cannot find.

---

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



