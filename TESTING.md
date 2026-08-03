# Testing guide

This document describes the current test structure and how to run tests quickly.

## Layout

- JVM unit tests (no Android dependencies):
    - `app/src/test/java/...`
    - Shared JVM helpers:
      `app/src/test/java/de/konradvoelkel/android/autokorrektur/shared/JvmTestUtils.kt`
- Instrumented Android tests (run on device/emulator):
    - `app/src/androidTest/java/...`
    - Shared Android test helpers:
      `app/src/androidTest/java/de/konradvoelkel/android/autokorrektur/shared/AndroidTestUtils.kt`
    - Base class for common setup: `AndroidInstrumentedBaseTest`

## Conventions

- File naming:
    - JVM tests: `FooBarTest`
    - Instrumented tests: `FooBarInstrumentedTest` (or keep historical names; annotate with
      `@SmallTest`, `@MediumTest`, `@LargeTest` when applicable)
- Packages grouped by domain: `ml/`, `image/`, `ui/`, `pipeline/`.

## Running tests

- Fast JVM tests only:
    - `./gradlew :app:testDebugUnitTest`
- Full suite (JVM + instrumented):
    - `./gradlew :app:testDebugUnitTest :app:connectedDebugAndroidTest`
- Static Analysis (Detekt):
    - `./gradlew detekt` (uses `app/detekt-baseline.xml`)
    - `./gradlew detektBaseline` (re-generates baseline XML)

## 3. Physical Device Hardware Matrix & Exception Fallback Policy

- **Emulator vs Physical SoC Driver Discrepancy**: x86_64 emulators often bypass or software-emulate NNAPI delegates. Real ARM64 hardware (e.g. Tensor SoC on Pixel devices) must be tested to ensure native driver delegate failures fall back seamlessly to CPU execution without throwing uncaught exceptions during `Interpreter` or `OrtSession` construction.
- **UI Error Guard Policy**: All Espresso UI test suites must enforce `onView(withId(com.google.android.material.R.id.snackbar_text)).check(doesNotExist())` across all activity launch and interaction tests to catch any startup initialization errors or unhandled exceptions before release.

## Current utilities

- Android-only helpers:
    - `AndroidTestUtils.initOpenCV()` — initializes OpenCV once per class (used by
      `AndroidInstrumentedBaseTest`).
    - `AndroidTestUtils.copyAssetToCache(context, file)` — copies an asset into the app's cache dir.
- JVM-only helpers:
    - `JvmTestUtils.deterministicDoubles()` — a simple deterministic RNG iterator for stable tests.
    - `JvmTestUtils.approxEquals(a, b, eps)` — float approximate comparison.

## Notes

- Some pure image-processing logic has been extracted to `ml/ImageProcessingUtils.kt` to allow JVM
  tests (e.g., `divStride`).
- Avoid Android framework types in JVM tests; keep those checks in `androidTest`.

## Backend Testing

To test the Server SDXL Premium Edit functionality locally:
1. Navigate to the root directory and create a Python virtual environment using `uv`:
   ```bash
   uv venv .venv
   uv pip install --python .venv/bin/python -r backend/requirements.txt
   ```
2. Run backend pytest unit tests:
   ```bash
   PYTHONPATH=. .venv/bin/pytest backend/test_server.py
   ```
3. Start the FastAPI server locally:
   ```bash
   .venv/bin/uvicorn backend.server:app --host 127.0.0.1 --port 8000
   ```
4. Update `serverUrl` in `app/src/main/java/de/konradvoelkel/android/autokorrektur/ml/api/ServerSdxlApi.kt` if you are testing on a physical device. For emulator testing, `10.0.2.2:8000` is already configured.

