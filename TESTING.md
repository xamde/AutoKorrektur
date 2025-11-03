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
