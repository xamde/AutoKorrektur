# AutoKorrektur - Copilot Instructions

## Repository Overview

AutoKorrektur is an Android application that automatically removes cars from photographs using machine learning. This is a reimplementation of the [AutoKorrektur Web Version](https://github.com/BenB2/AutoKorrektur), based on Till Schellscheidt's Bachelor Thesis "Autokorrektur - Automatisierte Objektersetzung in Fotos".

The app's purpose is to help users visualize cities with fewer cars using a hybrid architecture: fast 100% on-device processing by default, with opt-in Server SDXL processing for ultra high-resolution results.

## Technology Stack

- **Language**: Kotlin
- **Platform**: Android (minSdk 29, targetSdk 36)
- **Build System**: Gradle with Kotlin DSL
- **Java Version**: JDK 21 (`jvmToolchain(21)`)
- **Key Libraries**:
    - AndroidX Core, AppCompat, ConstraintLayout
    - Material Design Components
    - Navigation Component
    - ONNX Runtime Android (with NNAPI EP support)
    - TensorFlow Lite (with NNAPI Delegate support)
    - OpenCV for Android
    - OkHttp 4.x (for server SDXL API communication)

### Machine Learning Models

- **Instance Segmentation**: YOLOv11-seg (`yolo11n-seg` / `yolo11s-seg`) for vehicle detection
- **Inpainting (Local)**: MI-GAN (`mi-gan-512.onnx`) for local on-device inpainting
- **Inpainting (Server Opt-In)**: SDXL via FastAPI backend (`backend/server.py`) for high-res inpainting

## Project Structure

```
AutoKorrektur/
├── app/
│   ├── src/
│   │   ├── main/java/de/konradvoelkel/android/autokorrektur/
│   │   │   ├── MainActivity.kt - App entry point
│   │   │   ├── FirstFragment.kt - Primary UI fragment for single & batch image processing
│   │   │   ├── pipeline/
│   │   │   │   └── StaticImagePipeline.kt - Decoupled ML execution pipeline
│   │   │   ├── ml/ - Machine learning components
│   │   │   │   ├── ImageProcessor.kt - Pre/post processing utilities
│   │   │   │   ├── MiGanInference.kt - MI-GAN ONNX model engine
│   │   │   │   └── api/ - YOLO services & ServerSdxlApi
│   │   │   └── utils/ - Device performance, EXIF rotation & logging utilities
│   │   ├── androidTest/ - Instrumented tests (Espresso & MediaStore tests)
│   │   └── test/ - JVM unit tests
│   ├── build.gradle.kts - App build configuration
│   └── proguard-rules.pro - ProGuard & R8 rules
├── backend/
│   ├── server.py - FastAPI SDXL server endpoint
│   ├── Dockerfile - Production deployment container
│   └── requirements.txt - Python dependencies
└── TESTING.md - Test execution guide
```

## Build and Test

The app builds as four product flavors (`core`/`plus`/`beta`/`full`, see
`docs/MVP_FEATURE_FLAG_PLAN.md`) — most Gradle tasks need a flavor prefix. The examples below use
`full` (every feature on, all ABIs — the pre-flavor app, and the dev/CI baseline) and `core`
(the Play Store candidate) as needed; drop the flavor name (e.g. `./gradlew test`) to run all four.

### Building the Project

```bash
# Debug build
./gradlew assembleFullDebug

# Release build (R8 minification & resource shrinking enabled)
./gradlew assembleCoreRelease

# Release Android App Bundle (.aab) — core is the Play Store flavor
./gradlew bundleCoreRelease
```

### Running Tests & Static Analysis

```bash
# Fast JVM unit tests
./gradlew :app:testFullDebugUnitTest

# Static code analysis (Detekt)
./gradlew detekt

# Re-generate Detekt baseline
./gradlew detektBaseline

# Instrumented Android tests
./gradlew :app:connectedFullDebugAndroidTest
```

## Code Style & Static Analysis

- Follow official Kotlin code style (`kotlin.code.style=official` in `gradle.properties`)
- Use View Binding for UI components
- Enforce clean Detekt static analysis rules (`./gradlew detekt` using `app/detekt-baseline.xml`)

## Features & Orientation Support

- **Full Orientation Support**: Supports both portrait and landscape photographs with automatic EXIF rotation correction.
- **Memory Safety**: Display Bitmaps clamped to screen bounds to prevent OOM errors on high-megapixel camera photos.
- **Interactive Before/After Slider**: Touch and swipe to compare original vs car-free results.
- **Instagram Comparison Export**: Generates 1:1, 4:5, and 9:16 comparison graphics.

## Licensing

**Critical**: This project is licensed under **GNU AGPLv3** due to the YOLOv11-seg model dependency.
Any code changes must comply with AGPLv3 terms:
- All derivative works must also be licensed under AGPLv3.
- Source code must be made available.
