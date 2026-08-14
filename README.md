# AutoKorrektur

We re-implemented the [AutoKorrektur Web Version](https://github.com/BenB2/AutoKorrektur) in Android.
This web version was based on the Bachelor Thesis "Autokorrektur - Automatisierte Objektersetzung in Fotos" by Till Schellscheidt.

<table>
  <tr>
    <td><img src="media/image_1_with_car_640x640.png" alt="Example before processing" width="400"/></td>
    <td><img src="media/image_1_without_car_640x640.png" alt="Example after processing" width="400"/></td>
  </tr>
</table>

This application is intended to remove cars from pictures to make it easier to imagine a world in which the most dangerous animal in cities (cars) is less prevalent.
Default processing is done 100% on-device. Usage is free.

## Features

* **Hybrid Inference**: Supports both 100% on-device processing (default) and opt-in Premium Server processing.
* **On-Device Inference**: Fast local processing using YOLOv11-seg and MI-GAN.
* **Premium Cloud Inference**: Secure, memory-only cloud processing via a FastAPI backend using SDXL for photorealistic results.
* **Orientation Support**: Full support for both portrait and landscape mode photos with automatic EXIF rotation correction.
* **Interactive Mask Touch-Up**: Paint brush and eraser tool for tweaking mask boundaries before inpainting.

## Tech Stack

* **Android Client**: Kotlin, ONNX Runtime, OpenCV, AndroidX, Material Design.
* **Python Backend**: FastAPI, Uvicorn, Python-Multipart, Pytest. See [backend/README.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/backend/README.md).
* **Models**: 
  * Instance Segmentation: **YOLOv11-seg** (Local)
  * Inpainting: **MI-GAN** (Local) and **SDXL** (Server)

## Development & Testing

> **Last Verified**: 2026-08-14 (100% test pass on JVM unit tests, detekt, connected Android benchmarks, and backend pytest suite).

See [TESTING.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/TESTING.md) for full testing instructions.

* **Build Debug APK**:
  ```bash
  ./gradlew assembleDebug
  ```
* **Run Android Unit Tests**:
  ```bash
  ./gradlew testDebugUnitTest
  ```
* **Run Connected Benchmark Tests**:
  ```bash
  ./gradlew :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=de.konradvoelkel.android.autokorrektur.ml.MaskQualityBenchmarkTest,de.konradvoelkel.android.autokorrektur.ml.NonCarOverMaskingTest
  ```
* **Run Static Analysis (Detekt)**:
  ```bash
  ./gradlew detekt
  ```
* **Run Backend Service Tests**:
  ```bash
  uv run --directory backend pytest --cov=.
  ```

## Documentation

* [TODO.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/TODO.md) — Project status, completed milestones, and prioritized development roadmap.
* [TESTING.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/TESTING.md) — Testing guidelines, benchmark metrics, and hardware matrix.
* [backend/README.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/backend/README.md) — Backend API specification, GDPR privacy model, and Docker deployment.
* [PRIVACY_POLICY.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/PRIVACY_POLICY.md) — Privacy policy and GDPR data retention terms.
* [RELEASE_CHECKLIST.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/RELEASE_CHECKLIST.md) — Release preparation and Play Store checklist.
* [STORE_LISTING.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/STORE_LISTING.md) — Play Store title, short description, and full description.

## Licenses

The licensing of this project is governed by the licenses of some components.

* **YOLOv11-seg:** Licensed under GNU AGPLv3. You must comply with its terms, which may require this entire project to be licensed similarly.

Therefore this Project is licensed under the GNU AGPLv3 License. 
