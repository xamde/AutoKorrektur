# Production Readiness & Feature TODO List

- [x] **1. Memory Safety & OutOfMemory (OOM) Protection**
  - Implement downscaled display Bitmaps for UI rendering (clamped to max 1920px screen bounds) to prevent OOM errors on high-resolution camera photos.
  - Preserve full-resolution Bitmaps strictly for file saving and high-res export.

- [x] **2. Configuration Change & Orientation State Preservation**
  - Retain selected image URI, processed image URI, and slider position across screen rotation / dark mode toggle via ViewModel or SavedStateHandle.

- [x] **3. Asynchronous Execution & ANR Prevention Audit**
  - Audit and enforce that all OpenCV matrix conversions, EXIF rotation decoding, and ONNX model inferences run strictly on `Dispatchers.Default` / `Dispatchers.IO`.

- [x] **4. Zero-Permission Storage Access Verification**
  - Ensure the app uses Android System Photo Picker (`ActivityResultContracts.PickVisualMedia`) for privacy-friendly zero-permission gallery picking on API 33/34+.

- [x] **5. Processing Progress Updates & Inference Cancellation Support**
  - Add percentage-based stage updates during YOLO detection and MI-GAN inpainting.
  - Support job cancellation (`Job.cancel()`) when user cancels or leaves screen.

- [x] **6. Manual Mask Touch-Up Brush & Eraser Component**
  - Create a custom mask touch-up view allowing users to paint or erase vehicle detection masks before running inpainting.

- [x] **7. Release Build & R8/ProGuard Optimization Rules**
  - Configure `app/build.gradle.kts` release build type with ProGuard rules (`proguard-rules.pro`) preserving ONNX runtime and OpenCV JNI bindings.

- [x] **8. Espresso Instrumented UI Test Suite**
  - Add [MainActivityEspressoTest.kt](file:///home/konrad/files/work/__drafts/AutoKorrektur/app/src/androidTest/java/de/konradvoelkel/android/autokorrektur/MainActivityEspressoTest.kt) to test UI navigation, button clicks, and dialogs end-to-end on the emulator.

## Milestone 2: Hybrid Server Architecture, NPU Acceleration & Quality Control

- [x] **9. Hybrid SDXL Server & GDPR Privacy Flow**
  - Implement FastAPI backend service with memory-only image handling and client-side opt-in GDPR consent.

- [x] **10. Hardware Acceleration & NNAPI Execution Provider**
  - Probe device NPU capabilities via `DevicePerformanceHelper` and enable Android NNAPI EP for ONNX Runtime & TFLite with safe CPU fallback.

- [x] **11. Self-Staging Integration Tests**
  - Enhance `VehicleTestDataIntegrationTest.kt` to dynamically extract test images from assets if missing from `/sdcard/`.

- [x] **12. Detekt Static Analysis Integration**
  - Configure `app/detekt-baseline.xml` and document `./gradlew detekt` in `TESTING.md`.

## Milestone 3: Release Optimization & Bundle Shrinking

- [x] **13. R8 Code & Resource Shrinking**
  - Enable `isMinifyEnabled = true` and `isShrinkResources = true` in `app/build.gradle.kts` release build type.

- [x] **14. Gradle Dependency Clean Up**
  - Clean up duplicate dependency declarations in `app/build.gradle.kts`.

- [x] **15. Release Bundle & APK Verification**
  - Verify `./gradlew assembleRelease` and `./gradlew bundleRelease` pass without R8 compilation errors.


