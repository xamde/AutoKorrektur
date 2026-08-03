# AutoKorrektur — Project Goal & Roadmap

## Overall Goal

Build a production-ready Android application for automatic vehicle "correction" (removal/inpainting)
using a hybrid machine learning architecture (on-device YOLO/MI-GAN + optional high-quality SDXL
backend).

## Current Status

The project has undergone several security, stability, and architectural improvements:

- Automated versioning and secure signing configurations.
- Native memory leaks in OpenCV/ONNX fixed.
- ML components migrated to Kotlin Coroutines for async execution.
- Reactive UI state model (MVVM) implemented with `StateFlow`.
- ML logic decomposed into maintainable units and unified threshold management.

---

## Current Implementation Plan

The remaining work focuses on Dependency Injection, Test Coverage, CI/CD pipelines, and
Localisation.

1. **Test Coverage & CI (Milestone D & E)**: Get instrumented tests running in CI, implement
   Robolectric for JVM tests, and establish coverage gates.
2. **Architecture Hardening (Milestone C continued)**: Add Hilt/Dagger for dependency injection,
   split god classes like `ImageProcessor`, and extract hardcoded configurations.
3. **Backend Refinement (Milestone B, C, D continued)**: Fix SDXL concurrency, improve rate
   limiting, and add security middleware.
4. **Localization & Accessibility (Milestone F)**: Add German translations and fix accessibility
   gaps (touch targets, content descriptions).
5. **Documentation & Polish (Milestone G)**: Finalize KDoc and release checklists.

---

## Remaining TODOs

### 🔴 Critical / Release Blockers

- [x] **A6. Real Play Integrity verification**
    - Implement genuine attestation against the Google Play Integrity API in the backend.

### 🟠 High Priority

- [x] **B11. Backend SDXL concurrency guard**
    - Guard global SDXL pipeline with a semaphore or move to a worker queue to prevent CUDA
      OOM/crashes.
- [x] **B12. Backend rate limiting security**
    - Key rate limits on attested identity + IP; use atomic Redis operations; prune in-memory
      fallback.
- [x] **D1. Missing Test Coverage (Core ML)**
    - Unit tests for `YoloMaskAssembler`, `Preprocessor`, `ServerSdxlApi` (MockWebServer), and
      `BatchProcessingWorker`.
- [x] **E3. CI Instrumented Tests**
    - Add emulator job to GitHub Actions to run the 25+ instrumented test classes.
- [ ] **E4. Binary Model Management**
    - Move 241MB of ML models out of Git into LFS or download-on-demand.
- [x] **F1 & F2. German Localization & String Extraction**
    - Add `values-de/strings.xml` and move remaining ~35 hardcoded strings from `FirstFragment` and
      layouts.

### 🟡 Medium Priority

- [ ] **C3. Dependency Injection**
    - Introduce Hilt or Koin to remove manual construction in UI classes.
- [x] **C9. ML Layer Injectability**
    - Define interfaces for `YoloTFLiteEngine`, `MiGanInference`, and `OkHttpClient` to allow proper
      mocking.
- [x] **C10. Split `ImageProcessor`**
    - Decouple URI dispatching, EXIF rotation, and Mat scaling into separate classes.
- [x] **C16. `BeforeAfterSliderView` Quality**
    - Fix GC churn (object allocations in `onDraw`), add accessibility support, and implement state
      save/restore.
- [x] **C18. Backend Input Validation**
    - Add dimension caps, magic byte sniffing, and re-encode images before returning.
- [ ] **E10. Code Coverage Measurement**
    - Integrate Kover/Jacoco for Android and enforce backend coverage gates.

### ⚪ Low Priority / Technical Debt

- [x] **B17. AppLogger Context Fix**
    - Initialise from `Application` context instead of `Activity`.
- [x] **C12. Remove dead API surface**
    - Clean up unused methods in `MiGanInference` and `YoloPostprocessor`.
- [x] **C14. Directory Structure Alignment**
    - Move `ImageProcessingUtils.kt` to its correct package directory.
- [x] **G1-G4. Documentation Cleanup**
    - Align `backend/README.md` with implementation; fix "SDXL" naming; add missing KDoc.
- [ ] **E17. License & Attribution**
    - Add SBOM/license plugins for AGPLv3 compliance.
