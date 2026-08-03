# AutoKorrektur — Project Goal & Roadmap

## Overall Goal

Build a production-ready Android application for automatic vehicle "correction" (removal/inpainting)
using a hybrid machine learning architecture (on-device YOLO/MI-GAN + optional high-quality SDXL
backend).

## Current Status

The project has undergone extensive hardening across client and backend:

- **Client**: MVVM architecture with `StateFlow`, clean `ImageProcessor` decomposition, full German
  localization, and robust `BeforeAfterSliderView`.
- **Backend**: Concurrency-guarded SDXL pipeline, IP-based rate limiting, Play Integrity
  attestation, and memory-only GDPR processing.
- **Testing**: Parallelized CI with instrumented tests on real emulators; >80% backend coverage.

---

## Remaining TODOs

### 🔴 Critical / Release Blockers

- [ ] **E4. Binary Model Management**
  - Move 241MB of ML models out of Git into LFS or download-on-demand to reduce repository bloat.

### 🟠 High Priority

- [ ] **D2. Rigorous Inpainting Benchmark (50 Triples)**
  - Implement an instrumented test running the full `StaticImagePipeline` on all 50 reference
    triples.
  - Verify zero cars detected in results and calculate PSNR/SSIM quality metrics.
- [ ] **G5. Architecture Visuals & Report Guide**
  - Add Mermaid diagrams of the hybrid ML pipeline to `README.md`.
  - Create a standardized guide for generating visual "Before/After" reports from benchmarks.

### 🟡 Medium Priority

- [ ] **C3. Dependency Injection**
  - Introduce Hilt or Koin to remove manual construction in UI classes and further improve
    testability.
- [ ] **C19. Extract Consent Management**
  - Move SDXL GDPR consent logic from `FirstFragment` to a dedicated `ConsentManager` or
    `SettingsRepository`.
- [ ] **C20. ML Exception Standardization**
  - Implement a consistent exception hierarchy (`InferenceException`, `ModelLoadException`) across
    all engines.
- [ ] **D3. Backend Play Integrity Mock Tests**
  - Add unit tests for `verify_token` mocking the Google Cloud client to verify verdict handling.
- [ ] **E10. Code Coverage Measurement**
  - Integrate Kover/Jacoco for Android and enforce backend coverage gates in CI.

### ⚪ Low Priority / Technical Debt

- [ ] **G6. App Troubleshooting Guide**
  - Add a section to `TESTING.md` for on-device failure modes (NNAPI, memory pressure).
- [ ] **E17. License & Attribution**
  - Add SBOM/license plugins for AGPLv3 compliance.
