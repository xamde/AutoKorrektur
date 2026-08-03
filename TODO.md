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

- [x] **H1. README Accuracy Audit**
  - Audit `README.md` and `backend/README.md` against the final refactored implementation. Notice
    any discrepancies in feature descriptions, setup steps, or environment variables (e.g., SDXL
    concurrency, IP-based rate limiting).

### 🟡 Medium Priority

- [ ] **C3. Dependency Injection**
  - Introduce Hilt or Koin to remove manual construction in UI classes and further improve
    testability.
- [ ] **E10. Code Coverage Measurement**
  - Integrate Kover/Jacoco for Android and enforce backend coverage gates in CI.

### ⚪ Low Priority / Technical Debt

- [ ] **E17. License & Attribution**
    - Add SBOM/license plugins for AGPLv3 compliance.
