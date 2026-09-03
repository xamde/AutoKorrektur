# Changelog

All notable changes to the AutoKorrektur project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `StringResourceLocalizationTest` (Tier 5): catches silent language-mixing in string resources for locales without a dedicated `values-<locale>` override. Found and ratcheted 51 pre-existing German strings in the default (locale-neutral) `values/strings.xml` that disagree with `values-en`. See `TESTING.md` §8.
- Proposed Tier 5 UI-quality automation plan in `TESTING.md`: Paparazzi config-matrix screenshot tests (locale × night mode × width), Espresso `AccessibilityChecks`, and a nightly monkey/fuzz stability job — aimed at the class of "minor issues" that field testing surfaces one at a time but automated sweeps catch exhaustively.
- GitHub Actions CI workflow executing backend pytest, Android lint, unit tests, JaCoCo coverage, and headless Android emulator tests.
- Comprehensive JaCoCo coverage reporting via `./gradlew jacocoTestReport`.
- Dedicated unit and instrumented tests for `ImageProcessor`, `YoloTFLiteEngine`, `ServerInpainter`, `ConsentManager`, `QuotaManager`, `MainViewModel`, and UI delegates.
- Pre-commit configuration for Python linting and formatting.
- Comprehensive `ARCHITECTURE.md` specification detailing mask polarity, color spaces, coordinate transformations, and neural pipeline lifecycle.
- `docs/MVP_FEATURE_FLAG_PLAN.md`: proposal for a build-time feature-flag tier system (`core`/`plus`/`beta`/`full`) to ship a radically simplified MVP flavor while keeping cut features re-enterable. Migration steps 1–2 implemented: see Changed/Removed below.
- Per-tier `BuildConfig` feature flags (`FEATURE_LIVE_AR`, `FEATURE_VIDEO_SNIPPETS`, `FEATURE_CLOUD_SDXL`, `FEATURE_HIGH_RES_PROGRESSIVE`, `FEATURE_MANUAL_MASK_BRUSH`, `FEATURE_BATCH_PROCESSING`, `FEATURE_EXTRA_EXPORT_LAYOUTS`, `FEATURE_EVALUATION_MODE`) — all currently hardcoded `true` (`FEATURE_EVALUATION_MODE`: debug-only), no call sites gated yet, zero behavior change.

### Changed
- Refactored `BatchProcessingWorker` to store batch URIs in temporary JSON files, bypassing WorkManager's 10KB Data payload limit.
- Standardized backend diffusion inpainting configuration setting to `sd_model_id`.
- Replaced Canvas-based Bitmap downsampling with hardware Matrix-based scaling.
- Replaced ThreadLocal prototype channel buffers in `YoloMaskAssembler` to reduce per-frame GC allocations.

### Removed
- Dead `useFP16` code path through `YoloEngine`/`YoloTFLiteEngine`/`YoloService`/`YoloServiceImpl` and the unused `yolo11s-seg_float16.tflite` asset (~20MB). Audit found every production call site hardcoded `useFP16 = false`; the only path that ever requested the fp16 model was one instrumented test that already tolerated its failure. Not a real NNAPI fallback — that fallback (NNAPI→CPU) operates on whichever single model buffer was already loaded, never re-selects a model file. See `docs/MVP_FEATURE_FLAG_PLAN.md` §3.

### Fixed
- Fixed bitmap memory leaks in `BatchProcessingWorker` batch iteration loop.
- Fixed bitmap leaks in `ImageProcessor` upon coroutine cancellation.
- Eliminated all unsafe `!!` operators in `YoloTFLiteEngine`.
- Fixed blocking Play Integrity gRPC calls stalling FastAPI async event loop.
- Enforced streaming chunk upload size limits in cloud inpainting endpoint.
- Added native matrix cleanup fallback in `TemporalBackgroundAccumulator`.

---

## [1.0.0] - 2026-08-14

### Added
- Complete on-device vehicle detection and segmentation pipeline powered by YOLOv11-seg (TFLite).
- High-quality neural inpainting using on-device MI-GAN (ONNX Runtime) and remote Stable Diffusion XL (SDXL).
- Real-time AR mode with temporal background accumulation for live vehicle erasure.
- Instagram social comparison graphic export with 1:1, 4:5, and 9:16 aspect ratios.
- Multi-image background batch processing via Android WorkManager with CSV export.
- Desktop and on-device benchmark harnesses evaluating IoU, Dice $F_1$, Boundary-IoU, and PSNR/SSIM.
- Play Integrity attestation and daily free quota enforcement for cloud SDXL requests.
