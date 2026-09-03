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
- `docs/MVP_FEATURE_FLAG_PLAN.md`: proposal for a build-time feature-flag tier system (`core`/`plus`/`beta`/`full`) to ship a radically simplified MVP flavor while keeping cut features re-enterable. Migration steps 1–5 implemented: see Changed/Removed below; only step 6 (promoting proven tier features into `core` over time) remains, and it's ongoing product work, not a one-shot task.
- Per-tier `BuildConfig` feature flags (`FEATURE_LIVE_AR`, `FEATURE_VIDEO_SNIPPETS`, `FEATURE_CLOUD_SDXL`, `FEATURE_HIGH_RES_PROGRESSIVE`, `FEATURE_MANUAL_MASK_BRUSH`, `FEATURE_BATCH_PROCESSING`, `FEATURE_EXTRA_EXPORT_LAYOUTS`, `FEATURE_EVALUATION_MODE`). `FirstFragment.applyFeatureFlags()` and `InstagramExportBottomSheet.applyFeatureFlags()` gate their respective UI entry points on these flags.
- Four product flavors — `core` (Play Store candidate, all flags off, `arm64-v8a` only), `plus` (+ extra export layouts), `beta` (+ cloud/high-res/brush/batch), `full` (everything on, all ABIs — the pre-flavor app, now the CI/dev baseline) — see `app/build.gradle.kts` and `docs/MVP_FEATURE_FLAG_PLAN.md` §4/§6.
- `app/src/core/AndroidManifest.xml` and `app/src/plus/AndroidManifest.xml`: flavor-specific manifest override making `MainActivity` (Studio) the app's `MAIN`/`LAUNCHER` activity for those two tiers instead of `ArCameraActivity`. Correction found while wiring this up: `ArCameraActivity`, not `MainActivity`, was already the app's actual launcher — this plan's earlier framing of AR as one optional button off a Studio home screen had that backwards (see `docs/MVP_FEATURE_FLAG_PLAN.md` §2).
- `app/lint-baseline.xml`: baselines 119 pre-existing `MissingTranslation` lint errors (`values/strings.xml` vs `values-en`), confirmed present before this session's work (same root cause as the `StringResourceLocalizationTest` ratchet, TESTING.md §8), so CI isn't blocked by unrelated debt. Doesn't fix or hide the underlying issue — a newly introduced lint issue still fails CI.

### Changed
- `.github/workflows/ci.yml`, `README.md`, `TESTING.md`, `RELEASE_CHECKLIST.md`, `TODO.md`, `TODO-for-human.md`, `HUMAN_RELEASE_CHECKLIST.md`, `.github/copilot-instructions.md`: updated Gradle task names for the new product flavors (e.g. `testDebugUnitTest` → `testFullDebugUnitTest`, `bundleRelease` → `bundleCoreRelease`) — some bare `<buildType>`-only task names (`testDebugUnitTest`, `lintDebug`, `connectedDebugAndroidTest`) stopped resolving once flavors were added; others (`assembleDebug`, `bundleRelease`) still work but now aggregate all four flavors, so were made explicit instead. CI now runs lint/unit/instrumented tests against `full` and builds the release bundle from `core`.
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
