# Changelog

All notable changes to the AutoKorrektur project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Opt-in, on-device-only diagnostics (`telemetry/` package, "Diagnostics" menu entry on every flavor): when the user switches it on, the pipelines append primitive-only JSON Lines events (`session_start`, `pipeline_run` with per-stage timings/pixel count/detections/outcome, `ar_session`, `video_inpaint`, `export`, `crash` = exception class + first in-app frame) to a 2 MB-capped file in the app's private storage. Nothing is uploaded; the dialog offers Export (share sheet via FileProvider) and Delete. Off by default; `Telemetry.record()` is a no-op until enabled. Documented in `PRIVACY_POLICY.md` §5 / `PRIVACY_POLICY.en.md` §6. `TelemetryStoreTest` covers the store and encoding on the JVM.
- `PipelineStage`: localizable progress steps replacing the hard-coded English stage strings the pipelines used to hand the UI (`"Running YOLO Segmentation"` next to "Verarbeitung…" on German devices). `MainUiState.Loading` now carries a `PipelineStage` plus batch position; `FirstFragment`/`VideoPreviewActivity` resolve the text in the current locale.
- `PRIVACY_POLICY.en.md`: English translation of the privacy policy (German text stays binding).
- `site/`: the static website for autokorrektur.org (landing page de/en, `/privacy`, `/privacy-en`, `/impressum`) — `build.sh` renders the two policy files with pandoc so the hosted text cannot drift, `deploy.sh` rsyncs to the Hetzner VPS whose Caddy/DNS config lives in the infrastructure repo (`~/files/work/server`, role `autokorrektur`). Same shape as laberampel.de.
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
- **Brand colours implemented** (BRANDING.md, hue 55° orange, decided 2026-09-22): `values/colors.xml` holds the full palette, every tone derived from that one hue with OKLCH rather than picked by hand, and the Material3 theme (light + dark) is wired to it — including surfaces, because Material's baseline surface is violet-tinted and fought the brand. The identity colours hard-coded across five layouts (indigo `#818CF8`, sky `#38BDF8`, the near-black chrome, muted slate) became named tokens; feedback green/red and Instagram's pink stay off-hue on purpose. Launcher icon, website palette (`site/site.css`, written as `oklch()`) and the two Play Store icon PNGs follow the same hue.
- Security/privacy audit of the public repo (2026-09-21): gitleaks over all 203 commits found no credentials (one false positive, an IDE UUID); no secrets, keys or tokens were ever committed, so no history rewrite. Cleaned up what a public repo shouldn't carry: 41 `file:///home/…` links in the docs became relative links, the Pixel's LAN address left `README.md`/`TODO-for-human.md`, the deploy SSH target moved from `site/site.env` to the gitignored `site/deploy.local.env`, `backend/docker-compose.yml` no longer has a default Redis password (`REDIS_PASSWORD` is required; `backend/.env` gitignored), and the IDE state files `.idea/planningMode.xml`/`deviceManager.xml` are untracked.
- `docs/PLAY_STORE_LISTING.md` rewritten for the `core` flavor: the previous copy advertised Live AR, video clips, three engine tiers, cloud SDXL, brush and multi-layout export — none of which the Play build contains (a Play policy problem, not just stale text). Screenshot note added: the three in `media/play_store_assets/` show non-core features and must be retaken.
- `RELEASE_CHECKLIST.md` stale items (hand-edit versionCode/versionName, generate keystore) marked resolved; `TODO-for-human.md` states that its six scenarios target the `full` flavor.
- **Localization: German and English are now both complete.** `values/strings.xml` is purely English (the fallback for every other locale), `values-de/strings.xml` is a complete German override, and `values-en/` is gone (it was a half-override that, together with a half-German default file, only happened to cover de/en). Remaining hard-coded UI text (`BatchUiDelegate` CSV dialog, `VideoPreviewActivity` share title, `"Unknown error"`) moved to resources. `StringResourceLocalizationTest` is now a strict invariant (key parity incl. string-arrays/plurals, placeholder parity, no untranslated copy-paste, no `values-en`) instead of a 51-entry ratchet; the 119 `MissingTranslation` entries left `app/lint-baseline.xml`. The About dialog shows the real `versionName` instead of a hard-coded "1.0.0".
- `HUMAN_RELEASE_CHECKLIST.md` §2 (keystore), §3 (privacy hosting) and §7 (locale footprint) updated: keystore generated, hosting moves to autokorrektur.org, locale question resolved.
- `.github/workflows/ci.yml`, `README.md`, `TESTING.md`, `RELEASE_CHECKLIST.md`, `TODO.md`, `TODO-for-human.md`, `HUMAN_RELEASE_CHECKLIST.md`, `.github/copilot-instructions.md`: updated Gradle task names for the new product flavors (e.g. `testDebugUnitTest` → `testFullDebugUnitTest`, `bundleRelease` → `bundleCoreRelease`) — some bare `<buildType>`-only task names (`testDebugUnitTest`, `lintDebug`, `connectedDebugAndroidTest`) stopped resolving once flavors were added; others (`assembleDebug`, `bundleRelease`) still work but now aggregate all four flavors, so were made explicit instead. CI now runs lint/unit/instrumented tests against `full` and builds the release bundle from `core`.
- Refactored `BatchProcessingWorker` to store batch URIs in temporary JSON files, bypassing WorkManager's 10KB Data payload limit.
- Standardized backend diffusion inpainting configuration setting to `sd_model_id`.
- Replaced Canvas-based Bitmap downsampling with hardware Matrix-based scaling.
- Replaced ThreadLocal prototype channel buffers in `YoloMaskAssembler` to reduce per-frame GC allocations.

### Removed
- `TODO-for-human.md` (owner-only device/testing notes), `HUMAN_RELEASE_CHECKLIST.md` (owner-only release steps, rewritten short) and `BRANDING.md` (design decision doc) are no longer in the repo — gitignored, kept locally. The first two were also purged from history along with every large binary: `git filter-repo` removed the OpenCV `.so`s, the models, the test fixtures, the old `AndroidKorrektur/` binaries and the generated benchmark report from all 224 commits, taking the repo from **668 MB to 13 MB**. HEAD's tree is byte-identical to before the rewrite; every branch and tag was rewritten and force-pushed, so **existing clones and forks must re-clone** (`git fetch` will not reconcile).
- **Large binaries left the git tree** (574 MB tracked → 21 MB). `app/src/main/jniLibs/*/libopencv_java4.so` (136 MB) were dead weight: OpenCV 5 comes from Maven (`org.opencv:opencv:5.0.0.1`, which ships `libopencv_java5.so` for every ABI), and the APK was shipping both — the `core` release bundle carried 23 MB of native code nothing loaded. The models and test fixtures now live on the `assets-v1` release, pinned by SHA-256 in `scripts/assets.manifest` and installed by `scripts/fetch_assets.sh` (cached, verified, idempotent); Gradle's `verifyAssets` task fails early with that command if something is missing, and CI fetches with a cache keyed on the manifest. The 50 reference triples had been committed three times (`androidTest/assets`, `test/resources`, `backend/tests/fixtures`, 97 MB each, byte-identical); one archive is now extracted into all three. Also dropped: `backend/yolov8n-seg.pt` (7 MB, referenced nowhere).
- `core`/`plus` manifests drop `INTERNET`, `READ_MEDIA_VIDEO` and `READ_MEDIA_AUDIO` (`tools:node="remove"`): those tiers have no network feature and no video/audio handling, and `PRIVACY_POLICY.md` §1 now promises "no internet permission" for the published app. `beta`/`full` keep them.
- `PRIVACY_POLICY.md` / `.en.md` trimmed to what `core` actually does (photo in, on-device inpainting, save/share, opt-in diagnostics): AR mode, video clips, high-res tile mode and batch processing are no longer described either — they ship only in unpublished flavors.
- `PRIVACY_POLICY.md` / `.en.md` §4 "Optional cloud processing (Cloud SDXL – Frankfurt)": the cloud tier is unfinished and off in every public flavor, so the policy (and the hosted site) no longer describes it; the text is in git history for when it ships.
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
