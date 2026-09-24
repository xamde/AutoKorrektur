# Changelog

All notable changes to the AutoKorrektur project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Opt-in, on-device diagnostics (`telemetry/`, "Diagnostics" in the menu): JSON-Lines events (stage timings, image size, detections, outcome, AR frame rate, crash class) in a 2 MB-capped private file. Never uploaded — export via the share sheet or delete. Off by default; see `PRIVACY_POLICY.md` §5.
- `PipelineStage`: pipeline progress labels are string resources, so a German device no longer sees "Running YOLO Segmentation" next to "Verarbeitung…".
- `site/` — the static website for autokorrektur.org (landing de/en, `/privacy`, `/privacy-en`, Impressum), rendered from the policy files by `build.sh` so the hosted text cannot drift.
- `PRIVACY_POLICY.en.md` — English translation; the German text stays binding.
- `scripts/fetch_assets.sh` + `scripts/assets.manifest`: models and test fixtures are fetched from the `assets-v1` release and SHA-256 verified; Gradle's `verifyAssets` fails early when they are missing, CI caches them.
- Four product flavors (`core`/`plus`/`beta`/`full`) driven by per-tier `BuildConfig` feature flags; `core` is the Play Store candidate. See `docs/PRODUCT_TIERS.md`.
- Test and CI infrastructure: `StringResourceLocalizationTest`, `TelemetryStoreTest`, unit/instrumented tests for the ML and UI layers, JaCoCo coverage, and a GitHub Actions workflow (backend pytest, lint, unit tests, emulator tests, release bundle).
- `ARCHITECTURE.md` — mask polarity, colour spaces, coordinate transforms and JNI lifecycle rules.

### Changed
- `docs/MVP_FEATURE_FLAG_PLAN.md` → `docs/PRODUCT_TIERS.md`: it was a 20 KB proposal for work finished in the same month, complete with Gradle snippets "ready to apply" and a migration sequence. Now a 3.8 KB description of what the four flavors actually are, what `core` deliberately lacks and how to promote a feature, with `app/build.gradle.kts` named as the authority for the flags.
- Documentation trimmed to what does not rot: `TESTING.md` lost the stale run dates, timings and measured values and gained the invariants worth knowing (13.6 → 4.8 KB); `README.md` points at `docs/INDEX.md` instead of repeating it.
- **Brand colours** (hue 55° orange): palette in `values/colors.xml` with every tone derived via OKLCH, wired through the Material3 light/dark themes (surfaces included), the launcher icon, the website and the Play icons.
- **German and English are both complete**: `values/strings.xml` is English-only and the fallback for every locale, `values-de/` a full override, `values-en/` gone. The localization test is now a strict invariant instead of a ratchet, and the `MissingTranslation` lint baseline is empty.
- Privacy policy, Play listing copy and the Data Safety answers describe only what `core` actually ships — no cloud tier, no AR, video or batch features.
- Public-repo hygiene after a security audit: gitleaks found no secrets in 203 commits; removed absolute `/home` paths, a phone's LAN address, a default Redis password and machine-local IDE state.
- `BatchProcessingWorker` passes batch URIs through temp JSON files (WorkManager's 10 KB limit); hardware Matrix downsampling replaces Canvas scaling; fewer per-frame allocations in `YoloMaskAssembler`; backend setting renamed to `sd_model_id`.

### Removed
- **The cloud backend moved to its own repository**, [konradvoelkel/autokorrektur-backend](https://github.com/konradvoelkel/autokorrektur-backend), with its history: the SDXL service, its container and deploy setup, and the desktop ML benchmark. It was never used by a published build — `core` has no network permission — and it carried the app repo's only Python, two CI steps and a deployment guide. The app keeps its client code behind `FEATURE_CLOUD_SDXL`.
- `walkthrough.md` (a second testing-architecture description that duplicated `TESTING.md`) and the sitemap table duplicated between `README.md` and `docs/INDEX.md`.
- **Large binaries left git and its history** (`git filter-repo`): the OpenCV 4 `.so`s — dead weight, since OpenCV 5 comes from Maven and the release bundle carried 23 MB of native code nothing loaded — plus the models, the test fixtures (the 50 reference triples had been committed three times), a 67 MB generated benchmark report and the old `AndroidKorrektur/` binaries. A clone went from **668 MB to 13 MB**; existing clones and forks must re-clone.
- `core`/`plus` no longer request `INTERNET`, `READ_MEDIA_VIDEO` or `READ_MEDIA_AUDIO` — the published app has no network permission at all.
- Owner-only notes (`TODO-for-human.md`, `HUMAN_RELEASE_CHECKLIST.md`, `BRANDING.md`) and the stale Play screenshots and feature graphic, which advertised features `core` does not have.
- Dead `useFP16` code path and the unused fp16 model (~20 MB).

### Fixed
- Bitmap leaks in the `BatchProcessingWorker` loop and in `ImageProcessor` on coroutine cancellation.
- Unsafe `!!` operators in `YoloTFLiteEngine`; native matrix cleanup in `TemporalBackgroundAccumulator`.
- Backend: blocking Play Integrity gRPC call stalling the asyncio loop; unbounded upload chunks in the inpainting endpoint.

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
