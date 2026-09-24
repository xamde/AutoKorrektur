# Product tiers and feature flags

The app builds as four product flavors, each a strict superset of the previous one. Only **`core`**
goes to the public Play Store; `plus`/`beta`/`full` are for the maintainer and opt-in testers.

Written as a plan in 2026-09 and implemented the same month; this file now describes what the
build actually does. `app/build.gradle.kts` is the authority for the flags — if the two disagree,
the Gradle file is right and this file is stale.

## The tiers

| Feature | `core` | `plus` | `beta` | `full` |
|---|:---:|:---:|:---:|:---:|
| Camera / gallery capture, fast on-device inpainting | ✅ | ✅ | ✅ | ✅ |
| Split-card share (VORHER/AUTOFREI, one layout) | ✅ | ✅ | ✅ | ✅ |
| Vision gallery (past shots) | ✅ | ✅ | ✅ | ✅ |
| Extra export layouts (carousel, animated sweep) + aspect ratios | ❌ | ✅ | ✅ | ✅ |
| High-res progressive tile inpainting | ❌ | ❌ | ✅ | ✅ |
| Manual mask brush / eraser | ❌ | ❌ | ✅ | ✅ |
| Cloud SDXL tier (+ consent dialog, quota, network code) | ❌ | ❌ | ✅ | ✅ |
| Batch processing + CSV export | ❌ | ❌ | ✅ | ✅ |
| Live AR viewfinder | ❌ | ❌ | ❌ | ✅ |
| 5 s AR video snippets (MediaCodec pipeline) | ❌ | ❌ | ❌ | ✅ |
| Evaluation sliders (mask upscale/downshift, threshold, model chooser) | debug builds only, on every flavor |

`core` and `plus` additionally drop the `INTERNET`, `READ_MEDIA_VIDEO` and `READ_MEDIA_AUDIO`
permissions, and make `MainActivity` (Studio) the launcher instead of `ArCameraActivity` — the app
opens into the AR camera only where AR exists. `core` is `arm64-v8a` only; `full` builds all ABIs
and is the CI and development baseline because it exercises every code path.

AR sits alone in `full` because it is the least finished subsystem and shares no models with the
rest of the app — cutting it saves complexity and risk, not bytes.

## Why the cuts go deeper than hiding buttons

- **No screen the user must configure before their first result** — no engine picker, no consent
  dialog, no model chooser. A feature that needs a decision the user must understand first is not
  in `core`.
- **One export shape, not a matrix.** The full export is 3 layouts × 3 ratios behind one button;
  `core` ships the split card. The job is a shareable image in two taps, not a design suite.
- **Capture → result → share is the whole critical path.** Everything else is at most one tap off
  it, never on it.
- **Absent, not hidden.** Where a flag is off, the entry point is gone rather than greyed out, so
  the simplicity is real.

Surface area is also where bugs live: field testing produced "lots of minor issues, too much to
report", which is what a large surface feels like. Every screen `core` does not have is a category
of issue a first-time user cannot hit.

## Working with the flags

Flags are `BuildConfig` booleans set per flavor in `app/build.gradle.kts`
(`FEATURE_LIVE_AR`, `FEATURE_VIDEO_SNIPPETS`, `FEATURE_CLOUD_SDXL`, `FEATURE_HIGH_RES_PROGRESSIVE`,
`FEATURE_MANUAL_MASK_BRUSH`, `FEATURE_BATCH_PROCESSING`, `FEATURE_EXTRA_EXPORT_LAYOUTS`,
`FEATURE_EVALUATION_MODE`). UI entry points are gated in `FirstFragment.applyFeatureFlags()` and
`InstagramExportBottomSheet.applyFeatureFlags()`.

Gradle tasks need a flavor prefix — `./gradlew :app:testFullDebugUnitTest`, `:app:bundleCoreRelease`;
bare `testDebugUnitTest` no longer resolves.

**Promoting a feature into `core`** is flipping its flag to `true` and checking the UI gating — never
copying code. That is the point of the arrangement. Before promoting anything into `core`, check
whether it changes what the privacy policy, the Play listing and the Data Safety answers claim:
they currently state that the published app has no network access and does none of the above.
