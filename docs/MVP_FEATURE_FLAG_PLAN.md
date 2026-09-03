# AutoKorrektur — MVP Amputation & Build-Time Feature Flag Plan

**Status:** in progress. Written 2026-09-03 against the actual repo state. Migration steps 1–5
(§6) done 2026-09-03 from a session with full local shell + SDK access to this checkout — only
step 6 (cut `core` loose / promote proven `plus`/`beta` features into it) remains, and that's a
product decision that happens over time as tiers get real testing, not a one-shot task.
**Goal:** ship a radically simple, easy-to-share "campaign tool" app first, while keeping every
cut feature's code alive and re-enterable behind a build-time flag, so a beta-tester APK with
more features is one Gradle task away, never a re-implementation.

---

## 1. Design principle: ease-of-use is an architectural constraint, not a UI cut

The instinct to simplify has to go deeper than "hide some buttons." Every one of these should be
true for the MVP flavor specifically, not just "true by default":

- **No screen the user must configure before their first result.** No engine picker, no GDPR
  consent dialog, no segmentation-model chooser. If a feature requires a decision the user has to
  understand first, it isn't in the MVP.
- **One export shape, not a matrix.** Today's Instagram export is 3 layouts × 3 aspect ratios = 9
  combinations behind one button. The MVP ships exactly one (the split card), because a campaign
  tool's job is to get a shareable image into someone's hands in two taps, not to be a design
  suite.
- **The share button is the destination, not an afterthought.** Capture → result → share should
  be the entire critical path. Everything else is one tap away from that path at most, never on it.
- **If a screen doesn't exist in this flavor, it doesn't compile in this flavor.** Not hidden
  behind a flag check that still ships the code and the cognitive load of "maybe there's more
  here" — actually absent, so the simplicity is real, not cosmetic.

This is also, mechanically, good bug hygiene: the field-testing feedback ("lots of minor issues,
too much to report") is a symptom of surface area. Every screen removed from the MVP is a
category of minor issue that can no longer occur in the build real first-time users get.

---

## 2. The tier menu

Four named tiers, each a strict superset of the previous one. Only **`core`** is a public Play
Store candidate; `plus`/`beta`/`full` are for you and opt-in beta testers only (see §5).

| Feature | `core` (MVP) | `plus` | `beta` | `full` (today's app) |
|---|:---:|:---:|:---:|:---:|
| Camera / gallery capture, Fast on-device inpainting | ✅ | ✅ | ✅ | ✅ |
| Split-card share (VORHER/AUTOFREI, one layout) | ✅ | ✅ | ✅ | ✅ |
| Vision gallery (past shots) | ✅ | ✅ | ✅ | ✅ |
| Extra export layouts (carousel, animated video-sweep) + aspect ratios | ❌ | ✅ | ✅ | ✅ |
| High-Res Progressive tile inpainting | ❌ | ❌ | ✅ | ✅ |
| Manual mask brush / eraser | ❌ | ❌ | ✅ | ✅ |
| Cloud SDXL tier (+ GDPR consent dialog, quota, network code) | ❌ | ❌ | ✅ | ✅ |
| Batch processing + CSV export | ❌ | ❌ | ✅ | ✅ |
| Live AR viewfinder | ❌ | ❌ | ❌ | ✅ |
| 5s AR video snippets + MediaCodec pipeline | ❌ | ❌ | ❌ | ✅ |
| Evaluation-mode dev sliders (mask upscale/downshift, score threshold, model chooser) | debug builds only, **on every flavor** — this was never a real end-user feature and shouldn't ship to anyone but you |

Why AR sits alone in `full`: it's explicitly the subsystem you flagged as not yet bug-free, it
shares no models with the rest of the app (so cutting it saves zero APK bytes — this is a
complexity/risk cut, not a size cut), and it deserves its own dedicated bug-fixing and
field-testing round before it's in front of anyone but you again.

> **Correction found during step 3 (2026-09-03), important for step 4:** `ArCameraActivity`, not
> `MainActivity`/`FirstFragment`, is the app's actual `MAIN`/`LAUNCHER` activity today
> (`AndroidManifest.xml`) — confirmed on-device, the app opens directly into the AR camera, and
> `FirstFragment` ("Studio") is reached *from* AR mode via `btnOpenStudio`, not the other way
> around. `TODO-for-human.md` already documents this correctly ("Test 3: ... From AR mode, tap
> the Studio floating action button"); this plan's framing of AR as one optional button off a
> campaign-tool home screen was the piece that didn't account for it. **Consequence for `core`:**
> hiding `arLiveModeButton` in `FirstFragment` (done in step 3, see §6) is necessary but not
> sufficient — a `core`/`plus` build must also make `MainActivity` the launcher instead of
> `ArCameraActivity`, via a flavor-specific `AndroidManifest.xml` (e.g.
> `app/src/core/AndroidManifest.xml` overriding the `<intent-filter>` on each activity) in step 4.
> Flagging now so step 4 doesn't get blindsided by it.

---

## 3. What actually shrinks the APK (and what doesn't)

Being precise about this matters because "make it toggleable" and "make it smaller" are two
different mechanisms:

- **ABI filtering — the single biggest lever.** The four bundled `libopencv_java4.so` files
  together are ~134MB (arm64-v8a 23MB, armeabi-v7a 15MB, x86 41MB, x86_64 55MB). `x86`/`x86_64`
  exist almost solely for emulators; essentially no real user's phone needs them. Restrict
  `core`/`plus`/`beta` to `arm64-v8a` only (covers the overwhelming majority of Android devices
  sold since ~2017) and keep all four ABIs only on `full` for your own emulator/dev use. Note:
  since you already build an `.aab` and Play Store already does per-device ABI splitting from it,
  this specific win mostly matters for **sideloaded beta APKs**, not the eventual Play-distributed
  build — but that's exactly the distribution path beta testers use before you have a Play
  listing for a given tier.
- **The YOLO model duplication — resolved, cut.** `assets/model/` used to ship *both*
  `yolo11s-seg_float16.tflite` (20MB) and `yolo11s-seg_float32.tflite` (40MB), gated by a
  `useFP16` parameter threaded through `YoloEngine`/`YoloTFLiteEngine`/`YoloService`/
  `YoloServiceImpl`. Audit (2026-09-03): every real production call site
  (`StaticImagePipeline`, `RealtimeArPipeline`, `VideoPreviewActivity`) hardcoded
  `useFP16 = false`; `useFP16 = true` was reachable only from one instrumented test
  (`MlComponentTests.testYoloServiceModels`), which itself wrapped that path in a
  swallowed-exception try/catch while requiring the `false` path to succeed — i.e. the test's own
  author already treated fp16 as non-load-bearing. The NNAPI-vs-CPU fallback in
  `YoloTFLiteEngine.initialize` (a real, load-bearing fallback) operates on whichever single model
  buffer was already loaded — it never re-loads a different model file, so it was never coupled
  to the float16/float32 choice at all. Conclusion: leftover redundancy, not a real fallback path.
  Removed the float16 asset and the `useFP16` parameter entirely (dead branch, not worth keeping
  behind a flag) — saves ~20MB in every flavor, including `core`.
- **Feature flags mostly cut code and screens, not megabytes.** Cloud SDXL, manual brush, batch
  processing, AR — none of these ship a bundled model of their own; cutting them from `core`
  shrinks the DEX/resources modestly and (more importantly) shrinks the surface area of things
  that can break. Don't oversell the flags as an APK-size story; sell them as a complexity and
  risk story. The ABI filter and the model audit are the actual size story.
- **A forward-looking hook, not immediate work:** `strings.xml` already carries a
  `yolo_model_options` array (Nano/Small/Medium). A future ultra-light flavor could ship
  YOLO-Nano instead of YOLO-Small for an even smaller, faster budget build — worth knowing the
  hook exists, not worth building until a tier actually needs it.

---

## 4. Gradle mechanics (ready to apply to `app/build.gradle.kts`)

This slots into the existing file — same pattern already used for `BACKEND_URL`, so it's
consistent with how the codebase already does build-time configuration:

```kotlin
android {
    // ... existing compileSdk / defaultConfig ...

    flavorDimensions += "scope"
    productFlavors {
        create("core") {
            dimension = "scope"
            buildConfigField("boolean", "FEATURE_LIVE_AR", "false")
            buildConfigField("boolean", "FEATURE_VIDEO_SNIPPETS", "false")
            buildConfigField("boolean", "FEATURE_CLOUD_SDXL", "false")
            buildConfigField("boolean", "FEATURE_HIGH_RES_PROGRESSIVE", "false")
            buildConfigField("boolean", "FEATURE_MANUAL_MASK_BRUSH", "false")
            buildConfigField("boolean", "FEATURE_BATCH_PROCESSING", "false")
            buildConfigField("boolean", "FEATURE_EXTRA_EXPORT_LAYOUTS", "false")
            ndk { abiFilters += "arm64-v8a" }
            // no applicationIdSuffix: this is "the app" as far as Play/users are concerned
        }
        create("plus") {
            dimension = "scope"
            applicationIdSuffix = ".plus"
            buildConfigField("boolean", "FEATURE_LIVE_AR", "false")
            buildConfigField("boolean", "FEATURE_VIDEO_SNIPPETS", "false")
            buildConfigField("boolean", "FEATURE_CLOUD_SDXL", "false")
            buildConfigField("boolean", "FEATURE_HIGH_RES_PROGRESSIVE", "false")
            buildConfigField("boolean", "FEATURE_MANUAL_MASK_BRUSH", "false")
            buildConfigField("boolean", "FEATURE_BATCH_PROCESSING", "false")
            buildConfigField("boolean", "FEATURE_EXTRA_EXPORT_LAYOUTS", "true")
            ndk { abiFilters += "arm64-v8a" }
        }
        create("beta") {
            dimension = "scope"
            applicationIdSuffix = ".beta"
            buildConfigField("boolean", "FEATURE_LIVE_AR", "false")
            buildConfigField("boolean", "FEATURE_VIDEO_SNIPPETS", "false")
            buildConfigField("boolean", "FEATURE_CLOUD_SDXL", "true")
            buildConfigField("boolean", "FEATURE_HIGH_RES_PROGRESSIVE", "true")
            buildConfigField("boolean", "FEATURE_MANUAL_MASK_BRUSH", "true")
            buildConfigField("boolean", "FEATURE_BATCH_PROCESSING", "true")
            buildConfigField("boolean", "FEATURE_EXTRA_EXPORT_LAYOUTS", "true")
            ndk { abiFilters += "arm64-v8a" }
        }
        create("full") {
            dimension = "scope"
            applicationIdSuffix = ".full"
            buildConfigField("boolean", "FEATURE_LIVE_AR", "true")
            buildConfigField("boolean", "FEATURE_VIDEO_SNIPPETS", "true")
            buildConfigField("boolean", "FEATURE_CLOUD_SDXL", "true")
            buildConfigField("boolean", "FEATURE_HIGH_RES_PROGRESSIVE", "true")
            buildConfigField("boolean", "FEATURE_MANUAL_MASK_BRUSH", "true")
            buildConfigField("boolean", "FEATURE_BATCH_PROCESSING", "true")
            buildConfigField("boolean", "FEATURE_EXTRA_EXPORT_LAYOUTS", "true")
            // no abiFilters override -> all 4 ABIs, for emulators/dev
        }
    }
}
```

`BuildConfig.FEATURE_X` fields are `static final boolean` constants, so R8 constant-propagates
through `if (BuildConfig.FEATURE_X) { ... }` branches in release builds and dead-code-eliminates
the unreachable side — this is the same mechanism that already makes `isMinifyEnabled = true`
useful for you today, so gating with `BuildConfig` booleans (rather than, say, a runtime
`SharedPreferences` toggle) is what actually keeps the disabled code out of the `core` APK's DEX,
not just out of its UI.

A hidden, tier-independent debug flag for the evaluation-mode sliders:
```kotlin
buildTypes {
    debug {
        buildConfigField("boolean", "FEATURE_EVALUATION_MODE", "true")
    }
    release {
        buildConfigField("boolean", "FEATURE_EVALUATION_MODE", "false")
    }
}
```

---

## 5. Distribution: who gets which tier

- **`core`** is the only flavor that ever goes to the public Play Store listing.
- **`plus`/`beta`/`full`** are for you and opt-in testers only, distributed as directly-installed
  APKs (a GitHub Release asset, or a Play Console **closed/internal testing track** if you want
  Play's own crash reporting and staged rollout for them) — never as separate public listings.
  Running multiple tiers side-by-side on your own test device for comparison is exactly what the
  `applicationIdSuffix` per non-`core` flavor buys you.
- When a `plus`/`beta` feature proves itself with real testers, the move is to flip its flag to
  `true` in `core` (i.e. promote it), not to duplicate code — the whole point of the flag
  architecture is that promotion is a one-line Gradle change plus a UI-gating check, not a
  re-implementation.

---

## 6. Migration sequence (incremental, safety-net-first)

Given the existing test suite is your actual safety net here, do this in small, independently
verifiable steps rather than one large refactor:

1. ~~**Audit `ModelAssetProvider`** to settle the float16/float32 question from §3 before touching
   any assets.~~ **Done 2026-09-03** — see §3, float16 asset and `useFP16` param removed.
2. ~~**Introduce the `BuildConfig` flags with no flavors yet** — all flags hardcoded `true` in a
   single build type, zero behavior change.~~ **Done 2026-09-03** — `FEATURE_LIVE_AR`,
   `FEATURE_VIDEO_SNIPPETS`, `FEATURE_CLOUD_SDXL`, `FEATURE_HIGH_RES_PROGRESSIVE`,
   `FEATURE_MANUAL_MASK_BRUSH`, `FEATURE_BATCH_PROCESSING`, `FEATURE_EXTRA_EXPORT_LAYOUTS` added
   to `defaultConfig` (all `true`), `FEATURE_EVALUATION_MODE` added per build type (`true` debug /
   `false` release). No call sites reference them yet — that's step 3. `testDebugUnitTest` stayed
   green throughout.
3. ~~**Gate UI entry points**~~ **Done 2026-09-03** — `FirstFragment.applyFeatureFlags()` hides
   `arLiveModeButton` (`FEATURE_LIVE_AR`), the cloud/high-res quality chips (`FEATURE_CLOUD_SDXL`,
   `FEATURE_HIGH_RES_PROGRESSIVE`), `btnMaskBrush` (`FEATURE_MANUAL_MASK_BRUSH`), the batch-mode
   options row (`FEATURE_BATCH_PROCESSING`), and the evaluation-mode dev sliders/spinners
   (`FEATURE_EVALUATION_MODE`); `InstagramExportBottomSheet.applyFeatureFlags()` hides the
   carousel/video export chips and the whole aspect-ratio picker (`FEATURE_EXTRA_EXPORT_LAYOUTS`).
   `FEATURE_VIDEO_SNIPPETS` has no separate entry point yet — video snippets are only reachable
   through AR mode, already gated by `FEATURE_LIVE_AR`; revisit if a tier ever wants AR preview
   without recording. Verified zero behavior change: `compileDebugKotlin` +
   `testDebugUnitTest` green, `assembleDebug` installed and screenshotted on a Pixel 9 Pro XL
   emulator (all flags still hardcoded `true`, so nothing is hidden yet). Full Espresso suite not
   run per-flag (would need a per-flag emulator boot); deferred to a single pass once step 4's
   flavors give the gates something to actually verify. See the launcher-activity correction above
   — required for step 4, not addressed by this step's BuildConfig gating alone.
4. ~~**Add the `flavorDimensions`/`productFlavors` block**~~ **Done 2026-09-03** — exactly the §4
   block, `core`/`plus`/`beta`/`full` all created. Also added the flavor-specific
   `app/src/core/AndroidManifest.xml` / `app/src/plus/AndroidManifest.xml` (identical files)
   swapping the launcher from `ArCameraActivity` to `MainActivity` for those two tiers, per the
   correction above — needed `tools:node="merge"` **and** `tools:replace="android:exported"` on
   both `<activity>` elements (a bare `tools:node="merge"` isn't enough; the manifest merger
   treats a same-named attribute present in both the base and flavor manifest with different
   values as an unresolved conflict, not an override, unless `tools:replace` says which attribute
   the flavor manifest is allowed to win on). Verified on real merged manifests
   (`app/build/intermediates/merged_manifest/*/`): `core`/`plus` → `MainActivity` is
   `MAIN`/`LAUNCHER`, `ArCameraActivity` has `exported="false"` and no intent-filter; `full`/`beta`
   → unchanged, `ArCameraActivity` still launches directly, identical to the pre-flavor app.
   `bundleCoreRelease` and `bundlePlusDebug` both build clean end to end.
5. ~~**Update CI and docs for flavor-qualified task names**~~ **Done 2026-09-03**, folded into the
   same change per this step's own instruction. What actually breaks after adding flavors is more
   specific than "plain task names stop resolving" — verified against `./gradlew tasks --all`:
   `assemble<BuildType>` and `bundle<BuildType>` (e.g. `assembleDebug`, `bundleRelease`) keep
   working as aggregates across all four flavors, but `test<BuildType>UnitTest`,
   `lint<BuildType>`, and `connected<BuildType>AndroidTest` (e.g. `testDebugUnitTest`,
   `lintDebug`, `connectedDebugAndroidTest`) do not — only `test`/`lint`/`connectedAndroidTest`
   (all four flavors) or the fully flavor-qualified per-variant task exist. Updated
   `.github/workflows/ci.yml`, `README.md`, `TESTING.md`, `RELEASE_CHECKLIST.md`, `TODO.md`,
   `TODO-for-human.md`, `HUMAN_RELEASE_CHECKLIST.md`, and `.github/copilot-instructions.md`.
   CI now targets `full` for lint/unit-tests/instrumented-tests (the only flavor exercising every
   code path, so no coverage regression from adding flavors) and `core` for the release bundle
   (the actual Play Store artifact) — see `app/build.gradle.kts`'s `jacocoTestReport` task, which
   needed its hardcoded `debug`-variant paths updated to `fullDebug` the same way.
   **Unrelated finding surfaced along the way:** `lintFullDebug` fails with 119 `MissingTranslation`
   errors (`values/strings.xml` vs `values-en`) — confirmed via a throwaway worktree at commit
   `a2a81e9` that this is pre-existing debt, present before any of this session's work, same root
   cause as the 51-string mismatch `StringResourceLocalizationTest` (TESTING.md §8) already
   ratchets. Added `app/lint-baseline.xml` (via `lint { baseline = file("lint-baseline.xml") }` in
   `app/build.gradle.kts`) so CI isn't blocked by it — this snapshots the *existing* issues as
   known, it does not fix or hide them going forward; a lint issue introduced from here still
   fails CI. Actually fixing the missing translations is separate, unstarted work.
6. **Cut `core` loose** as the Play Store candidate; keep the rest internal. This is the one
   remaining step, and it isn't a single task — it's what happens as `plus`/`beta` features get
   real field testing (see `TODO.md` Milestone 1) and either get promoted into `core` (flip the
   flag) or stay tier-gated.

Steps 1-2 are low-risk (asset/config only, no UI logic) and I can take a pass at them once you've
looked over this plan — the actual UI-gating in step 3 touches `FirstFragment.kt` (30KB, the
biggest file in the app) and the nav graph, which I currently can't read through the device
bridge (its path is nested one level deeper than the bridge's folder-staging limit allows), so
that step needs either you, or a session with broader local access to the repo.

---

## 7. Where this leaves the blog post

The "paper" isn't about the ML pipeline — it's about the process: an agent-built app that arrived
at v1.0.0 fully loaded (AR, cloud tier, manual brush, batch mode, a 4-tier benchmark suite) before
a single outside human had used it, and the deliberate, incremental *reduction* back to something
a stranger can pick up and understand in ten seconds. "The agents over-built it, and then we had
to teach them (and ourselves) restraint" is a much more interesting beat for an agentic-coding
retrospective than "here's a well-tested car-removal pipeline" — and this MVP cut is the part of
that story that hasn't happened yet, which makes it worth writing *as* you do it, not after.
