# Testing Guide

How the AutoKorrektur Android client is tested and how to run each suite. The optional cloud
service and the desktop ML benchmark moved to
[konradvoelkel/autokorrektur-backend](https://github.com/konradvoelkel/autokorrektur-backend) and
are tested there.

> **Before the first run**: `scripts/fetch_assets.sh`. The models and the 50 reference triples are
> not in git (`scripts/assets.manifest`); one archive is extracted into both places that need it —
> `app/src/androidTest/assets/triples/` and `app/src/test/resources/triples/`. Gradle fails with
> that command if anything is missing.

The app builds as four product flavors (`docs/PRODUCT_TIERS.md`), so Gradle test tasks
need a flavor prefix — bare `testDebugUnitTest` does not resolve. The commands below use `full`,
the flavor that exercises every code path; substitute `core`/`plus`/`beta` to test a tier.

---

## 1. Suites

| Suite | Where | What it covers |
| :--- | :--- | :--- |
| **JVM unit tests** | `app/src/test/java/…` | ViewModel and UI state, quota/consent managers, mask and image maths, diagnostics store, string-resource contract. MockK; no emulator. |
| **Instrumented tests** | `app/src/androidTest/java/…` | Real model execution, mask quality against the ground-truth triples, colour fidelity, end-to-end workflows, Espresso UI flows. |
| **Static analysis** | — | `detekt` for Kotlin, Android Lint with a baseline for pre-existing debt. |

```bash
./gradlew :app:testFullDebugUnitTest          # JVM tests
./gradlew :app:connectedFullDebugAndroidTest  # instrumented (device or emulator)
./gradlew detekt :app:lintFullDebug           # static analysis
./gradlew jacocoTestReport                    # coverage (full flavor)
```

A single instrumented class:

```bash
./gradlew :app:connectedFullDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=de.konradvoelkel.android.autokorrektur.ml.MaskQualityBenchmarkTest
```

CI (`.github/workflows/ci.yml`) runs lint, unit tests and the emulator suite against `full`, then
builds the `core` release bundle.

## 2. Invariants worth knowing about

These are the assertions that encode hard-won knowledge; the test files themselves are the
reference for the exact thresholds.

- **Mask polarity and colour spaces** — `ColorSpacePreservationTest`, `VehicleMaskSegmentationTest`:
  RGBA↔RGB conversions must not swap channels, and the mask convention in `ARCHITECTURE.md` §2 holds
  end to end.
- **Untouched pixels stay untouched** — `InpaintingQualityBenchmarkTest`: outside the car mask the
  output must match the input to a high PSNR floor.
- **Shadows and clutter** — `VehicleShadowSegmentationTest`, `MultiVehicleClutteredSceneTest`: cast
  shadows and tire contact points are removed without eating the pavement, and multiple vehicles
  are detected as distinct instances.
- **Lifecycle and failure paths** — `RotationLifecycleInferenceTest` (rotation mid-inference keeps
  the result, runs inference once), `ServerSdxlApiFallbackTest` (network failure preserves the
  daily quota).
- **Localization contract** — `StringResourceLocalizationTest`: `values/strings.xml` is English and
  complete, `values-de/` a complete override, no `values-en/`, placeholders match. Strict since
  2026-09-21; it exists because the two files once covered complementary halves of the key space,
  so every locale other than de/en got a mixed-language UI.
- **Diagnostics** — `TelemetryStoreTest`: JSON Lines encoding, the size cap, and that the store
  degrades to a no-op when disabled. On a device: menu → Diagnostics → on, then
  `adb shell run-as de.konradvoelkel.android.autokorrektur cat files/telemetry/events.jsonl`.

## 3. Hardware caveat

x86_64 emulators software-emulate NNAPI and translate arm64 code, so delegate fallback and native
crashes behave differently there than on real hardware — the TFLite interpreter, for instance,
segfaults under arm64 translation on the Pixel AVD. Model-execution changes need a physical ARM64
device before release; every Espresso suite additionally asserts that no error Snackbar appears on
launch, which catches initialization failures that would otherwise pass silently.

## 4. Proposed: config-matrix screenshots and accessibility checks

Field testing surfaced "lots of minor issues, too much to report" — presentation bugs across
locale × theme × width, which no one sweeps by hand. Two cheap additions would catch them:
[Paparazzi](https://github.com/cashapp/paparazzi) golden screenshots rendered on the JVM across
those axes, and AndroidX's `AccessibilityChecks` enabled for the existing Espresso suite (one line,
flags touch targets under 48 dp, missing content descriptions, poor contrast). Neither is
implemented yet.
