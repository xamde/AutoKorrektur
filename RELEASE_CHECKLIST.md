# Android Release Preparation Checklist

Engineering-side checklist for building, signing and publishing **AutoKorrektur** to Google Play.
`core` is the flavor that goes to the Store (see [docs/PRODUCT_TIERS.md](docs/PRODUCT_TIERS.md));
the owner keeps the account-side steps (keystore backup, Console forms) in a private note.

Status as of 2026-09-24.

---

## 1. Build configuration & security
- [x] R8 code shrinking and resource shrinking enabled in the release build type.
- [x] ProGuard rules preserve the OpenCV, ONNX Runtime and TFLite JNI symbols (`app/proguard-rules.pro`).
- [x] Release keystore generated and stored outside the repository; `keystore.properties` (gitignored)
      supplies it, so `bundleCoreRelease` is signed automatically and no key material is in git.
- [x] `core`/`plus` ship without the `INTERNET` permission — keeps the promise the privacy policy makes.

## 2. Metadata & versioning
- [x] `versionCode` / `versionName` are derived from git (`rev-list --count`, `describe --tags`); nothing to hand-edit.
- [ ] **Tag the release** (`git tag -a v1.0.0 …`) before building, or the version name reads `<old-tag>-N-gsha`.
- [x] `targetSdk = 36`, `compileSdk = 37` — re-check against Play's current minimum at upload time.
- [x] Strings are complete in German and English; `StringResourceLocalizationTest` fails CI if they drift.
- [x] App icon carries the brand colour (`app/src/main/res/drawable/ic_launcher_*`, regenerated 2026-09-24).

## 3. Privacy, licensing & disclosures
- [x] Privacy policy hosted at <https://autokorrektur.org/privacy> (German, binding) and `/privacy-en`.
      Rendered from `PRIVACY_POLICY.md` by `site/build.sh`, so the hosted text cannot drift.
- [x] AGPLv3 notice and third-party licences listed in the app under "About & Licenses".
- [ ] Play Console **Data Safety form**: for `core` the answer to "collect or share user data" is **No**
      — no internet permission, no accounts, no analytics; the opt-in diagnostics file never leaves the
      device unless the user exports it (`PRIVACY_POLICY.md` §5).

## 4. Listing assets
- [x] Short and full descriptions (German + English) in [docs/PLAY_STORE_LISTING.md](docs/PLAY_STORE_LISTING.md),
      trimmed to what `core` actually does — advertising absent features violates Play policy.
- [x] Hi-res icon 512×512 (`media/play_store_assets/`, generated from `site/icons/icon.svg`).
- [ ] Feature graphic 1024×500 — the committed one is still the old indigo design.
- [ ] Screenshots **from the `core` build** — the three committed ones show AR, engine tiers and
      multi-layout export, none of which `core` has.

## 5. Build, test, upload
- [ ] `scripts/fetch_assets.sh` first — the models are not in git (see `scripts/assets.manifest`).
- [ ] `./gradlew :app:bundleCoreRelease` → `app/build/outputs/bundle/coreRelease/app-core-release.aab`.
- [ ] Smoke-test the release build on a physical device (`:app:installCoreRelease`; `core` is arm64-only).
- [ ] Upload to the **Internal Testing** track, read the Pre-launch Report, then promote to **Production**.
