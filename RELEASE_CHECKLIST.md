# Android Release Preparation Checklist

This checklist tracks the tasks required to build, sign, and publish **AutoKorrektur** to the Google Play Store.

---

## 1. Build Configuration & Security
- [x] Enable R8 code shrinking and resource shrinking in `app/build.gradle.kts` (`isMinifyEnabled = true`, `isShrinkResources = true`).
- [x] Verify ProGuard rules in `app/proguard-rules.pro` preserve OpenCV, ONNX Runtime, and TFLite native JNI symbols.
- [ ] Generate production release key/keystore (`release.keystore`).
- [ ] Configure signing credentials safely using environment variables or `local.properties` (never commit keys to Git).

## 2. Metadata & Versioning
- [ ] Increment `versionCode` (e.g. `1` -> `2`) in `app/build.gradle.kts`.
- [ ] Update `versionName` (e.g. `"1.0.0"`) in `app/build.gradle.kts`.
- [ ] Confirm `targetSdk` is up to date with Google Play requirements.
- [ ] Review app title, icon, and strings across `res/values/strings.xml`.

## 3. Privacy, Licensing & Disclosures
- [ ] Confirm AGPLv3 license notices and source code disclosures are present in app metadata / about section.
- [ ] Complete Google Play Console **Data Safety Form** (disclose 100% on-device processing).
- [ ] Host public **Privacy Policy** URL covering camera/photo access.

## 4. Media & Play Store Listing Assets
- [ ] High-resolution app icon (512x512 PNG).
- [ ] Feature graphic (1024x500 PNG/JPEG).
- [ ] Phone & tablet screenshots showcasing car removal (before & after, landscape & portrait).
- [ ] Short description (max 80 characters) and full store description.

## 5. Build Artifacts & Internal Testing
- [ ] Build release Android App Bundle (`./gradlew :app:bundleCoreRelease` — `core` is the Play Store flavor, see `docs/MVP_FEATURE_FLAG_PLAN.md`).
- [ ] Test release build locally (`./gradlew :app:installCoreRelease`).
- [ ] Upload `.aab` to Google Play Console **Internal Testing** track.
- [ ] Review Google Play Pre-launch Report (Firebase Test Lab compatibility, performance, and accessibility checks).
- [ ] Promote to **Production** track.
