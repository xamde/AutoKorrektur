# AutoKorrektur — What Only You Can Do (Release Checklist)

Everything below requires a human decision, a Google account, or a physical signature — none
of it can be automated away. Ordered so each step unblocks the next. Cross-checked against the
actual repo state on 2026-09-03.

---

## 0. Already done for you (don't waste time on these)

- **`versionCode` / `versionName`**: `app/build.gradle.kts` already computes both automatically
  — `versionCode` from `git rev-list --count HEAD` (total commit count) and `versionName` from
  `git describe --tags --always`. `RELEASE_CHECKLIST.md` item 2 ("increment versionCode / update
  versionName") is stale — there is nothing to hand-edit.
- **R8/ProGuard, resource shrinking**: already enabled in the release build type.

## 1. Tag the release (2 minutes) — do this first

`git describe` only produces a clean `1.0.0`-style name if there's a matching tag on the commit
you build from. Right now the only tag in the repo is `working_app` (Nov 2025), so a release
build today would stamp something like `working_app-190-g3f8a1c2` as the version name in the
Play Store listing. Fix it:
```bash
git tag -a v1.0.0 -m "AutoKorrektur v1.0.0 — first Play Store release"
git push origin v1.0.0
```

## 2. Generate the release keystore (5 minutes)

`build.gradle.kts` already looks for `keystore.properties` at the repo root and wires up the
`release` signing config automatically if it's present — you just need to create it.
```bash
keytool -genkeypair -v -keystore release.keystore -alias autokorrektur \
  -keyalg RSA -keysize 4096 -validity 10000
```
You'll be prompted for a store password, a key password, and your name/org (used only inside the
certificate, not shown to users). Then create `keystore.properties` at the repo root:
```properties
storeFile=release.keystore
storePassword=<the password you just chose>
keyAlias=autokorrektur
keyPassword=<the password you just chose>
```
**Store the keystore file and its passwords in a password manager or offline backup.** If you
lose it, you can never update this app on the Play Store again under the same listing — Google
cannot reset this for you. I've already added `keystore.properties`, `*.keystore`, and `*.jks` to
`.gitignore` (they weren't excluded before — nothing was stopping you from accidentally
committing your signing key and passwords to the public GitHub repo).

## 3. Host the privacy policy (5 minutes — pick one)

**Quick (works today, zero setup):** use the raw GitHub URL directly — Play Console just needs
any stable, publicly reachable URL, it doesn't have to be pretty HTML:
```
https://raw.githubusercontent.com/xamde/AutoKorrektur/main/PRIVACY_POLICY.md
```
**Polished (matches the URL already referenced in TODO-for-human.md):** enable GitHub Pages —
repo Settings → Pages → Source: "Deploy from a branch" → Branch: `main`, folder `/docs` — then
add `docs/privacy.md` with a Jekyll permalink so it lands at exactly
`https://xamde.github.io/AutoKorrektur/privacy`:
```markdown
---
permalink: /privacy/
title: Datenschutzerklärung
---
<!-- paste PRIVACY_POLICY.md content here -->
```

## 4. Play Console Data Safety Form — draft answers

Based on the actual `PRIVACY_POLICY.md` content:

| Question | Answer |
|---|---|
| Does your app collect or share any user data? | **Yes** (photos/videos, only for the optional cloud tier) |
| Photos and videos | Collected: **No** (not retained) — mark as **processed ephemerally**: uploaded only when the user opts in to Cloud SDXL, held in RAM during inference, deleted immediately after. Shared with third parties: **No**. |
| Is data encrypted in transit? | **Yes** (HTTPS/TLS via Caddy automatic certificates) |
| Can users request data deletion? | **Not applicable** — no data is retained server-side to delete |
| Personal info (name, email, account) | **Not collected** — no accounts, no registration |
| Device or other identifiers | **Not collected** for tracking; Play Integrity attestation is used only for abuse/quota enforcement, not analytics |
| Analytics / advertising | **None** |
| Data collection is required or optional | The cloud tier (the only data-leaving-device path) is **opt-in**, gated behind an explicit GDPR consent dialog |

## 5. Store listing assets

- **Feature graphic, hi-res icon, screenshots**: generated this session, committed to
  `media/play_store_assets/` — see the separate note on which ones are launch-ready vs. which
  need a real on-device capture.
- **Short/full descriptions**: already written in `docs/PLAY_STORE_LISTING.md` (German + English)
  — just copy-paste into Play Console.

## 6. Build & upload

`core` is the flavor that goes to the public Play Store listing (see
`docs/MVP_FEATURE_FLAG_PLAN.md`) — `bundleRelease` alone now aggregates all four flavors, so
build `core` specifically:
```bash
./gradlew bundleCoreRelease
```
Output: `app/build/outputs/bundle/coreRelease/app-core-release.aab`. Upload to Play Console →
Internal Testing track first, review the automated Pre-Launch Report (accessibility/performance/
crash scan on real device farm hardware), then promote to Production when clean.

## 7. Decide your locale footprint (5-minute decision, not a task)

Your string resources currently only fully support German and English (see the new
`StringResourceLocalizationTest` and `TESTING.md` §8 for the details) — 51 strings silently
fall back to German text for any other locale. Either restrict the Play Console listing's
"Countries/regions" to markets where de/en is an acceptable default, or budget time to make
`values/strings.xml` locale-neutral before going fully worldwide. Not a blocker for a
DE/EN-market launch, but worth deciding deliberately rather than by accident.
