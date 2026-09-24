# AutoKorrektur — Active Roadmap & Next Milestones

> **Current Version**: 1.0.0 (Release Candidate)  
> **Status**: Core ML engine, CameraX AR live viewfinder, MediaCodec video pipeline, progressive tile inpainting, security, GDPR zero-storage, and CI/CD fully implemented and passing 100% of test suites.  
> **Historical Archive**: See [docs/ARCHIVE_TODO.md](docs/ARCHIVE_TODO.md) for completed milestones M1–M8 and Phases 1–4.

---

## 🎯 Active Milestones

### 🏷️ Milestone 0: AutoKorrektur 2.0 and the move to an own repository
The [web version by Benjamin Beckers](https://github.com/BenB2/AutoKorrektur) is **AutoKorrektur 1.0**;
this native Android rewrite is **AutoKorrektur 2.0**.
- [x] **REPO-01. Slim the repository** — binaries, owner-only notes and the cloud backend are out;
      a clone is ~13 MB, the tracked tree under 3 MB (2026-09-24).
- [x] **REPO-02. Rebrand to 2.0** — README lineage, changelog headings, version fallback, store
      listing and website (2026-09-24).
- [x] **REPO-03. Canonical repository is now `konradvoelkel/AutoKorrektur`** (2026-09-24), a fork of
      `xamde/AutoKorrektur`, which stays as the historical upstream. The repo was already a fork of
      [BenB2/AutoKorrektur](https://github.com/BenB2/AutoKorrektur), so nothing was lost by forking
      again; a fresh, network-detached repository remains an option for a later major version.
- [ ] **REPO-04. Ask GitHub Support to garbage-collect the fork network.** The pre-rewrite objects
      are still reachable by SHA (`.../commit/<old-sha>`), so `TODO-for-human.md` and
      `HUMAN_RELEASE_CHECKLIST.md` can still be fetched from old commits. Only a support-side GC (or
      a fresh repository pushed from the rewritten history) removes them.

### 🏙️ Milestone 1: Field Testing & Data Collection
- [ ] **FT-01. Physical Field Testing on Device**
  - Walk through real urban environments (residential street, commercial parking, mixed bike/pedestrian zones).
  - Execute the test scenarios in [docs/FIELD_TESTING_AND_DATA_COLLECTION.md](docs/FIELD_TESTING_AND_DATA_COLLECTION.md) (the owner also keeps a private, device-specific walkthrough outside this repo).
- [ ] **FT-02. Batch Telemetry & CSV Metric Collection**
  - Run multi-photo batch processing across varied lighting conditions and export execution CSVs for performance review.
  - Since 2026-09-21 every flavor (incl. `core`) also has opt-in on-device diagnostics: menu → Diagnostics → switch on, use the app, Export (share sheet) → `autokorrektur-diagnostics-<date>.jsonl` with per-stage timings, AR fps, export durations and crash lines. Off by default; see `PRIVACY_POLICY.md` §5.
- [ ] **FT-03. Social Media Split Export Trials**
  - Generate split cards, 4:5 carousels, and animated sweep MP4s on real street photos to verify Instagram readiness.

---

### 🚀 Milestone 2: Google Play Store Release
- [ ] **REL-01. Google Play Console Listing Setup**
  - Paste prepared German & English metadata from [docs/PLAY_STORE_LISTING.md](docs/PLAY_STORE_LISTING.md).
- [x] **REL-02. Privacy Policy Hosting** — live since 2026-09-21 at https://autokorrektur.org/privacy (`/privacy-en` English); redeploy with `site/deploy.sh`. Still to do by hand: paste the URL into Play Console → App content.
- [ ] **REL-03. Release App Bundle Generation**
  - Build signed `.aab` bundle via `./gradlew bundleCoreRelease` (`core` is the Play Store flavor) and upload to Play Console Internal Testing track.

---

### ☁️ Milestone 3: Optional cloud inpainting (deferred)
The SDXL service lives in [konradvoelkel/autokorrektur-backend](https://github.com/konradvoelkel/autokorrektur-backend)
and is not part of any published build. Reviving it means, in this repo: point `BACKEND_URL`
(release build type) at the live host, and update the privacy policy, the Data Safety answers and
the listing copy **first** — all three currently state that the published app has no network access.
