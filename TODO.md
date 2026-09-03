# AutoKorrektur — Active Roadmap & Next Milestones

> **Current Version**: 1.0.0 (Release Candidate)  
> **Status**: Core ML engine, CameraX AR live viewfinder, MediaCodec video pipeline, progressive tile inpainting, security, GDPR zero-storage, and CI/CD fully implemented and passing 100% of test suites.  
> **Historical Archive**: See [docs/ARCHIVE_TODO.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/docs/ARCHIVE_TODO.md) for completed milestones M1–M8 and Phases 1–4.

---

## 🎯 Active Milestones

### 🏙️ Milestone 1: Field Testing & Data Collection
- [ ] **FT-01. Physical Field Testing on Device**
  - Walk through real urban environments (residential street, commercial parking, mixed bike/pedestrian zones).
  - Execute test scenarios outlined in [TODO-for-human.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/TODO-for-human.md) and [docs/FIELD_TESTING_AND_DATA_COLLECTION.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/docs/FIELD_TESTING_AND_DATA_COLLECTION.md).
- [ ] **FT-02. Batch Telemetry & CSV Metric Collection**
  - Run multi-photo batch processing across varied lighting conditions and export execution CSVs for performance review.
- [ ] **FT-03. Social Media Split Export Trials**
  - Generate split cards, 4:5 carousels, and animated sweep MP4s on real street photos to verify Instagram readiness.

---

### 🚀 Milestone 2: Google Play Store Release
- [ ] **REL-01. Google Play Console Listing Setup**
  - Paste prepared German & English metadata from [docs/PLAY_STORE_LISTING.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/docs/PLAY_STORE_LISTING.md).
- [ ] **REL-02. Privacy Policy Hosting**
  - Publish [PRIVACY_POLICY.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/PRIVACY_POLICY.md) to GitHub Pages or project website.
- [ ] **REL-03. Release App Bundle Generation**
  - Build signed `.aab` bundle via `./gradlew bundleCoreRelease` (`core` is the Play Store flavor) and upload to Play Console Internal Testing track.

---

### ☁️ Milestone 3: Community Cloud Inpainting (Optional Frankfurt Backend)
- [ ] **SRV-01. Deploy Docker Compose to German VPS**
  - Follow [backend/DEPLOY_FRANKFURT.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/backend/DEPLOY_FRANKFURT.md) to launch the SDXL backend with automatic SSL and Redis rate limiting.
- [ ] **SRV-02. Configure Production Backend URL**
  - Update `BACKEND_URL` in `app/build.gradle.kts` release build type with live production domain.
