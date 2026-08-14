# Customer Personas & UX Architecture — AutoKorrektur

This document establishes the user persona archetypes, jobs-to-be-done (JTBD), and user journey specifications guiding the AutoKorrektur interface overhaul.

---

## 1. Primary Customer Personas

```
                     ┌──────────────────────────────────────────────┐
                     │          AutoKorrektur User Personas         │
                     └──────────────────────┬───────────────────────┘
            ┌───────────────────────┬───────┴────────┬──────────────────────┐
            ▼                       ▼                ▼                      ▼
┌───────────────────────┐ ┌──────────────────┐ ┌──────────────┐ ┌──────────────────────┐
│ 1. Dealership Pro     │ │ 2. Real Estate   │ │ 3. Street/   │ │ 4. On-Location       │
│ (High-Volume Batch)   │ │    Architecture  │ │    Privacy   │ │    Creator (Live AR) │
│ - 50+ photos/session  │ │ - 1-2 hero shots │ │ - Social     │ │ - Instant point &    │
│ - Speed & automation  │ │ - Max precision  │ │   presets    │ │   capture            │
└───────────────────────┘ └──────────────────┘ └──────────────┘ └──────────────────────┘
```

### Persona 1: Automotive Dealership Photographer (Marcus)
- **Profile**: Vehicle inventory manager / professional automotive reseller.
- **Goal**: Ingest 30–100 dealer lot photos, remove customer cars in the background or surrounding lot clutter in seconds, and export a clean catalog.
- **Pain Points**: Editing one photo at a time is impossible. Needs automated batch ingestion, progress feedback, preset selection, and ZIP/CSV export.
- **Primary Mode**: **Batch Queue**.

### Persona 2: Real Estate & Architectural Photographer (Elena)
- **Profile**: Commercial photographer shooting residential homes, villas, and commercial facades.
- **Goal**: Remove 1 or 2 unsightly parked delivery vans or resident cars blocking the driveway or building entrance.
- **Pain Points**: Artifacts on pavement or building textures ruin the shot. Needs high-resolution inpainting (SDXL or LaMa), interactive before/after split slider, and fine mask touch-up.
- **Primary Mode**: **Studio Precision Editor**.

### Persona 3: Urban Street Photographer & Privacy Creator (Leo)
- **Profile**: Instagram/TikTok creator and street photographer.
- **Goal**: Clean up urban backgrounds, remove distracting cars, and export side-by-side comparison reels/posts in 1:1, 4:5, or 9:16 aspect ratios.
- **Pain Points**: Standard photo editors don't provide before/after comparison frames with custom branding or aspect presets.
- **Primary Mode**: **Social Comparison Export**.

### Persona 4: On-Location Mobile User (Sarah)
- **Profile**: Mobile-first user wanting to see the car vanish live on the screen before capturing the photo.
- **Goal**: Point the camera at a scene, walk around the stationary vehicle, and snap a clean photo without post-processing.
- **Primary Mode**: **Live AR Viewfinder**.

---

## 2. Navigation Architecture & Screen Design

To serve all 4 archetypes without cognitive clutter, AutoKorrektur uses a clean **Material 3 Bottom Navigation Bar**:

1. **Studio (`nav_studio`)**:
   - Single-image canvas with interactive Before/After comparison slider.
   - 1-Tap "Start Inpainting" (Local MI-GAN / LaMa or Cloud SDXL).
   - Social Export button with Instagram 1:1 / 4:5 / 9:16 aspect ratio sheet.

2. **Batch Queue (`nav_batch`)**:
   - Multi-image picker (up to 100 images).
   - Real-time progress bar with per-item status cards.
   - 1-Tap CSV and ZIP export.

3. **Live AR (`nav_ar`)**:
   - Direct launch of the 30 FPS CameraX real-time vehicle erasure viewfinder with live HUD.

4. **Settings & Quota (`nav_settings`)**:
   - Cloud SDXL free daily quota counter (5/5 remaining).
   - Model tier selection: **MI-GAN (Fast)** vs **LaMa (High-Fidelity)**.
   - GDPR consent management & About info.
