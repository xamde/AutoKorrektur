# Product Strategy & Persona Specification — AutoKorrektur

> **Product Vision**: AutoKorrektur is a minimalistic, precision storytelling tool for mobility activists and urban advocates to visualize, document, and publish a car-free future directly from their mobile devices.

---

## 1. Primary Persona: The Mobility & Urban Transformation Activist

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   Primary Persona: The Mobility Activist                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│ • Core Motivation: Show how public space, plazas, and streets look without cars. │
│ • Primary Interaction: Real-Time AR "Magic Lens" while walking the city.         │
│ • Publication Loop: Capture hero scenes -> Select Social Export -> Cloud Enhance.│
│ • Voice & Tone: Minimalistic, technical, direct, zero fluff.                     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Key Workflow Loops:

1. **The AR Walk (Capture Mode)**:
   - App opens immediately into the **Live AR Camera (30 FPS)**.
   - Activist holds up phone like a magic viewfinder: parked and moving cars are erased in real-time.
   - Tapping the bottom shutter button captures a high-resolution frame with on-device preview inpainting and saves it to the **Vision Gallery**.

2. **The Vision Gallery (Browse Mode)**:
   - Tap the bottom-left thumbnail to view past car-free shots taken with the app.
   - Select any shot to open directly in the **Studio / Post Prep Mode**.

3. **Instagram & Social Post Prep (Export Mode)**:
   - **Interactive Split Slider**: Real-time before/after comparison.
   - **Clean Social Formats**: 1:1 Square, 4:5 Feed Portrait, 9:16 Reel / Story.
   - **No Fluff / No Watermarks**: Clean, high-impact aesthetic.

4. **Privacy-First Cloud Enhancement (Publication Mode)**:
   - Fast local on-device preview is unlimited and offline.
   - 1-Tap **"Enhance for Publication (Server)"** runs Stable Diffusion XL on high-resolution shots.
   - **GDPR & Privacy Guarantee**: `🔒 Processed in-memory in Frankfurt, Germany. Zero persistent storage.`
   - **Quota Transparency**: Clean indicator showing monthly high-res server quota (e.g. `1/2 Free Server Enhancements remaining`).

---

## 2. Information Architecture & Navigation

```
                       ┌──────────────────────────────┐
                       │     1. Live AR Viewfinder    │ ◄── (Default App Entry)
                       │   (30 FPS Car-Free Magic)    │
                       └──────────────┬───────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
┌──────────────────────────────┐               ┌──────────────────────────────┐
│       2. Vision Gallery      │               │     3. Studio & Post Prep    │
│  (Past Shots & Batch Grid)   ├──────────────►│ (Before/After Slider & SDXL) │
└──────────────────────────────┘               └──────────────┬───────────────┘
                                                              │
                                                              ▼
                                               ┌──────────────────────────────┐
                                               │    4. Social Export Sheet    │
                                               │  (1:1, 4:5, 9:16 Video/Img)  │
                                               └──────────────────────────────┘
```
