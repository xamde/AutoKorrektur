# AutoKorrektur: Autofreie Visionen & AR

> [!NOTE]
> 🤖 **An Experiment in Vibe Coding & Agentic Software Engineering**  
> This project is an exploratory experiment in **Vibe Coding** and **Agentic AI Engineering**. The entire Android application—from real-time OpenCV AR pipelines and CameraX video recording to on-device neural inpainting, progressive tile rendering, and Play Store release automation—was designed, implemented, benchmarked, and deployed through collaborative pair-programming between human vision and autonomous AI coding agents.

---

## 🌆 Über das Projekt (About AutoKorrektur)

**AutoKorrektur** is an Android application that automatically removes cars from live camera views and photographs using on-device machine learning and augmented reality.

This project is a native Android reimplementation and major architectural evolution of the original [AutoKorrektur Web Version](https://github.com/BenB2/AutoKorrektur) created by **Benjamin Beckers**, which was based on the Bachelor Thesis *"Autokorrektur – Automatisierte Objektersetzung in Fotos"* by **Till Schellscheidt**.

<table>
  <tr>
    <td><img src="media/image_1_with_car_640x640.png" alt="Vorher: Straße mit parkendem Auto" width="400"/></td>
    <td><img src="media/image_1_without_car_640x640.png" alt="Nachher: Autofreie Vision" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>Vorher (Originalaufnahme)</b></td>
    <td align="center"><b>Autofreie Vision (AutoKorrektur KI)</b></td>
  </tr>
</table>

The primary mission is to empower mobility activists, urban planners, and citizens to visualize what our cities could look like when public space is reclaimed from parked cars for people, greenery, and sustainable mobility.

Default processing is executed **100% on-device** without accounts or telemetry. Usage is free and open source.

---

## 🤖 The Vibe Coding & Agentic Architecture Paradigm

This codebase serves as a benchmark for how far modern **Agentic AI** and **Vibe Coding** can go when building production-grade mobile applications with complex native requirements:

1. **Vibe Coding to Native Execution**: High-level activist persona workflows, user interviews, and intuitive design goals ("vibe") translated directly into low-level C++/JNI OpenCV operations, Android MediaCodec hardware video encoders, and StateFlow architectures.
2. **Autonomous Tool Augmentation**: AI agents orchestrating compiler checks, running headless test suites, executing wireless ADB debugging (`adb -s 192.168.178.117:44077`), capturing physical Pixel screen artifacts, and resolving JNI lifecycle edge cases autonomously.
3. **Rigorous Verification Loop**: Combining prompt-driven engineering with deterministic engineering standards: strict ProGuard/R8 rules, 75+ unit and instrumented tests, automated detekt static analysis, and zero-storage GDPR compliance.

---

## 🌟 Key Features

### 1. 🎥 Live AR Viewfinder (30–60 FPS)
- **Decoupled Passthrough**: Native camera preview runs at full hardware framerate (30–60 FPS).
- **Asynchronous Inpainting Patch**: YOLO segmentation detects vehicles and overlays a transparent inpainting patch in real time.
- **Auto-Shadow Expansion**: Morphologically expands vehicle masks downwards to swallow contact shadows and ground puddle reflections automatically.

### 2. 🎬 5-Second AR Video Snippets & HQ Temporal Inpainting
- **Tactile Recording**: Long-press the shutter to record a 5-second video snippet with an animated circular progress ring.
- **Optical Flow Inpainting**: Offline frame-by-frame post-processing engine stabilized with temporal background accumulation to eliminate video flicker.
- **Hardware MP4 Encoding**: Generates 30 FPS H.264 MP4 videos ready for Instagram Reels and TikTok.
- **Instant Video Toggle**: Real-time Before/After toggle during looped playback.

### 3. 💎 3-Tier Studio Quality Engine
- ⚡ **Fast On-Device**: Instant single-pass inpainting (1–2 MP) for rapid preview.
- 💎 **High-Res Progressive (On-Device)**: Tile-based contextual inpainting at full sensor resolution with Gaussian alpha boundary feathering and live preview streaming.
- ☁️ **German Cloud SDXL**: In-memory zero-storage processing on ISO-certified Frankfurt servers for photorealistic poster-grade enhancements.

### 4. 🖌️ Interactive Manual Mask Brush & Eraser
- **🖌️ Pinsel**: Paint custom mask regions over dark shadows, tinted window reflections, or bike trailers.
- **🧹 Radierer**: Erase mask parts to protect bicycles, cargo bikes, and pedestrians.
- **Precision Slider**: Adjust brush size dynamically from 10dp to 100dp.

### 5. 📸 Instagram & Social Media Multi-Layout Export
- **📸 Split Cards**: Side-by-Side or Stacked Vorher/Autofrei comparison cards.
- **🔄 2-Slide Swipe Carousel**: Dual synchronized slides formatted for 4:5 portrait feeds.
- **🎬 Animated Sweep Video**: 3.5-second looping MP4 video where the comparison divider sweeps smoothly back and forth for Reels and Stories.

---

## 🛠️ Tech Stack

- **Android Client**: Kotlin, ONNX Runtime, OpenCV Android SDK 5, TensorFlow Lite, CameraX (Core, Camera2, Lifecycle, View, Video), AndroidX, Material3 Design.
- **Python Backend**: FastAPI, Uvicorn, Pytest, Docker, Caddy. See [backend/README.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/backend/README.md).
- **ML Models**:
  - Detection / Segmentation: **YOLOv11-seg** (On-Device TFLite/ONNX)
  - Inpainting: **MI-GAN** (On-Device) and **SDXL** (Optional Frankfurt Cloud Backend)

---

## 🧪 Development & Testing

The app ships as four product flavors (`core`/`plus`/`beta`/`full`, see
[`docs/MVP_FEATURE_FLAG_PLAN.md`](docs/MVP_FEATURE_FLAG_PLAN.md)) — Gradle tasks are
flavor-qualified accordingly. `full` reproduces the pre-flavor app exactly (every feature on, all
ABIs) and is the day-to-day dev target; `core` is the Play Store candidate.

```bash
# Build Debug APK (dev target: full flavor, or drop "Full" to build all 4 flavors at once)
./gradlew assembleFullDebug

# Run Unit Tests (75+ passing)
./gradlew testFullDebugUnitTest

# Build Release APK (with ProGuard/R8 minification) — core is the Play Store flavor
./gradlew assembleCoreRelease

# Run Static Analysis (Detekt)
./gradlew detekt

# Run Backend Service Tests
uv run --directory backend pytest --cov=.
```

---

## 📚 Academic Attribution & Scientific Lineage

AutoKorrektur is the result of continuous academic research conducted at the **Heinrich-Heine-Universität Düsseldorf (HHU)** under the supervision of **Dr. Konrad Völkel**:

### 1. Foundational Research & Concept
- **Author**: **Till Schellscheidt**
- **Thesis**: *"Autokorrektur – Automatisierte Objektersetzung in Fotos"* (Bachelorarbeit, Lehrstuhl für Dialog Systems and Machine Learning, HHU Düsseldorf, Februar 2024).
- **Core Contributions**:
  - Conceptualized automated vehicle removal for mobility activism and urban space reclamation.
  - Developed the two-cycle latent diffusion inpainting workflow with contextual negative/positive environmental conditioning.
  - Established the foundational vehicle bounding-box padding and initial 25 px mask shadow expansion rules.
  - Defined the qualitative 4-criteria evaluation framework (*Instanzsegmentierung*, *Realismus*, *Konsistenz*, *Natürlichkeit*).

### 2. On-Device Mobile Inpainting & Open-Source Web Version
- **Author**: **Ben Beckers** (Benjamin Beckers)
- **Thesis**: *"Autokorrektur – Inpainting auf mobilen Endgeräten"* (Bachelorarbeit, Institut für Informatik, HHU Düsseldorf, Mai 2025).
- **Repository**: [github.com/BenB2/AutoKorrektur](https://github.com/BenB2/AutoKorrektur)
- **Core Contributions**:
  - Shifted the paradigm from expensive cloud GPUs to 100% on-device inference using **ONNX-Runtime-Web (WASM)** and **OpenCV.js**.
  - Integrated and evaluated **YOLOv11-seg** model size tradeoffs (Nano/Small/Medium) on the Mapillary Vistas dataset.
  - Integrated **MI-GAN 512×512** (Picsart AI Research) for sub-5-second neural on-device inpainting.
  - Formulated the 1.2x mask scaling and 0.07/0.02 directional vertical shadow downshift transformations.

### 3. Native Android AR & Agentic Real-Time Pipeline
- **Architect & Supervisor**: **Dr. Konrad Völkel**
- **Core Contributions**:
  - Supervision and first examination of both bachelor theses at HHU Düsseldorf.
  - Native Android implementation with real-time **30–60 FPS CameraX AR passthrough**, hardware **MediaCodec H.264** video pipeline, **Progressive Tile Inpainting** with Gaussian alpha feathering, and **Temporal Background Plate Accumulation**.
  - Production engineering and autonomous multi-agent orchestration.

```bibtex
@bachelorthesis{schellscheidt2024autokorrektur,
  author       = {Till Schellscheidt},
  title        = {Autokorrektur -- Automatisierte Objektersetzung in Fotos},
  school       = {Heinrich-Heine-Universit{\"a}t D{\"u}sseldorf},
  year         = {2024},
  month        = {February},
  type         = {Bachelor's Thesis}
}

@bachelorthesis{beckers2025autokorrektur,
  author       = {Ben Beckers},
  title        = {Autokorrektur -- Inpainting auf mobilen Endger{\"a}ten},
  school       = {Heinrich-Heine-Universit{\"a}t D{\"u}sseldorf},
  year         = {2025},
  month        = {May},
  type         = {Bachelor's Thesis},
  url          = {https://github.com/BenB2/AutoKorrektur}
}
```

---

## 📄 Documentation Sitemap & Index

To navigate the comprehensive project documentation:

| Category | Document | Description |
|---|---|---|
| **🧪 Testing & Field Operations** | [TODO-for-human.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/TODO-for-human.md) | Step-by-step physical phone testing guide, camera walkthroughs & checklist. |
| | [docs/FIELD_TESTING_AND_DATA_COLLECTION.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/docs/FIELD_TESTING_AND_DATA_COLLECTION.md) | Field testing protocol, batch CSV metrics collection, and ADB telemetry. |
| | [docs/TESTING_INSIGHTS_FROM_THESES.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/docs/TESTING_INSIGHTS_FROM_THESES.md) | 5-criteria evaluation framework, failure mode taxonomy & edge cases. |
| | [TESTING.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/TESTING.md) | Testing guidelines, hardware matrix, unit & instrumented test suites. |
| **🏗️ Architecture & Engineering** | [ARCHITECTURE.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/ARCHITECTURE.md) | System architecture, CameraX AR loop, and MediaCodec video pipeline. |
| | [docs/IMAGE_PIPELINE_SPECIFICATION.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/docs/IMAGE_PIPELINE_SPECIFICATION.md) | ML pipeline specification, coordinate math, and tensor normalization. |
| | [walkthrough.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/walkthrough.md) | Implementation walkthrough of newly added features and UI components. |
| **🚀 Release & Play Store** | [docs/PLAY_STORE_LISTING.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/docs/PLAY_STORE_LISTING.md) | Google Play Store copy and metadata in German and English. |
| | [RELEASE_CHECKLIST.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/RELEASE_CHECKLIST.md) | Play Store release pre-flight checks, signing keys, and AAB bundle steps. |
| | [PRIVACY_POLICY.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/PRIVACY_POLICY.md) | GDPR/DSGVO privacy policy, on-device guarantees, and zero-storage terms. |
| | [CHANGELOG.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/CHANGELOG.md) | Version history, release tags, and feature changelog. |
| **☁️ Backend & Cloud** | [backend/README.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/backend/README.md) | FastAPI SDXL cloud inpainting server documentation. |
| | [backend/DEPLOY_FRANKFURT.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/backend/DEPLOY_FRANKFURT.md) | Deployment guide for German Frankfurt VPS with Docker & Caddy SSL. |
| **🧭 Product & Roadmaps** | [TODO.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/TODO.md) | Active milestone roadmap (Field Testing, Play Store, Cloud Backend). |
| | [docs/ARCHIVE_TODO.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/docs/ARCHIVE_TODO.md) | Historical archive of completed milestones M1–M8 and Phases 1–4. |
| | [docs/PERSONAS.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/docs/PERSONAS.md) | User personas (Mobility Activists, Lot Managers, Photographers). |

---

## ⚖️ License

The licensing of this project is governed by its machine learning dependencies (YOLOv11-seg licensed under GNU AGPLv3).  
Therefore, this project is licensed under the **GNU AGPLv3 License**. See [LICENSE](file:///home/konrad/files/work/__drafts/AutoKorrektur/LICENSE) for details.
