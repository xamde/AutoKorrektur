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

```bash
# Build Debug APK
./gradlew assembleDebug

# Run Unit Tests (75+ passing)
./gradlew testDebugUnitTest

# Build Release APK (with ProGuard/R8 minification)
./gradlew assembleRelease

# Run Static Analysis (Detekt)
./gradlew detekt

# Run Backend Service Tests
uv run --directory backend pytest --cov=.
```

---

## 📚 Academic Attribution & Credits

- **Till Schellscheidt**: Bachelor Thesis *"Autokorrektur – Automatisierte Objektersetzung in Fotos"*, providing the core research foundation for automated vehicle removal.
- **Benjamin Beckers**: Creator of the original web implementation ([BenB2/AutoKorrektur](https://github.com/BenB2/AutoKorrektur)).
- **Konrad Völkel**: Android architecture, AR video pipeline, progressive tile inpainting, and agentic engineering.

---

## 📄 Documentation

- [TODO-for-human.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/TODO-for-human.md) — Detailed human testing instructions on physical phones, thesis review checklist, and launch steps.
- [docs/PLAY_STORE_LISTING.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/docs/PLAY_STORE_LISTING.md) — Google Play Store metadata and descriptions in German & English.
- [PRIVACY_POLICY.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/PRIVACY_POLICY.md) — GDPR/DSGVO privacy policy and zero-storage data terms.
- [backend/DEPLOY_FRANKFURT.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/backend/DEPLOY_FRANKFURT.md) — Step-by-step Frankfurt cloud server deployment guide.
- [TESTING.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/TESTING.md) — Testing guidelines and benchmark metrics.

---

## ⚖️ License

The licensing of this project is governed by its machine learning dependencies (YOLOv11-seg licensed under GNU AGPLv3).  
Therefore, this project is licensed under the **GNU AGPLv3 License**. See [LICENSE](file:///home/konrad/files/work/__drafts/AutoKorrektur/LICENSE) for details.
