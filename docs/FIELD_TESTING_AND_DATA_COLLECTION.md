# 🏙️ Field Testing, Data Collection & Evaluation Protocol

This guide outlines the protocol for executing real-world field tests with **AutoKorrektur**, collecting empirical datasets across varied urban environments, recording diagnostic performance telemetry, and evaluating inpainting fidelity for public advocacy and scientific analysis.

---

## 1. Field Testing Objectives

The goal of field testing is to evaluate AutoKorrektur in real urban lighting conditions, varied vehicle geometries, and dynamic pedestrian environments:

1. **Live AR Passthrough Stability**: Test real-time car removal while walking through streets at 30–60 FPS.
2. **5-Second AR Video Capture**: Record real-world video snippets and evaluate temporal inpainting stability.
3. **Full Native Resolution vs Fast Preview**: Test high-resolution progressive inpainting on complex street corners with parked cars, delivery vans, and bicycles.
4. **Edge Cases**: Evaluate performance under direct sunlight glare, heavy shadows, wet asphalt reflections, snow, and dense car clusters.

---

## 2. In-App Data Collection & Diagnostic Recording

AutoKorrektur includes automated diagnostic logging, artifact caching, and batch CSV performance export:

### A. Batch Performance & Metric Logging
1. Open the app $\rightarrow$ Switch to **Studio** or **Batch Mode**.
2. Pick a folder of field test photos.
3. Once batch inference completes, tap **"CSV exportieren"**.
4. The app generates a structured CSV file in the device's `Downloads/` directory:
   ```csv
   filename,model,scoreThreshold,maskUpscale,maskDownshift,downscaleMp,inferenceTimeMs,timestamp
   IMG_20260815_120101.jpg,YOLOv11s-seg,0.25,1.20,0.02,No Scaling,2450,1786780861000
   IMG_20260815_120145.jpg,YOLOv11s-seg,0.25,1.20,0.02,No Scaling,2890,1786780905000
   ```

### B. Capturing Real-Time Video & Before/After Pairs
- **AR Video Clips**: Long-press the shutter in AR mode. The original 30 FPS video and the post-processed car-free HQ MP4 are saved automatically to your device's Movies/AutoKorrektur folder.
- **Before/After Split Cards**: Use the **Instagram & Social Export** dialog (`📸 Split-Karte` or `🔄 2-Slide Karussell`) to generate synchronized visual comparison pairs.

### C. Extracting Device Telemetry via ADB
To inspect low-level hardware performance, memory usage, and execution logs from your computer:

```bash
# 1. Stream live application logs
adb logcat -s AutoKorrektur:* AndroidRuntime:*

# 2. Pull all captured photos, masks, and videos to your computer
adb pull /sdcard/Pictures/AutoKorrektur/ ./field_test_data/photos/
adb pull /sdcard/Movies/AutoKorrektur/ ./field_test_data/videos/
adb pull /sdcard/Download/ ./field_test_data/logs/

# 3. Check memory & GPU footprint during active inpainting
adb shell dumpsys meminfo de.konradvoelkel.android.autokorrektur
```

---

## 3. Systematic 5-Criteria Evaluation Framework

Following the methodology established by Schellscheidt (2024) and Beckers (2025), score each test capture on a 1–5 scale:

| Criterion | Aspect Evaluated | Target Score |
|---|---|:---:|
| **1. Instanzsegmentierung** | Did YOLOv11 detect 100% of vehicles without cutoffs or missing cars? | $\ge 4.5$ |
| **2. Realismus** | Are generated road textures, paving stones, and greenery plausible? | $\ge 3.5$ |
| **3. Konsistenz (Seams)** | Is the boundary transition between untouched background and inpaint seamless? | $\ge 4.0$ |
| **4. Natürlichkeit** | Does the overall scene look like a genuine, believable photo? | $\ge 3.5$ |
| **5. Geschwindigkeit** | Did on-device processing complete within acceptable time? | Fast: $<1\text{s}$<br>High-Res: $<6\text{s}$ |

---

## 4. Field Testing Checklist for Activists & Researchers

- [ ] **Pre-Trip Check**:
  - Phone battery $\ge 70\%$.
  - Latest AutoKorrektur APK installed.
  - Storage space $\ge 2\text{ GB}$ available for video and high-res captures.
- [ ] **In the Field**:
  - Test 1: Typical residential street with parked cars along sidewalk.
  - Test 2: Multi-vehicle cluster (commercial street or parking lot).
  - Test 3: Mixed active mobility scene (cars parked next to parked bicycles / pedestrians).
  - Test 4: Dynamic lighting (bright sunlight vs deep tree canopy shadows).
  - Test 5: 5-second AR video snippet while walking slowly along sidewalk.
- [ ] **Post-Trip Evaluation**:
  - Export CSV benchmark logs from Batch mode.
  - Review captures in `VisionGalleryBottomSheet`.
  - Rate samples against the 5-criteria rubric.
