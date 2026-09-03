# 📋 TODO for Human: Release, Testing & Attribution Guide

**App:** AutoKorrektur (Android) – Version 1.0.0  
**Target Persona:** Mobility Activist & Urban Visionary ("Die Stadt ohne Autos sehen")  
**Latest Build Status:** ✅ Built, Tested (75+ Unit Tests Passing), and Installed on Physical Pixel 10 Pro (`192.168.178.117:44077`).

---

## 🎯 1. Detailed Manual Testing Instructions (On Physical Phone)

Take your **Pixel 10 Pro** outside (or point it at a street with parked cars / a car photo on a screen) and follow these 6 test scenarios to verify full functionality:

---

### 🧪 Test 1: Live AR Mode & 30–60 FPS Viewfinder
1. **Launch**: Open **AutoKorrektur** on your Pixel 10 Pro. The app starts directly into AR camera mode.
2. **Detection & Passthrough**: Point the camera at a parked car.
   - *Expected Result*: The camera viewfinder remains buttery smooth (30–60 FPS native hardware passthrough). An asynchronous inpainting patch renders seamlessly over the car, making it disappear into pavement/background texture.
   - *Status Badge*: Top-left badge shows `● 30 FPS Camera • Active AR Layer`.
3. **Reset**: Tap the **RESET** button at the top right to clear the temporal background buffer.
4. **Shutter Snapshot**: Tap the white circular shutter button once.
   - *Expected Result*: The app composites the high-res camera frame with the car-free overlay and saves it to your gallery. A Snackbar appears with an `OPEN IN STUDIO` shortcut.

---

### 🧪 Test 2: 5-Second AR Video Snippets & HQ Post-Processing
1. **Record**: In AR mode, **long-press and hold** the shutter button.
   - *Expected Result*: The shutter turns red, the top badge shows `🔴 RECORDING (5s Video Snippet)`, and a circular red progress ring smoothly fills around the shutter button for 5 seconds.
2. **HQ Inpainting Transition**: When 5 seconds elapse, `VideoPreviewActivity` automatically opens.
   - The original raw video starts playing immediately while the progress bar shows `Inpainting Frame X/Y` as YOLO and MI-GAN process the clip frame-by-frame with optical flow stabilization.
3. **Playback & Toggle**: Once processing completes:
   - The car-free video loops automatically.
   - Tap the top-right button `AUTOFREI` $\leftrightarrow$ `VORHER (RAW)` to toggle between the original street and the car-free video in real time.
4. **Share & Save**:
   - Tap **Speichern** to save the MP4 to your device (`Movies/AutoKorrektur`).
   - Tap **Instagram Reels** to launch the Android share sheet for direct posting to Instagram Reels / Stories.

---

### 🧪 Test 3: Studio Mode & 3-Tier Quality Switcher
1. **Open Studio**: From AR mode, tap the **Studio** floating action button (bottom right) or pick an existing street photo from your gallery.
2. **Quality Tier Switching**:
   - ⚡ **Fast On-Device**: Instant single-pass inpainting.
   - 💎 **High-Res Progressive**: Full sensor resolution multi-tile inpainting. Tap **Inpaint** and watch the vehicles disappear tile-by-tile in real-time with Gaussian feathered borders on the slider!
   - ☁️ **Cloud SDXL**: Tap the Cloud chip. The GDPR consent dialog appears explaining the Frankfurt zero-storage policy. Once accepted, subtle badges appear: `🔒 Frankfurt, Germany` and `⚡ 1/2 Free Enhancements remaining`.
3. **Interactive Slider**: Drag the vertical split handle left and right on the `BeforeAfterSliderView` to inspect the clean road surface.

---

### 🧪 Test 4: Interactive Manual Mask Brush & Eraser (Pinsel / Radierer)
1. **Open Brush**: In Studio mode with an image loaded, tap the **🖌️ Pinsel** button next to "Bild auswählen".
2. **Paint Custom Mask**:
   - With `🖌️ Pinsel` selected, use your finger to draw over deep car shadows or window reflections in semi-transparent red.
   - Adjust the **Größe** slider (10dp to 100dp) to test fine vs broad strokes.
3. **Eraser Tool**:
   - Tap `🧹 Radierer (Fahrrad/Person)`.
   - Erase over a bicycle or pedestrian located near the car to protect them from being erased during inpainting.
4. **Apply**: Tap **Fertig**. Inpainting runs immediately using your customized mask.

---

### 🧪 Test 5: Instagram & Social Media Multi-Layout Export
1. In Studio mode with an inpainted image ready, tap **📸 INSTAGRAM EXPORT**.
2. **Test 1: Split-Bild (Side-by-Side)**:
   - Select `📸 Split-Bild` $\rightarrow$ Choose `1:1 Quadrat` or `9:16 Story`.
   - Tap `Exportieren & Teilen`. Check that the resulting JPEG has clean "VORHER" and "AUTOFREI" pill badges.
3. **Test 2: 2-Slide Swipe-Karussell**:
   - Select `🔄 2-Slide Swipe-Karussell` $\rightarrow$ Choose `4:5 Portrait Feed`.
   - Tap `Exportieren & Teilen`. Verify two synchronized slides (`1/2 VORHER`, `2/2 AUTOFREI`) are shared.
4. **Test 3: Animierter Video-Slider (Reels/Story)**:
   - Select `🎬 Animierter Video-Slider` $\rightarrow$ Choose `9:16 Story / Reel`.
   - Tap `Exportieren & Teilen`. Verify a smooth 3.5-second looping MP4 video is rendered with the comparison line sweeping back and forth.

---

### 🧪 Test 6: Vision Gallery & Batch CSV Export
1. In AR mode, tap the **thumbnail card** at the bottom-left.
2. The `VisionGalleryBottomSheet` slides up showing past captures with timestamps and quick-open options.
3. In Studio mode, check the **Batch-Modus** switch, pick 3+ photos, let the pipeline run, and tap **CSV exportieren** to verify execution metrics and speed logs.

---

## 📚 2. Academic Attribution: Thesis Review (Completed & Verified ✅)

The bachelor theses of **Till Schellscheidt** and **Ben Beckers** (Heinrich-Heine-Universität Düsseldorf) have been thoroughly reviewed and integrated into the project's documentation, code attribution, and app metadata:

- [x] **Till Schellscheidt (Feb 2024)**: *"Autokorrektur – Automatisierte Objektersetzung in Fotos"* (Supervisor: Dr. Konrad Völkel). Credited for conceptualizing automated car removal, two-cycle latent diffusion prompting, and foundational evaluation criteria.
- [x] **Ben Beckers (Mai 2025)**: *"Autokorrektur – Inpainting auf mobilen Endgeräten"* (Supervisors: Dr. Konrad Völkel, Dr. Markus Brenneis; [github.com/BenB2/AutoKorrektur](https://github.com/BenB2/AutoKorrektur)). Credited for shifting to on-device/in-browser inference, YOLOv11-seg evaluation, MI-GAN integration, and 1.2x mask scaling / vertical shadow extension.
- [x] **README & Metadata**: Full academic lineage and BibTeX citations added to [`README.md`](file:///home/konrad/files/work/__drafts/AutoKorrektur/README.md) and [`docs/PLAY_STORE_LISTING.md`](file:///home/konrad/files/work/__drafts/AutoKorrektur/docs/PLAY_STORE_LISTING.md).
- [x] **In-App Dialog**: Updated `R.string.about_dialog_content` in German and English with explicit academic credits.

---

## 🚀 3. Play Store Publishing Checklist

When you are ready to upload the app to the **Google Play Console**:

1. **Store Listing Copy**:
   - Use the ready-to-paste German and English descriptions in [`docs/PLAY_STORE_LISTING.md`](file:///home/konrad/files/work/__drafts/AutoKorrektur/docs/PLAY_STORE_LISTING.md).
2. **Privacy Policy**:
   - Host [`PRIVACY_POLICY.md`](file:///home/konrad/files/work/__drafts/AutoKorrektur/PRIVACY_POLICY.md) on your website or GitHub Pages (`https://xamde.github.io/AutoKorrektur/privacy`) and paste the URL in Google Play Console $\rightarrow$ App Content $\rightarrow$ Privacy Policy.
3. **Cloud Backend Deployment (Optional)**:
   - Follow [`backend/DEPLOY_FRANKFURT.md`](file:///home/konrad/files/work/__drafts/AutoKorrektur/backend/DEPLOY_FRANKFURT.md) to launch the SDXL container on a Frankfurt VM if you want live cloud processing for non-local devices.
4. **App Bundle (`.aab`)**:
   - Generate the release bundle (`core` is the Play Store flavor, see `docs/MVP_FEATURE_FLAG_PLAN.md`):
     ```bash
     ./gradlew bundleCoreRelease
     ```
   - Located at `app/build/outputs/bundle/coreRelease/app-core-release.aab`.
