# Privacy Policy for AutoKorrektur

**Effective Date:** July 30, 2026

AutoKorrektur ("we", "our", or "us") is committed to protecting your privacy. This Privacy Policy explains how our mobile application handles your data.

---

## 1. Hybrid ML Architecture: On-Device & Server SDXL

AutoKorrektur uses a hybrid machine learning architecture to offer both maximum privacy and premium editing quality.

### 1.1 Local On-Device Processing (Default)
By default, AutoKorrektur performs all image processing, vehicle detection (YOLO), and image inpainting (MI-GAN) **entirely on your device**. 
- **No Data Uploads:** Your photos, edited images, and personal data are **never uploaded** to external servers when using this default mode.
- **Offline Capable:** The application functions fully offline without an internet connection.

### 1.2 Premium Edit via Server SDXL (Opt-In)
We offer a "Premium Edit" feature that utilizes a high-resolution SDXL model hosted on our secure servers. **This feature is strictly opt-in.**
- If you choose to enable this feature, the app will securely upload your original photo and a locally computed vehicle mask to our backend server (`api.autokorrektur.konradvoelkel.de`).
- **Strict GDPR Compliance & No Data Retention:** Images are processed entirely in memory on our servers. As soon as the processed image is returned to your device, **all uploaded images and data are immediately permanently deleted from the server**. 
- We do not store your photos, nor do we use them for training AI models.

---

## 2. Information We Collect

**We do not collect, store, transmit, or share any personal information or usage data.**

- **No Account Needed:** AutoKorrektur does not require user registration or accounts.
- **Anonymous Rate Limiting:** When using the Premium Edit feature, a randomly generated, anonymous device UUID is used solely to enforce daily usage limits to prevent abuse of our servers. This UUID cannot be linked to your identity.
- **No Analytics / Telemetry:** We do not track user behavior, analytics, or app diagnostics.
- **No Advertising Trackers:** The app contains zero third-party advertising SDKs or tracking code.

---

## 3. Device Access & Permissions

AutoKorrektur only accesses device capabilities necessary for user-initiated features:

- **Photos & Media Access:** Used solely to let you select photos for editing and to save processed images to your device gallery. We use the Android System Photo Picker (`PickVisualMedia`), which allows photo selection **without granting broad storage read permissions**.
- **Internet Access:** Required **only** if you explicitly opt-in to use the Premium Edit (Server SDXL) feature.

---

## 4. Contact Us

If you have any questions or feedback regarding this Privacy Policy, please contact us at:
- **Email:** support@autokorrektur-app.de
- **Website:** https://autokorrektur-app.de
