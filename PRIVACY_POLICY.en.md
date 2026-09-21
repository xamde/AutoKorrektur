# Privacy Policy – AutoKorrektur

**Last updated:** September 2026

_English translation of [PRIVACY_POLICY.md](PRIVACY_POLICY.md) for convenience; the German text is the binding one. Keep both in sync — `site/build.sh` renders each into the website (`/privacy`, `/privacy-en`)._

The developers of **AutoKorrektur** take the protection of your personal data and your privacy seriously. This policy explains the nature, scope and purpose of any processing of personal data inside our Android application.

---

## 1. Principle: on-device and data minimisation

AutoKorrektur follows **privacy by design and by default**:
- Using the app requires **no registration**, **no user account** and **no entry of personal identity data**.
- We use **no advertising SDKs**, **no tracking tools** and **no behavioural analytics services**.

---

## 2. App permissions

For its core functions the app needs the following device permissions:

1. **Camera (`android.permission.CAMERA`)**:
   - **Purpose**: solely to produce the live AR camera view and to take photos and 5-second video clips.
   - **Processing**: camera frames are processed live in your device's volatile memory. There is no hidden transfer in the background.

2. **Storage / media library (`READ_MEDIA_IMAGES`, `READ_MEDIA_VIDEO`)**:
   - **Purpose**: lets you pick existing photos from your gallery for studio inpainting and save the finished before/after results to your media folder.

---

## 3. Data processing with on-device AI

When you use the **AR mode**, the **Fast mode** or the **Progressive High-Res mode**:
- Object detection (YOLO) and AI inpainting (MI-GAN / progressive tile engine) run **entirely locally on your device's processor / NPU**.
- At no point do your image or video data leave your device.

---

## 4. Optional cloud processing (Cloud SDXL – Frankfurt, Germany)

AutoKorrektur offers the optional feature of refining images in photorealistic SDXL quality on a dedicated server.

- **Explicit consent (opt-in)**: this feature is off by default and requires your explicit consent through a GDPR consent dialog before first use.
- **Server location**: processing happens exclusively on servers in **Frankfurt am Main, Germany**, in strict compliance with the General Data Protection Regulation (GDPR).
- **Ephemeral processing (zero-storage policy)**: the uploaded image is processed only in the inpainting engine's working memory (RAM). Once inference is complete and the result has been sent back to your phone, the image is **deleted from memory immediately and irrecoverably**. Your images are neither logged nor written to disk.

---

## 5. Disclosure to third parties

Your data is neither sold nor passed on to advertising networks or other unauthorised third parties. When you share finished images or videos via Instagram or other apps, this happens solely through Android's standard share system (`Intent.ACTION_SEND`), over which you keep full control at all times.

---

## 6. Optional diagnostics (on the device only, opt-in)

To make the app fast and stable on as many devices as possible, you can switch on the recording of technical measurements under **"Diagnostics"** in the menu. It is **off by default**.

- **What is recorded**: compute time of each processing step, image size in pixels, the selected mode, the number of detected vehicles, success or the kind of error, the frame rate in AR mode, and once per session the app version, Android version, device manufacturer and model, memory size and processor cores. Plus a random installation ID that the app generates when you switch diagnostics on and discards when you delete the data.
- **What is not recorded**: no images or videos, no file names, no location, no contacts, no advertising ID, no account or contact details, no free text from error messages.
- **Where the data lives**: only in a file in the app's private storage area on your device. The app does **not** transmit it automatically – there is no server receiving it.
- **Export and deletion**: you can pass the file on yourself at any time through the Android share sheet (for example by e-mail to the developers) or delete it with one tap. If you switch the feature off, nothing more is written.

Independently of this, the app keeps a technical log in its own storage area; it contains no images and likewise leaves the device only if you export it yourself.

---

## 7. Contact and open source

AutoKorrektur is an open-source project (GNU AGPLv3) in the service of the mobility transition and liveable cities. The controller responsible for data processing is Konrad Völkel, Düsseldorf; the full postal address is in the [Impressum](https://autokorrektur.org/impressum) (German). For privacy questions, reach us by e-mail at autokorrektur [at] konradvoelkel [dot] com or through the project's [GitHub repository](https://github.com/xamde/AutoKorrektur). You have the rights under Articles 15–21 GDPR and the right to lodge a complaint with a supervisory authority.
