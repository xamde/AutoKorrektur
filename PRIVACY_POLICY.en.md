# Privacy Policy – AutoKorrektur

**Last updated:** September 2026

_English translation of [PRIVACY_POLICY.md](PRIVACY_POLICY.md) for convenience; the German text is the binding one. Describes the published app (product flavor `core`); features of unpublished flavors are deliberately not covered. Keep both files in sync — `site/build.sh` renders each into the website (`/privacy`, `/privacy-en`)._

The developers of **AutoKorrektur** take the protection of your personal data and your privacy seriously. This policy explains which data the AutoKorrektur Android app processes – and which it does not.

---

## 1. Principle: everything stays on your device

AutoKorrektur follows **privacy by design and by default**:
- Using the app requires **no registration**, **no user account** and **no entry of personal data**.
- The app has **no internet permission**. It cannot send or receive anything – neither your photos nor usage data.
- We use **no advertising SDKs**, **no tracking tools** and **no analytics services**.

---

## 2. App permissions

1. **Camera (`android.permission.CAMERA`)**:
   - **Purpose**: only when you tap "Take photo", your device's camera app is opened to take a picture for editing. AutoKorrektur shows no camera preview of its own and records nothing in the background.
   - You can deny the permission and pick photos from the gallery instead.

2. **Photos / media library**:
   - **Picking**: through the Android photo picker the app only gets access to the exact photo you select, not to your whole gallery.
   - **Saving**: on your request, finished results are saved as JPEG into your device's "Pictures" folder; the in-app gallery shows the images the app itself saved there.

---

## 3. Processing of your photos

- Vehicle detection (YOLOv11) and AI inpainting (MI-GAN) run **entirely on your device's processor**. The models ship inside the app.
- Your photos never leave your device. There is no server they could be sent to (see §1: no internet permission).
- Intermediate results live only in memory; the only thing written to storage is what you save or share yourself.

---

## 4. Disclosure to third parties

Your data is neither sold nor passed on to advertising networks or any other third party. When you share a finished before/after image via Instagram or another app, this happens solely through Android's share sheet (`Intent.ACTION_SEND`): you choose the target app, and only that app receives the image.

---

## 5. Optional diagnostics (on the device only, opt-in)

To make the app fast and stable on as many devices as possible, you can switch on the recording of technical measurements under **"Diagnostics"** in the menu. It is **off by default**.

- **What is recorded**: compute time of each processing step, image size in pixels, the number of detected vehicles, success or the kind of error, and once per session the app version, Android version, device manufacturer and model, memory size and processor cores. Plus a random installation ID that the app generates when you switch diagnostics on and discards when you delete the data.
- **What is not recorded**: no images, no file names, no location, no contacts, no advertising ID, no account or contact details, no free text from error messages.
- **Where the data lives**: only in a file in the app's private storage area on your device. The app does **not** transmit it – without an internet permission it could not even if it wanted to.
- **Export and deletion**: you can pass the file on yourself at any time through the Android share sheet (for example by e-mail to the developers) or delete it with one tap. If you switch the feature off, nothing more is written.

Independently of this, the app keeps a technical log in its own storage area; it contains no images and likewise leaves the device only if you export it yourself.

---

## 6. Contact and open source

AutoKorrektur is an open-source project (GNU AGPLv3) in the service of the mobility transition and liveable cities. The controller responsible for data processing is Konrad Völkel, Düsseldorf; the full postal address is in the [Impressum](https://autokorrektur.org/impressum) (German). For privacy questions, reach us by e-mail at autokorrektur [at] konradvoelkel [dot] com or through the project's [GitHub repository](https://github.com/xamde/AutoKorrektur). You have the rights under Articles 15–21 GDPR and the right to lodge a complaint with a supervisory authority.
