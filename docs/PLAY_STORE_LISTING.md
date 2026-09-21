# Google Play Store Listing: AutoKorrektur

> Describes the **`core` flavor**, the only one that goes to Google Play (`docs/MVP_FEATURE_FLAG_PLAN.md`):
> photo in, cars removed on-device, before/after slider, split card shared through the Android
> share sheet, in-app gallery of past shots. Play's policy forbids advertising features the
> uploaded build doesn't have, so the AR / video / high-res / cloud / brush / batch copy that used
> to be here is not in the listing until those features are promoted into `core` (the old text is in
> git history, commit e1127ba and before). Privacy policy URL: https://autokorrektur.org/privacy.

## 1. App-Details & Metadaten

- **App-Name (Deutsch)**: `AutoKorrektur – Deine Straße ohne Autos`
- **App Name (English)**: `AutoKorrektur – Your Street Without Cars`
- **Paketname**: `de.konradvoelkel.android.autokorrektur`
- **Kategorie**: Fotografie
- **Altersfreigabe**: USK 0 / PEGI 3 / Everyone
- **Datenschutzerklärung**: https://autokorrektur.org/privacy (English: /privacy-en)
- **Website**: https://autokorrektur.org · **E-Mail**: autokorrektur@konradvoelkel.com

---

## 2. Kurzbeschreibungen (max. 80 Zeichen)

### Deutsch
> Entferne parkende Autos aus deinen Fotos – KI komplett auf dem Gerät, ohne Konto.

### English
> Remove parked cars from your photos – AI entirely on your device, no account.

---

## 3. Vollständige Beschreibung (Deutsch)

```markdown
Stell dir deine Straße vor – ohne Blechlawinen, ohne zugeparkte Gehwege. AutoKorrektur zeigt dir, wie das aussieht: Foto rein, Autos raus.

AutoKorrektur ist ein Werkzeug für Mobilitätsaktivist:innen, Stadtplaner:innen und alle, die zeigen wollen, wie viel Platz eine Straße ohne Autos hat. Mach ein Foto oder wähle eines aus der Galerie, und die App entfernt die parkenden Fahrzeuge – die komplette KI läuft auf deinem Smartphone.

🌟 Funktionen

📸 Foto rein, Autos raus
Eine Segmentierungs-KI (YOLOv11) findet die Fahrzeuge, ein Inpainting-Netz (MI-GAN) füllt die Lücke mit plausibler Straße, Grün und Gehweg. Alles auf dem Gerät, nach der Installation ohne Internet.

↔️ Vorher / Nachher
Ein Schieberegler zeigt den Unterschied. Die App merkt sich deine autofreien Aufnahmen in einer kleinen Galerie.

📤 Teilen in zwei Tipps
Die fertige Split-Karte („VORHER / AUTOFREI“) teilst du über das normale Android-Teilen-Menü – Instagram, Signal, E-Mail, was du willst – oder speicherst sie in deine Galerie.

🔒 Datenschutz
• Die App hat keine Internet-Berechtigung: Deine Fotos verlassen dein Gerät nicht.
• Kein Konto, keine Werbung, kein Tracking.
• Optionale Diagnosedaten (Rechenzeiten, Gerätemodell – nie Bilder) bleiben auf dem Gerät, bis du sie selbst exportierst. Standardmäßig aus.

🔬 Hintergrund
Das Ergebnis ist eine Skizze, keine Fotomontage in Druckqualität: Schatten, Spiegelungen und verdeckte Details erfindet das Netz plausibel, aber nicht immer richtig. Die App baut auf zwei Bachelorarbeiten an der Heinrich-Heine-Universität Düsseldorf auf (Till Schellscheidt 2024, Ben Beckers 2025). Freie Software (AGPLv3), Quellcode auf GitHub.

Die KI-Modelle stecken in der App, deshalb ist sie einige hundert Megabyte groß. Android 10 oder neuer; ein Gerät mit 4 GB RAM oder mehr ist empfehlenswert.
```

---

## 4. Full Description (English)

```markdown
Imagine your street without the wall of parked cars and the blocked sidewalks. AutoKorrektur shows you what that looks like: photo in, cars out.

AutoKorrektur is a tool for mobility activists, urban planners and anyone who wants to show how much room a street has once the cars are gone. Take a photo or pick one from your gallery and the app removes the parked vehicles – the entire AI runs on your phone.

🌟 Features

📸 Photo in, cars out
A segmentation network (YOLOv11) finds the vehicles; an inpainting network (MI-GAN) fills the gap with plausible road, greenery and pavement. All on-device, no internet needed after installing.

↔️ Before / after
A slider shows the difference. The app keeps your car-free shots in a small gallery.

📤 Share in two taps
The finished split card ("BEFORE / CAR-FREE") goes out through Android's ordinary share sheet – Instagram, Signal, e-mail, whatever you pick – or is saved to your gallery.

🔒 Privacy
• The app has no internet permission: your photos never leave your device.
• No account, no advertising, no tracking.
• Optional diagnostics (compute times, device model – never images) stay on the device until you export them yourself. Off by default.

🔬 Background
The result is a sketch, not a print-quality montage: shadows, reflections and hidden details are invented plausibly, not always correctly. The app builds on two bachelor theses at Heinrich Heine University Düsseldorf (Till Schellscheidt 2024, Ben Beckers 2025). Free software (AGPLv3), source code on GitHub.

The models ship inside the app, which is why it is a few hundred megabytes. Android 10 or newer; a device with 4 GB of RAM or more is recommended.
```

---

## 5. Grafiken

`media/play_store_assets/`: Feature-Grafik und Icon sind verwendbar. **Die drei Screenshots dort
zeigen Live-AR, den Engine-Wähler und den Multi-Layout-Export – Funktionen, die `core` nicht hat.**
Vor dem Upload neue Screenshots vom `core`-Build machen (Startbildschirm, Ergebnis mit Schieberegler,
Teilen-Menü mit der Split-Karte, Galerie); Vorgehen wie in `HUMAN_RELEASE_CHECKLIST.md` §5.
