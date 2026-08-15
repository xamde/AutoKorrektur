# Datenschutzerklärung (Privacy Policy) – AutoKorrektur

**Stand:** August 2026

Die Entwickler von **AutoKorrektur** nehmen den Schutz deiner persönlichen Daten und deiner Privatsphäre sehr ernst. Diese Datenschutzerklärung informiert dich über die Art, den Umfang und den Zweck der Verarbeitung personenbezogener Daten innerhalb unserer Android-Anwendung.

---

## 1. Grundprinzip: On-Device & Datensparsamkeit

AutoKorrektur folgt dem Grundsatz der **Privacy by Design & Default**:
- Die Nutzung der App erfordert **keine Registrierung**, **kein Benutzerkonto** und **keine Eingabe persönlicher Identitätsdaten**.
- Wir setzen **keine Werbe-SDKs**, **keine Tracking-Tools** und **keine Verhaltensanalyse-Dienste** ein.

---

## 2. Erforderliche App-Berechtigungen

Damit AutoKorrektur seine Kernfunktionen ausführen kann, benötigt die App folgende Geräteberechtigungen:

1. **Kamera (`android.permission.CAMERA`)**:
   - **Zweck**: Ausschließlich zur Erzeugung des Live-AR-Kamerabildes und zur Aufnahme von Fotos und 5-Sekunden-Videosequenzen.
   - **Verarbeitung**: Die Kamerabilder werden live im flüchtigen Speicher deines Endgeräts verarbeitet. Es findet keine unbemerkte Übertragung im Hintergrund statt.

2. **Speicher / Mediathek (`READ_MEDIA_IMAGES`, `READ_MEDIA_VIDEO`)**:
   - **Zweck**: Ermöglicht dir das Auswählen vorhandener Fotos aus deiner Galerie für das Studio-Inpainting sowie das Speichern der fertigen Vorher/Nachher-Ergebnisse in deinen Medienordner.

---

## 3. Datenverarbeitung bei lokaler KI-Nutzung (On-Device)

Wenn du den **AR-Modus**, den **Schnell-Modus** oder den **Progressiven High-Res Modus** nutzt:
- Die Objekterkennung (YOLO) und das KI-Inpainting (MI-GAN / Progressive Tile Engine) laufen **vollständig lokal auf dem Prozessor / NPU deines Geräts**.
- Zu keinem Zeitpunkt verlassen deine Bild- oder Videodaten dein Endgerät.

---

## 4. Optionale Cloud-Verarbeitung (Cloud SDXL – Frankfurt, Deutschland)

AutoKorrektur bietet die optionale Funktion, Bilder über einen spezialisierten Server in photorealistischer SDXL-Qualität zu veredeln.

- **Ausdrückliche Einwilligung (Opt-In)**: Diese Funktion ist standardmäßig deaktiviert und erfordert vor der ersten Nutzung deine ausdrückliche Zustimmung via DSGVO-Einwilligungsdialog.
- **Standort der Server**: Die Verarbeitung erfolgt ausschließlich auf Servern in **Frankfurt am Main, Deutschland** unter strikter Einhaltung der Datenschutz-Grundverordnung (DSGVO / GDPR).
- **Flüchtige Verarbeitung (Zero-Storage Policy)**: Das hochgeladene Bild wird ausschließlich im Arbeitsspeicher (RAM) der Inpainting-Engine verarbeitet. Nach Abschluss der Inferenz und Übertragung des Ergebnisses an dein Smartphone wird das Bild **unverzüglich und unwiederbringlich aus dem Speicher gelöscht**. Es findet keine Protokollierung oder Speicherung deiner Bilder auf Festplatten statt.

---

## 5. Weitergabe von Daten an Dritte

Deine Daten werden weder verkauft, noch an Werbenetzwerke oder unbefugte Dritte weitergegeben. Wenn du fertige Bilder oder Videos über Instagram oder andere Apps teilst, erfolgt dies ausschließlich über das standardmäßige Android-Freigabesystem (`Intent.ACTION_SEND`), über das du jederzeit die volle Kontrolle behältst.

---

## 6. Kontakt & Open-Source

AutoKorrektur ist ein Open-Source-Projekt im Dienste der Mobilitätswende und lebenswerter Städte. Bei Fragen zum Datenschutz erreichst du uns über das GitHub-Repository des Projekts.
