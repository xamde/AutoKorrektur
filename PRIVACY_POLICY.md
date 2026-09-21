# Datenschutzerklärung (Privacy Policy) – AutoKorrektur

**Stand:** September 2026

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

## 4. Weitergabe von Daten an Dritte

Deine Daten werden weder verkauft, noch an Werbenetzwerke oder unbefugte Dritte weitergegeben. Wenn du fertige Bilder oder Videos über Instagram oder andere Apps teilst, erfolgt dies ausschließlich über das standardmäßige Android-Freigabesystem (`Intent.ACTION_SEND`), über das du jederzeit die volle Kontrolle behältst.

---

## 5. Optionale Diagnosedaten (nur auf dem Gerät, Opt-In)

Um die App auf möglichst vielen Geräten schnell und stabil zu machen, kannst du unter **„Diagnosedaten“** im Menü das Mitschreiben technischer Messwerte einschalten. Standardmäßig ist das **ausgeschaltet**.

- **Was erfasst wird**: Rechenzeiten der einzelnen Verarbeitungsschritte, Bildgröße in Pixeln, gewählter Modus, Anzahl erkannter Fahrzeuge, Erfolg oder Fehlerart, Bildrate im AR-Modus, sowie einmal pro Sitzung App-Version, Android-Version, Gerätehersteller und -modell, Arbeitsspeicher und Prozessorkerne. Dazu eine zufällige Installations-ID, die die App beim Einschalten erzeugt und beim Löschen der Daten verwirft.
- **Was nicht erfasst wird**: keine Bilder oder Videos, keine Dateinamen, kein Standort, keine Kontakte, keine Werbe-ID, keine Konto- oder Kontaktdaten, keine Freitexte aus Fehlermeldungen.
- **Wo die Daten liegen**: ausschließlich in einer Datei im privaten Speicherbereich der App auf deinem Gerät. Die App überträgt sie **nicht** automatisch – es gibt keinen Server, der sie empfängt.
- **Export und Löschen**: Du kannst die Datei jederzeit über das Android-Teilen-Menü selbst weitergeben (z. B. per E-Mail an die Entwickler) oder mit einem Tipp löschen. Deaktivierst du die Funktion, wird nichts mehr geschrieben.

Unabhängig davon führt die App ein technisches Protokoll (Log) im app-eigenen Speicher, das keine Bilder enthält und das Gerät ebenfalls nur verlässt, wenn du es selbst exportierst.

---

## 6. Kontakt & Open-Source

AutoKorrektur ist ein Open-Source-Projekt (GNU AGPLv3) im Dienste der Mobilitätswende und lebenswerter Städte. Verantwortlich für die Datenverarbeitung ist Konrad Völkel, Düsseldorf; die vollständige Anschrift steht im [Impressum](https://autokorrektur.org/impressum). Bei Fragen zum Datenschutz erreichst du uns per E-Mail unter autokorrektur [at] konradvoelkel [punkt] com oder über das [GitHub-Repository](https://github.com/xamde/AutoKorrektur) des Projekts. Du hast die Rechte aus Art. 15–21 DSGVO und das Recht auf Beschwerde bei einer Aufsichtsbehörde.
