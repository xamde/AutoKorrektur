# Datenschutzerklärung (Privacy Policy) – AutoKorrektur

**Stand:** September 2026

_Beschreibt die veröffentlichte App (Produkt-Flavor `core`, siehe `docs/PRODUCT_TIERS.md`). Funktionen anderer, nicht veröffentlichter Flavors (Live-AR, Videoclips, High-Res-Kachel-Inpainting, Cloud-Inpainting, Stapelverarbeitung) sind hier absichtlich nicht beschrieben; wird davon etwas veröffentlicht, wird dieser Text zuerst erweitert. Rendering auf autokorrektur.org: `site/build.sh`._

Die Entwickler von **AutoKorrektur** nehmen den Schutz deiner persönlichen Daten und deiner Privatsphäre sehr ernst. Diese Datenschutzerklärung informiert dich darüber, welche Daten die Android-App AutoKorrektur verarbeitet – und welche nicht.

---

## 1. Grundprinzip: alles auf deinem Gerät

AutoKorrektur folgt dem Grundsatz der **Privacy by Design & Default**:
- Die Nutzung der App erfordert **keine Registrierung**, **kein Benutzerkonto** und **keine Eingabe persönlicher Daten**.
- Die App hat **keine Internet-Berechtigung**. Sie kann keine Daten senden oder empfangen – weder deine Fotos noch Nutzungsdaten.
- Wir setzen **keine Werbe-SDKs**, **keine Tracking-Tools** und **keine Analyse-Dienste** ein.

---

## 2. App-Berechtigungen

1. **Kamera (`android.permission.CAMERA`)**:
   - **Zweck**: Nur, wenn du auf „Foto aufnehmen“ tippst, wird die Kamera-App deines Geräts geöffnet, um ein Foto für die Bearbeitung zu machen. AutoKorrektur zeigt selbst kein Kamerabild an und nimmt nichts im Hintergrund auf.
   - Du kannst die Berechtigung verweigern und stattdessen Fotos aus der Galerie auswählen.

2. **Fotos / Mediathek**:
   - **Auswählen**: Über den Android-Fotoauswähler bekommt die App nur Zugriff auf genau das Foto, das du auswählst, nicht auf deine ganze Galerie.
   - **Speichern**: Fertige Ergebnisse legt die App auf deinen Wunsch als JPEG im Ordner „Pictures“ deines Geräts ab; die Galerie in der App zeigt diese von ihr selbst gespeicherten Bilder wieder an.

---

## 3. Verarbeitung deiner Fotos

- Die Fahrzeugerkennung (YOLOv11) und das KI-Inpainting (MI-GAN) laufen **vollständig auf dem Prozessor deines Geräts**. Die Modelle sind Teil der App.
- Deine Fotos verlassen dein Gerät zu keinem Zeitpunkt. Es gibt keinen Server, an den sie geschickt werden könnten (siehe §1: keine Internet-Berechtigung).
- Zwischenergebnisse liegen nur im Arbeitsspeicher; gespeichert wird ausschließlich, was du selbst speicherst oder teilst.

---

## 4. Weitergabe von Daten an Dritte

Deine Daten werden weder verkauft noch an Werbenetzwerke oder sonstige Dritte weitergegeben. Wenn du ein fertiges Vorher/Nachher-Bild über Instagram oder eine andere App teilst, geschieht das ausschließlich über das Android-Teilen-Menü (`Intent.ACTION_SEND`): Du wählst die Ziel-App, und nur sie erhält das Bild.

---

## 5. Optionale Diagnosedaten (nur auf dem Gerät, Opt-In)

Um die App auf möglichst vielen Geräten schnell und stabil zu machen, kannst du unter **„Diagnosedaten“** im Menü das Mitschreiben technischer Messwerte einschalten. Standardmäßig ist das **ausgeschaltet**.

- **Was erfasst wird**: Rechenzeiten der einzelnen Verarbeitungsschritte, Bildgröße in Pixeln, Anzahl erkannter Fahrzeuge, Erfolg oder Fehlerart, sowie einmal pro Sitzung App-Version, Android-Version, Gerätehersteller und -modell, Arbeitsspeicher und Prozessorkerne. Dazu eine zufällige Installations-ID, die die App beim Einschalten erzeugt und beim Löschen der Daten verwirft.
- **Was nicht erfasst wird**: keine Bilder, keine Dateinamen, kein Standort, keine Kontakte, keine Werbe-ID, keine Konto- oder Kontaktdaten, keine Freitexte aus Fehlermeldungen.
- **Wo die Daten liegen**: ausschließlich in einer Datei im privaten Speicherbereich der App auf deinem Gerät. Die App überträgt sie **nicht** – sie kann es mangels Internet-Berechtigung auch gar nicht.
- **Export und Löschen**: Du kannst die Datei jederzeit über das Android-Teilen-Menü selbst weitergeben (z. B. per E-Mail an die Entwickler) oder mit einem Tipp löschen. Deaktivierst du die Funktion, wird nichts mehr geschrieben.

Unabhängig davon führt die App ein technisches Protokoll (Log) im app-eigenen Speicher, das keine Bilder enthält und das Gerät ebenfalls nur verlässt, wenn du es selbst exportierst.

---

## 6. Kontakt & Open-Source

AutoKorrektur ist ein Open-Source-Projekt (GNU AGPLv3) im Dienste der Mobilitätswende und lebenswerter Städte. Verantwortlich für die Datenverarbeitung ist Konrad Völkel, Düsseldorf; die vollständige Anschrift steht im [Impressum](https://autokorrektur.org/impressum). Bei Fragen zum Datenschutz erreichst du uns per E-Mail unter kontakt [at] autokorrektur [punkt] org oder über das [GitHub-Repository](https://github.com/xamde/AutoKorrektur) des Projekts. Du hast die Rechte aus Art. 15–21 DSGVO und das Recht auf Beschwerde bei einer Aufsichtsbehörde.
