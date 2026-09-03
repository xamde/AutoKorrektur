package de.konradvoelkel.android.autokorrektur

import org.junit.Assert.assertTrue
import org.junit.Test
import org.w3c.dom.Element
import java.io.File
import javax.xml.parsers.DocumentBuilderFactory

/**
 * Guards against a real, already-present localization bug in this project's string resources.
 *
 * ## The bug
 * Android resolves a string that is missing from `values-<locale>/strings.xml` by falling back
 * to the unqualified `values/strings.xml` ("default"). That is normally harmless — but this
 * project's default file is not language-neutral: as of this writing, 51 of its 121 entries
 * are German copy (e.g. `ar_camera_title` = "AUTOFREIE VISION") rather than English, and 49 of
 * those aren't declared in `values-de/strings.xml` at all. The app currently "works" only
 * because `values-de` and `values-en` happen to cover complementary halves of the key space, so
 * German and English devices each end up fully covered by one override plus the default
 * fallback. Google Play ships to every locale by default though: a device set to French,
 * Spanish, Polish, etc. has no `values-fr`/`values-es`/`values-pl`, so it falls straight through
 * to the default file and renders a UI that mixes English and German on the same screen (e.g.
 * "Select Image" next to "ZURÜCKSETZEN").
 *
 * This is exactly the class of issue that is nearly impossible to catch by manually poking at
 * the app in one or two languages during field testing — it only shows up if you happen to
 * switch your test device to a third locale — which is why it survived to a release candidate
 * un-reported.
 *
 * ## What this test does
 * It does NOT block the build on the 51 pre-existing violations (that would turn CI red for a
 * pre-existing, already-shipped condition without anyone having decided how to fix it). Instead
 * it uses a ratchet: today's known offenders are allow-listed below, so the test passes now,
 * but it fails the moment anyone adds a *new* string to `values/strings.xml` that disagrees
 * with `values-en` without also being declared explicitly in `values-de` — i.e. it stops the bug
 * from growing, starting today.
 *
 * To actually fix the existing 51: either (a) restrict the Play Store listing to "Germany +
 * English-speaking markets only" so no third locale is ever served the default file, or (b)
 * make `values/strings.xml` purely English (matching `values-en`) and give `values-de` a
 * complete, non-partial override. Once fixed, shrink `KNOWN_DEFAULT_GERMAN_LEAKS` to emptySet()
 * so this test starts catching 100% of regressions instead of just new ones.
 */
class StringResourceLocalizationTest {

    /**
     * Keys where, as of 2026-08-15 (the last docs/localization pass), `values/strings.xml`
     * contains German text that disagrees with the deliberate English translation in
     * `values-en/strings.xml`. This is a ratchet, not an excuse: it must only ever shrink.
     */
    private val knownDefaultGermanLeaks = setOf(
        "about_dialog_content", "about_dialog_title", "ar_camera_subtitle", "ar_camera_title",
        "ar_error_no_frame", "ar_error_video_init", "ar_fps_active", "ar_msg_captured",
        "ar_recording", "brush_size", "brush_title", "brush_type_brush", "brush_type_eraser",
        "btn_delete", "btn_done", "btn_export_share", "btn_reset", "btn_save", "cd_back",
        "cd_recent_gallery", "cd_shot_thumbnail", "engine_privacy", "engine_quota",
        "export_aspect_ratio", "export_btn_exporting", "export_error_failed",
        "export_error_not_ready", "export_layout_carousel", "export_layout_split",
        "export_layout_type", "export_layout_video", "export_ratio_square", "export_subtitle",
        "first_frag_brush", "gallery_empty_text", "gallery_n_shots", "gallery_title",
        "gallery_zero_shots", "gdpr_checkbox", "gdpr_desc1", "gdpr_desc2", "video_btn_before",
        "video_btn_car_free", "video_error_inpainting", "video_error_none_to_save",
        "video_error_not_found", "video_error_save", "video_error_share", "video_msg_ready",
        "video_msg_saved", "video_title"
    )

    /**
     * Locates `app/src/main/res` regardless of whether the JVM test process's working directory
     * is the `app/` module (the Gradle default) or the repo root (some IDE run configurations).
     */
    private fun resDir(): File {
        val fromModuleDir = File("src/main/res")
        if (fromModuleDir.isDirectory) return fromModuleDir
        val fromRepoRoot = File("app/src/main/res")
        if (fromRepoRoot.isDirectory) return fromRepoRoot
        throw IllegalStateException(
            "Could not locate app/src/main/res from working directory " +
                "${File(".").absolutePath} — run this test via Gradle (./gradlew :app:testDebugUnitTest)."
        )
    }

    private fun parseStrings(file: File): Map<String, String> {
        assertTrue("Expected strings.xml at ${file.absolutePath}", file.exists())
        val doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(file)
        val nodes = doc.getElementsByTagName("string")
        val result = mutableMapOf<String, String>()
        for (i in 0 until nodes.length) {
            val el = nodes.item(i) as Element
            result[el.getAttribute("name")] = el.textContent
        }
        return result
    }

    @Test
    fun `default strings xml does not silently disagree with values-en for a new, un-allow-listed key`() {
        val res = resDir()
        val default = parseStrings(File(res, "values/strings.xml"))
        val en = parseStrings(File(res, "values-en/strings.xml"))

        val disagreeing = (default.keys intersect en.keys)
            .filter { key -> default.getValue(key) != en.getValue(key) }
            .toSet()

        val newRegressions = disagreeing - knownDefaultGermanLeaks
        val fixedEntries = knownDefaultGermanLeaks - disagreeing

        if (fixedEntries.isNotEmpty()) {
            println(
                "✅ ${fixedEntries.size} previously-known default/values-en mismatch(es) no longer " +
                    "reproduce — remove from knownDefaultGermanLeaks to tighten the ratchet: $fixedEntries"
            )
        }

        assertTrue(
            "Found ${newRegressions.size} NEW string(s) where values/strings.xml (the file every " +
                "locale other than de/en falls back to) disagrees with values-en/strings.xml. " +
                "Any user on a locale without its own values-<locale> override (fr, es, pl, ...) " +
                "will see this text in whatever language `default` happens to be in, which is a " +
                "silent language mix. Fix by adding an explicit values-de override for the key, or " +
                "by making the default entry itself correct English. Offending keys: $newRegressions",
            newRegressions.isEmpty()
        )
    }

    @Test
    fun `every string key defined anywhere exists in the default resource file`() {
        // The reverse failure mode: a key that ONLY exists in values-de or values-en and not in
        // default at all would crash at runtime for any third locale (resource not found), not
        // just look wrong. This has not happened yet, but it costs nothing to guard against it.
        val res = resDir()
        val default = parseStrings(File(res, "values/strings.xml"))
        val de = parseStrings(File(res, "values-de/strings.xml"))
        val en = parseStrings(File(res, "values-en/strings.xml"))

        val missingFromDefault = (de.keys + en.keys) - default.keys
        assertTrue(
            "These keys exist in a locale-specific override but have NO entry in the default " +
                "values/strings.xml, which would crash resource resolution for any locale that " +
                "isn't de or en: $missingFromDefault",
            missingFromDefault.isEmpty()
        )
    }
}
