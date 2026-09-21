package de.konradvoelkel.android.autokorrektur.telemetry

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * Pure-JVM coverage of the diagnostics store and its hand-rolled JSON Lines encoding
 * ([TelemetryEvent.toJsonLine]); the Android facade [Telemetry] is a thin wrapper around these.
 */
class TelemetryStoreTest {

    @get:Rule
    val tmp = TemporaryFolder()

    private fun event(name: String, vararg fields: Pair<String, Any?>) =
        TelemetryEvent(name, 1_700_000_000_000L, mapOf(*fields))

    @Test
    fun `toJsonLine writes one flat JSON object with event and timestamp first`() {
        val line = event(
            "pipeline_run",
            "mode" to "FAST_PREVIEW", "total_ms" to 812L, "detections" to 3,
            "success" to true, "error" to null, "avg_fps" to 12.5
        ).toJsonLine()
        assertEquals(
            """{"event":"pipeline_run","t":1700000000000,"mode":"FAST_PREVIEW","total_ms":812,""" +
                """"detections":3,"success":true,"error":null,"avg_fps":12.5}""",
            line
        )
        assertFalse("JSON Lines: one line per event", line.contains('\n'))
    }

    @Test
    fun `toJsonLine escapes strings and drops non-finite numbers`() {
        val line = event(
            "x",
            "s" to "a\"b\\c\nd",
            "nan" to Double.NaN,
            "inf" to Float.POSITIVE_INFINITY
        ).toJsonLine()
        assertTrue(line, line.contains("\"s\":\"a\\\"b\\\\c\\nd\""))
        assertTrue(line, line.contains("\"nan\":null"))
        assertTrue(line, line.contains("\"inf\":null"))
    }

    @Test
    fun `append creates the directory and file lazily and counts lines`() {
        val dir = File(tmp.root, "telemetry")
        val store = TelemetryStore(dir)
        assertEquals(0, store.eventCount())
        assertEquals(0L, store.sizeBytes())
        assertFalse(store.eventsFile.exists())

        store.append(event("a"))
        store.append(event("b", "k" to 1))

        assertTrue(dir.isDirectory)
        assertEquals(2, store.eventCount())
        assertEquals(
            listOf("""{"event":"a","t":1700000000000}""", """{"event":"b","t":1700000000000,"k":1}"""),
            store.readLines()
        )
        assertEquals(store.eventsFile.length(), store.sizeBytes())
    }

    @Test
    fun `clear removes everything and the store keeps working afterwards`() {
        val store = TelemetryStore(tmp.root)
        store.append(event("a"))
        store.clear()
        assertEquals(0, store.eventCount())
        assertFalse(store.eventsFile.exists())
        store.append(event("b"))
        assertEquals(listOf("""{"event":"b","t":1700000000000}"""), store.readLines())
    }

    @Test
    fun `exceeding maxBytes drops the oldest half and keeps the newest lines intact`() {
        val store = TelemetryStore(tmp.root, maxBytes = 400)
        val lineBytes = event("e", "i" to 0).toJsonLine().length + 1
        val n = 400 / lineBytes + 5 // enough to cross the cap once
        repeat(n) { store.append(event("e", "i" to it)) }

        val lines = store.readLines()
        assertTrue("store must stay bounded: ${store.sizeBytes()} bytes", store.sizeBytes() <= 400L + lineBytes)
        // Crossing the cap dropped the oldest half once; what was appended after that survived.
        assertTrue("expected fewer than $n lines after trimming, got ${lines.size}", lines.size < n)
        assertTrue("trimming must not empty the store", lines.size >= n / 2 - 1)
        // The survivors are the most recent ones, in order, and every line is still well-formed.
        val kept = lines.map { Regex("\"i\":(\\d+)").find(it)!!.groupValues[1].toInt() }
        assertEquals(kept.sorted(), kept)
        assertEquals(n - 1, kept.last())
        lines.forEach { assertTrue(it, it.startsWith("{") && it.endsWith("}")) }
    }
}
