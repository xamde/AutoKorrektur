package de.konradvoelkel.android.autokorrektur.telemetry

import java.io.File
import java.io.IOException

/**
 * Append-only JSON Lines file holding [TelemetryEvent]s, capped in size.
 *
 * Pure JVM (no Android types) so it is covered by plain unit tests. All writes go through
 * [append], which is synchronized: the pipelines record from `Dispatchers.Default`, the AR
 * pipeline from its own scope, and the UI thread reads counts for the Diagnostics dialog.
 *
 * When the file grows past [maxBytes] the oldest half of the lines is dropped, so the store is
 * bounded no matter how long diagnostics stay switched on (a `pipeline_run` line is ~300 bytes;
 * the default cap keeps roughly the last 6000 runs).
 */
class TelemetryStore(
    private val directory: File,
    private val maxBytes: Long = DEFAULT_MAX_BYTES,
) {
    private val file = File(directory, FILE_NAME)
    private val lock = Any()

    /** The JSONL file itself (may not exist yet); exposed for the export copy. */
    val eventsFile: File get() = file

    /** Appends one event; a failed write is logged by the caller, never thrown into the pipeline. */
    @Throws(IOException::class)
    fun append(event: TelemetryEvent) {
        synchronized(lock) {
            if (!directory.isDirectory && !directory.mkdirs()) {
                throw IOException("Cannot create telemetry directory $directory")
            }
            file.appendText(event.toJsonLine() + "\n")
            if (file.length() > maxBytes) trimOldestHalf()
        }
    }

    /** Number of recorded lines (0 when nothing has been written). */
    fun eventCount(): Int = synchronized(lock) {
        if (!file.exists()) 0 else file.useLines { lines -> lines.count { it.isNotBlank() } }
    }

    fun sizeBytes(): Long = synchronized(lock) { if (file.exists()) file.length() else 0L }

    /** All lines, oldest first — for tests and for building the export. */
    fun readLines(): List<String> = synchronized(lock) {
        if (!file.exists()) emptyList() else file.readLines().filter { it.isNotBlank() }
    }

    /** Deletes every recorded event. */
    fun clear() {
        synchronized(lock) { if (file.exists()) file.delete() }
    }

    private fun trimOldestHalf() {
        val lines = file.readLines().filter { it.isNotBlank() }
        val keep = lines.drop(lines.size / 2)
        file.writeText(if (keep.isEmpty()) "" else keep.joinToString("\n", postfix = "\n"))
    }

    companion object {
        const val FILE_NAME = "events.jsonl"
        const val DEFAULT_MAX_BYTES = 2L * 1024 * 1024
    }
}
