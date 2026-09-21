package de.konradvoelkel.android.autokorrektur.telemetry

/**
 * One diagnostics record: an event [name], a wall-clock [timestampMs] and a flat map of
 * primitive [fields] (String, Boolean, Int, Long, Float, Double; null is written as JSON null).
 *
 * Serialised by [toJsonLine] as a single JSON object per line (JSON Lines), without any JSON
 * library so the store is unit-testable on the plain JVM (`org.json` is an Android stub there).
 * Field values are deliberately restricted to primitives: nothing in here should ever be able to
 * carry an image, a URI or a file path — see PRIVACY_POLICY.md §5 for what may be recorded.
 */
data class TelemetryEvent(
    val name: String,
    val timestampMs: Long,
    val fields: Map<String, Any?> = emptyMap(),
) {
    fun toJsonLine(): String = buildString {
        append('{')
        appendJsonString("event"); append(':'); appendJsonString(name)
        append(",\"t\":").append(timestampMs)
        for ((key, value) in fields) {
            append(','); appendJsonString(key); append(':'); appendJsonValue(value)
        }
        append('}')
    }

    private fun StringBuilder.appendJsonValue(value: Any?) {
        when (value) {
            null -> append("null")
            is Boolean -> append(value)
            is Int, is Long, is Short, is Byte -> append(value)
            is Float -> appendFiniteNumber(value.toDouble())
            is Double -> appendFiniteNumber(value)
            else -> appendJsonString(value.toString())
        }
    }

    private fun StringBuilder.appendFiniteNumber(value: Double) {
        if (value.isNaN() || value.isInfinite()) append("null") else append(value)
    }

    private fun StringBuilder.appendJsonString(text: String) {
        append('"')
        for (ch in text) {
            when (ch) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> if (ch < ' ') append(String.format("\\u%04x", ch.code)) else append(ch)
            }
        }
        append('"')
    }
}
