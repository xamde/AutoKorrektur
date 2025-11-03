package de.konradvoelkel.android.autokorrektur.shared

/**
 * JVM-only testing utilities. No Android dependencies allowed here.
 */
object JvmTestUtils {
    /** Returns a deterministic sequence of doubles between 0.0 (inclusive) and 1.0 (exclusive). */
    fun deterministicDoubles(seed: Long = 0L): Iterator<Double> {
        var state = seed xor 0x9E3779B97F4A7C15uL.toLong()
        return iterator {
            while (true) {
                // Xorshift64*
                var x = state.toULong()
                x = x xor (x shr 12)
                x = x xor (x shl 25)
                x = x xor (x shr 27)
                state = (x * 0x2545F4914F6CDD1DuL).toLong()
                // Map to [0,1)
                val v = (state ushr 11) and ((1L shl 53) - 1)
                yield(v.toDouble() / (1L shl 53).toDouble())
            }
        }
    }

    /** Simple helper to build an IntArray from a lambda for size [n]. */
    fun buildIntArray(n: Int, init: (Int) -> Int): IntArray = IntArray(n, init)

    /** Approximately equal for floats with absolute epsilon. */
    fun approxEquals(a: Float, b: Float, eps: Float = 1e-3f): Boolean =
        kotlin.math.abs(a - b) <= eps
}