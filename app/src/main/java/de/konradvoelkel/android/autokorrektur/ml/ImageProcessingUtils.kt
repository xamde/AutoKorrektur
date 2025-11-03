package de.konradvoelkel.android.autokorrektur.ml

/**
 * Pure image-processing utilities that can be JVM-tested.
 */
internal object ImageProcessingUtils {
    /**
     * Returns dimensions adjusted to be divisible by [stride].
     * The value rounds up when the remainder is >= stride/2, otherwise rounds down.
     */
    fun divStride(stride: Int, width: Int, height: Int): Pair<Int, Int> {
        var widthDivisibleByStride = width
        var heightDivisibleByStride = height

        if (widthDivisibleByStride % stride != 0) {
            widthDivisibleByStride = if (widthDivisibleByStride % stride >= stride / 2) {
                (widthDivisibleByStride / stride + 1) * stride
            } else {
                (widthDivisibleByStride / stride) * stride
            }
        }

        if (heightDivisibleByStride % stride != 0) {
            heightDivisibleByStride = if (heightDivisibleByStride % stride >= stride / 2) {
                (heightDivisibleByStride / stride + 1) * stride
            } else {
                (heightDivisibleByStride / stride) * stride
            }
        }

        return Pair(widthDivisibleByStride, heightDivisibleByStride)
    }
}