package de.konradvoelkel.android.autokorrektur.ml

/**
 * Pure image-processing utilities that can be JVM-tested.
 */
object ImageProcessingUtils {
    /**
     * Returns dimensions adjusted to be divisible by [stride].
     * The value rounds up when the remainder is >= stride/2, otherwise rounds down.
     */
    fun divStride(stride: Int, width: Int, height: Int): Pair<Int, Int> {
        require(stride > 0) { "stride must be > 0" }
        return Pair(roundToStride(width, stride), roundToStride(height, stride))
    }

    private fun roundToStride(dim: Int, stride: Int): Int {
        val remainder = dim % stride
        return if (remainder == 0) {
            dim
        } else if (remainder >= stride / 2) {
            (dim / stride + 1) * stride
        } else {
            (dim / stride) * stride
        }
    }

    /**
     * Result values used when padding an image to a square and computing scaling ratios.
     */
    data class PaddingRatios(
        val xPad: Int,
        val yPad: Int,
        val xRatio: Float,
        val yRatio: Float,
        val maxSize: Int
    )

    /**
     * Computes the square padding amounts and scaling ratios for an image of size [w]x[h].
     */
    fun computeSquarePaddingAndRatios(w: Int, h: Int): PaddingRatios {
        val safeW = w.coerceAtLeast(1)
        val safeH = h.coerceAtLeast(1)
        val maxSize = kotlin.math.max(safeW, safeH)
        val xPad = maxSize - safeW
        val yPad = maxSize - safeH
        val xRatio = maxSize.toFloat() / safeW.toFloat()
        val yRatio = maxSize.toFloat() / safeH.toFloat()
        return PaddingRatios(xPad, yPad, xRatio, yRatio, maxSize)
    }
}
