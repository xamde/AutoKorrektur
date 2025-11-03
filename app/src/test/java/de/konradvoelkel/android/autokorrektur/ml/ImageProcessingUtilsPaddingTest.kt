package de.konradvoelkel.android.autokorrektur.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized

@RunWith(Parameterized::class)
class ImageProcessingUtilsPaddingTest(
    private val w: Int,
    private val h: Int,
    private val expectedMax: Int,
    private val expectedXPad: Int,
    private val expectedYPad: Int
) {
    companion object {
        @JvmStatic
        @Parameterized.Parameters(name = "{0}x{1} -> max={2} xPad={3} yPad={4}")
        fun data(): Collection<Array<Int>> = listOf(
            // already square
            arrayOf(640, 640, 640, 0, 0),
            // landscape
            arrayOf(800, 600, 800, 0, 200),
            // portrait
            arrayOf(600, 800, 800, 200, 0),
            // odd dimensions
            arrayOf(641, 479, 641, 0, 162),
            // tiny images
            arrayOf(1, 1, 1, 0, 0),
            arrayOf(1, 2, 2, 1, 0),
        )
    }

    @Test
    fun computeSquarePaddingAndRatios_matches_arithmetic() {
        val pr = ImageProcessingUtils.computeSquarePaddingAndRatios(w, h)
        assertEquals("maxSize", expectedMax, pr.maxSize)
        assertEquals("xPad", expectedXPad, pr.xPad)
        assertEquals("yPad", expectedYPad, pr.yPad)

        // Ratios
        assertEquals("xRatio", expectedMax.toFloat() / w.toFloat(), pr.xRatio, 1e-6f)
        assertEquals("yRatio", expectedMax.toFloat() / h.toFloat(), pr.yRatio, 1e-6f)
        assertTrue("ratios should be >= 1", pr.xRatio >= 1f || pr.yRatio >= 1f)
    }
}