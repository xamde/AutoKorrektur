package de.konradvoelkel.android.autokorrektur.ml

import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized

@RunWith(Parameterized::class)
class ImageProcessingUtilsPaddingTest(
    private val w: Int,
    private val h: Int,
    private val expMax: Int,
    private val expXPad: Int,
    private val expYPad: Int,
    private val expXRatio: Float,
    private val expYRatio: Float
) {
    companion object {
        @JvmStatic
        @Parameterized.Parameters(name = "{0}x{1} -> max={2}, pads=({3},{4}), ratios=({5},{6})")
        fun data(): Collection<Array<Any>> = listOf(
            // Square case
            arrayOf(640, 640, 640, 0, 0, 1.0f, 1.0f),
            // Landscape (wider than tall)
            arrayOf(1280, 720, 1280, 0, 560, 1.0f, 1280f / 720f),
            // Portrait (taller than wide)
            arrayOf(1080, 1920, 1920, 840, 0, 1920f / 1080f, 1.0f),
            // Tiny landscape
            arrayOf(100, 50, 100, 0, 50, 1.0f, 2.0f),
            // Tiny portrait
            arrayOf(40, 160, 160, 120, 0, 4.0f, 1.0f),
        )
    }

    @Test
    fun testComputeSquarePaddingAndRatios() {
        val pr = ImageProcessingUtils.computeSquarePaddingAndRatios(w, h)
        assertEquals("maxSize", expMax, pr.maxSize)
        assertEquals("xPad", expXPad, pr.xPad)
        assertEquals("yPad", expYPad, pr.yPad)
        assertEquals("xRatio", expXRatio, pr.xRatio, 0.001f)
        assertEquals("yRatio", expYRatio, pr.yRatio, 0.001f)
    }
}
