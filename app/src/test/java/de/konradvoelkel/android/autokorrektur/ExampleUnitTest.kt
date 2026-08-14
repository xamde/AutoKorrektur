package de.konradvoelkel.android.autokorrektur

import de.konradvoelkel.android.autokorrektur.ml.ImageProcessingUtils
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Unit tests for pure mathematical functions in [ImageProcessingUtils].
 */
class ImageProcessingUtilsUnitTest {

    @Test
    fun divStride_adjustsDimensionsToMultipleOfStride() {
        val (w1, h1) = ImageProcessingUtils.divStride(32, 640, 480)
        assertEquals(640, w1)
        assertEquals(480, h1)

        val (w2, h2) = ImageProcessingUtils.divStride(32, 645, 497)
        assertEquals(640, w2)
        assertEquals(512, h2)
    }

    @Test(expected = IllegalArgumentException::class)
    fun divStride_throwsOnNonPositiveStride() {
        ImageProcessingUtils.divStride(0, 100, 100)
    }

    @Test
    fun computeSquarePaddingAndRatios_calculatesCorrectRatios() {
        val ratios = ImageProcessingUtils.computeSquarePaddingAndRatios(1920, 1080)
        assertEquals(0, ratios.xPad)
        assertEquals(840, ratios.yPad)
        assertEquals(1.0f, ratios.xRatio, 0.001f)
        assertEquals(1920f / 1080f, ratios.yRatio, 0.001f)
        assertEquals(1920, ratios.maxSize)
    }
}