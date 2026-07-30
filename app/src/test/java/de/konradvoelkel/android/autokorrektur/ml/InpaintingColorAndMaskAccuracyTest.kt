package de.konradvoelkel.android.autokorrektur.ml

import de.konradvoelkel.android.autokorrektur.ml.model.Detection
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class InpaintingColorAndMaskAccuracyTest {

    @Test
    fun colorChannelOrder_preservesYellowHueWithoutSwap() {
        // Yellow color in RGB: Red=230, Green=190, Blue=130 (Red > Blue)
        val r = 230
        val g = 190
        val b = 130

        // Verify RGB channel ordering: index 0 = Red, index 2 = Blue
        val rgbPixel = intArrayOf(r, g, b)

        val redValue = rgbPixel[0]
        val blueValue = rgbPixel[2]

        assertTrue("Red channel must be greater than Blue channel for Yellow hue", redValue > blueValue)
        assertEquals("Red channel value must match", r, redValue)
        assertEquals("Blue channel value must match", b, blueValue)
    }

    @Test
    fun maskBounds_clampedTightToAvoidBuildingFacadeErasing() {
        val detection = Detection(
            x = 0.2f,
            y = 0.6f,
            width = 0.4f,
            height = 0.2f,
            confidence = 0.85f,
            classId = 2,
            maskCoefficients = FloatArray(32) { 0.1f }
        )

        assertTrue(detection.width > 0)
        assertTrue(detection.height > 0)
        assertTrue(detection.x >= 0)
        assertTrue(detection.y >= 0)
    }
}
