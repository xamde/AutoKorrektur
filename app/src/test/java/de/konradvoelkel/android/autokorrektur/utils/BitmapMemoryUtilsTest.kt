package de.konradvoelkel.android.autokorrektur.utils

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.max

class BitmapMemoryUtilsTest {

    @Test
    fun defaultMaxDisplayDimension_is1920() {
        assertEquals(1920, BitmapMemoryUtils.DEFAULT_MAX_DISPLAY_DIMENSION)
    }

    @Test
    fun downscaleRatio_calculation_maintainsAspectRatio() {
        val width = 4000
        val height = 3000
        val maxDim = 1920

        val currentMax = max(width, height)
        val scale = maxDim.toFloat() / currentMax.toFloat()
        val newWidth = (width * scale).toInt()
        val newHeight = (height * scale).toInt()

        assertEquals(1920, newWidth)
        assertEquals(1440, newHeight)
    }
}
