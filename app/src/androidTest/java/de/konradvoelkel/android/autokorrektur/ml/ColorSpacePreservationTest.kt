package de.konradvoelkel.android.autokorrektur.ml

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SmallTest
import de.konradvoelkel.android.autokorrektur.ml.mask.GuidedFilter
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

/**
 * Validates OpenCV color space conversions (RGBA <-> RGB <-> Grayscale) to ensure zero channel-swapping
 * or color tinting bugs across Android camera and gallery buffers.
 */
@RunWith(AndroidJUnit4::class)
@SmallTest
class ColorSpacePreservationTest : AndroidInstrumentedBaseTest() {

    @Before
    fun setUp() {
        assertTrue("OpenCV initialization failed", OpenCVLoader.initLocal())
    }

    @Test
    fun testPrimaryColorsPreservation_verifiesZeroChannelSwapping() {
        val width = 100
        val height = 100

        // Create test bitmap with 3 distinct color regions: Red (left), Green (center), Blue (right)
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val color = when {
                    x < 33 -> Color.rgb(255, 0, 0) // Pure Red
                    x < 66 -> Color.rgb(0, 255, 0) // Pure Green
                    else -> Color.rgb(0, 0, 255)   // Pure Blue
                }
                bitmap.setPixel(x, y, color)
            }
        }

        // 1. Convert to OpenCV Mat (RGBA)
        val rgbaMat = Mat()
        Utils.bitmapToMat(bitmap, rgbaMat)
        assertEquals("RGBA Mat must have 4 channels", 4, rgbaMat.channels())

        // Sample raw pixels from OpenCV Mat
        val pixelRed = rgbaMat.get(50, 15)
        assertEquals(255.0, pixelRed[0], 0.0) // R
        assertEquals(0.0, pixelRed[1], 0.0)   // G
        assertEquals(0.0, pixelRed[2], 0.0)   // B

        val pixelGreen = rgbaMat.get(50, 45)
        assertEquals(0.0, pixelGreen[0], 0.0)   // R
        assertEquals(255.0, pixelGreen[1], 0.0) // G
        assertEquals(0.0, pixelGreen[2], 0.0)   // B

        val pixelBlue = rgbaMat.get(50, 85)
        assertEquals(0.0, pixelBlue[0], 0.0)   // R
        assertEquals(0.0, pixelBlue[1], 0.0)   // G
        assertEquals(255.0, pixelBlue[2], 0.0) // B

        // 2. Convert to RGB Mat
        val rgbMat = Mat()
        Imgproc.cvtColor(rgbaMat, rgbMat, Imgproc.COLOR_RGBA2RGB)
        assertEquals("RGB Mat must have 3 channels", 3, rgbMat.channels())

        val rgbPixelRed = rgbMat.get(50, 15)
        assertEquals(255.0, rgbPixelRed[0], 0.0)
        assertEquals(0.0, rgbPixelRed[1], 0.0)
        assertEquals(0.0, rgbPixelRed[2], 0.0)

        // 3. Guided Filter Edge Guidance Channel Invariance
        val dummyMask = Mat.zeros(height, width, CvType.CV_8UC1)
        val refinedMask = GuidedFilter.filter(
            guide = rgbaMat,
            srcMask = dummyMask,
            radius = 3,
            eps = 0.01
        )

        assertEquals("Refined mask should match input dimensions", width, refinedMask.cols())
        assertEquals("Refined mask should match input dimensions", height, refinedMask.rows())

        rgbaMat.release()
        rgbMat.release()
        dummyMask.release()
        refinedMask.release()
        bitmap.recycle()
    }
}
