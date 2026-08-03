package de.konradvoelkel.android.autokorrektur.pipeline

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.InputStream
import java.util.Locale

/**
 * Instrumented Android test verifying all 50 image-triples loaded from assets.
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class FiftyImageTriplesInstrumentedTest : AndroidInstrumentedBaseTest() {

    @Test
    fun testAllFiftyImageTriplesAssetIntegrityAndMaskParity() {
        val totalTriples = 50

        for (i in 1..totalTriples) {
            val prefix = String.format(Locale.US, "triple_%02d", i)

            val carStream: InputStream = appContext.assets.open("triples/${prefix}_with_car.png")
            val maskStream: InputStream = appContext.assets.open("triples/${prefix}_mask.png")
            val carlessStream: InputStream = appContext.assets.open("triples/${prefix}_without_car.png")

            val carBitmap: Bitmap? = BitmapFactory.decodeStream(carStream)
            val maskBitmap: Bitmap? = BitmapFactory.decodeStream(maskStream)
            val carlessBitmap: Bitmap? = BitmapFactory.decodeStream(carlessStream)

            assertNotNull("Triple $i with_car bitmap should decode", carBitmap)
            assertNotNull("Triple $i mask bitmap should decode", maskBitmap)
            assertNotNull("Triple $i without_car bitmap should decode", carlessBitmap)

            val width = carBitmap!!.width
            val height = carBitmap.height

            assertEquals("Triple $i mask width should match car width", width, maskBitmap!!.width)
            assertEquals("Triple $i mask height should match car height", height, maskBitmap.height)
            assertEquals("Triple $i carless width should match car width", width, carlessBitmap!!.width)
            assertEquals("Triple $i carless height should match car height", height, carlessBitmap.height)

            var maskPixelCount = 0
            val totalPixels = width * height

            val carPixels = IntArray(totalPixels)
            val maskPixels = IntArray(totalPixels)
            val carlessPixels = IntArray(totalPixels)

            carBitmap.getPixels(carPixels, 0, width, 0, 0, width, height)
            maskBitmap.getPixels(maskPixels, 0, width, 0, 0, width, height)
            carlessBitmap.getPixels(carlessPixels, 0, width, 0, 0, width, height)

            var nonCarMismatchCount = 0

            for (idx in 0 until totalPixels) {
                val maskVal = maskPixels[idx] and 0xFF
                if (maskVal > 128) {
                    maskPixelCount++
                } else {
                    val carPixel = carPixels[idx]
                    val carlessPixel = carlessPixels[idx]

                    val rDiff = Math.abs(((carPixel shr 16) and 0xFF) - ((carlessPixel shr 16) and 0xFF))
                    val gDiff = Math.abs(((carPixel shr 8) and 0xFF) - ((carlessPixel shr 8) and 0xFF))
                    val bDiff = Math.abs((carPixel and 0xFF) - (carlessPixel and 0xFF))

                    if (rDiff + gDiff + bDiff > 15) {
                        nonCarMismatchCount++
                    }
                }
            }

            val maskCoverageRatio = maskPixelCount.toDouble() / totalPixels.toDouble()
            assertTrue(
                "Triple $i mask coverage ratio ($maskCoverageRatio) should be > 1%",
                maskCoverageRatio > 0.01
            )
            assertTrue(
                "Triple $i mask coverage ratio ($maskCoverageRatio) should be < 80%",
                maskCoverageRatio < 0.80
            )

            val mismatchRatio = nonCarMismatchCount.toDouble() / totalPixels.toDouble()
            val msg = "Triple $i background outside mask should match carless image (mismatch=$mismatchRatio)"
            assertTrue(msg, mismatchRatio <= 0.02)

            carStream.close()
            maskStream.close()
            carlessStream.close()

            carBitmap.recycle()
            maskBitmap.recycle()
            carlessBitmap.recycle()
        }
    }
}
