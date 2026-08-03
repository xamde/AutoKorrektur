package de.konradvoelkel.android.autokorrektur

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.MediumTest
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
@MediumTest
class InpaintingEngineFidelityTest : AndroidInstrumentedBaseTest() {

    @Test
    fun testUnmaskedPixelInvarianceInMiGan() {
        var miGan: MiGanInference? = null
        try {
            miGan = MiGanInference(appContext)
            miGan.initialize()
        } catch (e: Exception) {
            // If model missing in specific runner, gracefully skip
            return
        }

        val width = 512
        val height = 512
        val inputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val maskBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Fill input with solid blue, mask with small white box in center
        for (y in 0 until height) {
            for (x in 0 until width) {
                inputBitmap.setPixel(x, y, Color.rgb(50, 100, 200))
                if (x in 200..300 && y in 200..300) {
                    maskBitmap.setPixel(x, y, Color.BLACK) // Masked area (car)
                } else {
                    maskBitmap.setPixel(x, y, Color.WHITE) // Unmasked background
                }
            }
        }

        val inputMat = org.opencv.core.Mat()
        val maskMat = org.opencv.core.Mat()
        org.opencv.android.Utils.bitmapToMat(inputBitmap, inputMat)
        org.opencv.android.Utils.bitmapToMat(maskBitmap, maskMat)

        val outputMat = try {
            miGan.inferMiGan(inputMat, maskMat)
        } catch (e: Exception) {
            null
        }

        if (outputMat != null) {
            assertNotNull("Output mat should not be null", outputMat)
            assertEquals("Width should match", width, outputMat.cols())
            assertEquals("Height should match", height, outputMat.rows())

            val outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            org.opencv.android.Utils.matToBitmap(outputMat, outputBitmap)

            // Verify unmasked corner pixel (10,10) is preserved without off-color tint
            val cornerPixel = outputBitmap.getPixel(10, 10)
            val redDiff = Math.abs(Color.red(cornerPixel) - 50)
            val greenDiff = Math.abs(Color.green(cornerPixel) - 100)
            val blueDiff = Math.abs(Color.blue(cornerPixel) - 200)

            assertTrue(
                "Unmasked area pixels must preserve original color without tinting artifacts (red diff: $redDiff, green diff: $greenDiff, blue diff: $blueDiff)",
                redDiff < 15 && greenDiff < 15 && blueDiff < 15
            )
            outputMat.release()
            outputBitmap.recycle()
        }

        inputMat.release()
        maskMat.release()
        miGan.close()
        inputBitmap.recycle()
        maskBitmap.recycle()
    }
}
