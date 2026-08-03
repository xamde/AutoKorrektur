package de.konradvoelkel.android.autokorrektur

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SmallTest
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import de.konradvoelkel.android.autokorrektur.utils.MaskOverlayUtils
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream

@RunWith(AndroidJUnit4::class)
@SmallTest
class ColorFidelityAndMaskOverlayInstrumentedTest : AndroidInstrumentedBaseTest() {

    @Test
    fun testBitmapToMatColorConversion_preservesRGBOrder() {
        val processor = ImageProcessor(appContext)

        // Create a 640x640 pure Red Bitmap (R=255, G=0, B=0, A=255)
        val redBitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
        for (y in 0 until 640) {
            for (x in 0 until 640) {
                redBitmap.setPixel(x, y, Color.RED)
            }
        }

        val file = File(appContext.cacheDir, "test_red_image.png")
        FileOutputStream(file).use { out ->
            redBitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        val uri = android.net.Uri.fromFile(file)

        // Process image
        val processed = processor.processInputImage(uri, modelWidth = 640, modelHeight = 640)

        // Inspect originalMat pixel values at (0, 0)
        val pixelBytes = ByteArray(3)
        processed.originalMat.get(0, 0, pixelBytes)
        val redChannel = pixelBytes[0].toInt() and 0xFF
        val greenChannel = pixelBytes[1].toInt() and 0xFF
        val blueChannel = pixelBytes[2].toInt() and 0xFF

        // Channel 0 MUST be Red (255), Channel 1 MUST be Green (0), Channel 2 MUST be Blue (0)
        assertEquals("Channel 0 must be Red (255)", 255, redChannel)
        assertEquals("Channel 1 must be Green (0)", 0, greenChannel)
        assertEquals("Channel 2 must be Blue (0)", 0, blueChannel)

        processed.release()
        file.delete()
        redBitmap.recycle()
    }

    @Test
    fun testCreateRedOverlayBitmap_producesRedTransparentOverlayForMaskedArea() {
        val maskBitmap = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)
        for (y in 0 until 10) {
            for (x in 0 until 10) {
                val color = if (x < 5) Color.BLACK else Color.WHITE
                maskBitmap.setPixel(x, y, color)
            }
        }

        val overlay = MaskOverlayUtils.createRedOverlayBitmap(
            maskBitmap = maskBitmap,
            outWidth = 10,
            outHeight = 10,
            threshold = 128,
            alpha = 128
        )

        val maskedPixel = overlay.getPixel(2, 5)
        val alpha = Color.alpha(maskedPixel)
        val red = Color.red(maskedPixel)
        val green = Color.green(maskedPixel)
        val blue = Color.blue(maskedPixel)

        assertTrue("Masked pixel alpha must be non-zero", alpha > 0)
        assertTrue("Masked pixel must be red", red > 200)
        assertEquals("Masked pixel green channel must be 0", 0, green)
        assertEquals("Masked pixel blue channel must be 0", 0, blue)

        val unmaskedPixel = overlay.getPixel(8, 5)
        assertEquals("Unmasked pixel must be fully transparent", 0, Color.alpha(unmaskedPixel))

        maskBitmap.recycle()
        overlay.recycle()
    }
}
