package de.konradvoelkel.android.autokorrektur

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import de.konradvoelkel.android.autokorrektur.utils.MaskOverlayUtils
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.Utils
import org.opencv.imgcodecs.Imgcodecs

@RunWith(AndroidJUnit4::class)
class ImageOverlayTests : de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest() {


    @Test
    fun overlay_should_not_tint_entire_image_and_should_tint_lower_left_region() {
        // Load reference mask from assets
        val maskFile = de.konradvoelkel.android.autokorrektur.shared.AndroidTestUtils
            .copyAssetToCache(appContext, "photo_with_car_1_mask.png")
        val maskMat = Imgcodecs.imread(maskFile.absolutePath, Imgcodecs.IMREAD_GRAYSCALE)
        require(!maskMat.empty()) { "Failed to load reference mask" }

        // Convert to bitmap
        val maskBitmap =
            Bitmap.createBitmap(maskMat.cols(), maskMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(maskMat, maskBitmap)

        // Create a white base image and draw overlay
        val base = Bitmap.createBitmap(maskBitmap.width, maskBitmap.height, Bitmap.Config.ARGB_8888)
        base.eraseColor(0xFFFFFFFF.toInt())
        MaskOverlayUtils.drawOverlayOnto(base, maskBitmap, threshold = 128, alpha = 128)

        // Analyze result pixels
        val w = base.width
        val h = base.height
        val total = w * h
        val pixels = IntArray(total)
        base.getPixels(pixels, 0, w, 0, 0, w, h)

        var tintedCount = 0
        for (p in pixels) {
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            // detect reddish tint vs pure white background
            val isTinted = (r > g + 20) && (r > b + 20)
            if (isTinted) tintedCount++
        }

        // Expect some but not all pixels to be tinted
        assertTrue("Overlay should tint some pixels", tintedCount in 1 until total)

        // Expect more tint in lower-left quadrant than upper-right (car is lower-left)
        val halfW = w / 2
        val halfH = h / 2
        var llTint = 0
        var urTint = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val idx = y * w + x
                val p = pixels[idx]
                val r = (p shr 16) and 0xFF
                val g = (p shr 8) and 0xFF
                val b = p and 0xFF
                val isTinted = (r > g + 20) && (r > b + 20)
                if (x < halfW && y >= halfH && isTinted) llTint++
                if (x >= halfW && y < halfH && isTinted) urTint++
            }
        }
        // We expect lower-left to have noticeably more tinted pixels than upper-right
        assertTrue(
            "Lower-left should have more red overlay than upper-right (ll=$llTint, ur=$urTint)",
            llTint > urTint * 2
        )

        // Cleanup
        maskMat.release()
        base.recycle()
        maskBitmap.recycle()
    }
}
