package de.konradvoelkel.android.autokorrektur

import android.graphics.Bitmap
import de.konradvoelkel.android.autokorrektur.utils.MaskOverlayUtils
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import org.opencv.android.Utils
import org.opencv.imgcodecs.Imgcodecs

import org.opencv.imgproc.Imgproc

@RunWith(Parameterized::class)
class ImageOverlayTests(
    private val threshold: Int,
    private val alpha: Int
) : de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest() {

    companion object {
        @JvmStatic
        @Parameterized.Parameters(name = "threshold={0}, alpha={1}")
        fun params(): Collection<Array<Int>> = listOf(
            arrayOf(96, 96),
            arrayOf(96, 128),
            arrayOf(128, 96),
            arrayOf(128, 128),
            arrayOf(160, 96),
            arrayOf(160, 128),
        )
    }

    @Test
    fun overlay_should_not_tint_entire_image_and_should_tint_lower_left_region() {
        // Load reference mask from assets directly into Bitmap
        val maskFile = de.konradvoelkel.android.autokorrektur.shared.AndroidTestUtils
            .copyAssetToCache(appContext, "photo_with_car_1_mask.png")
        val maskBitmap = android.graphics.BitmapFactory.decodeFile(maskFile.absolutePath)
        requireNotNull(maskBitmap) { "Failed to load reference mask" }

        // Create a white base image and draw overlay
        val base = Bitmap.createBitmap(maskBitmap.width, maskBitmap.height, Bitmap.Config.ARGB_8888)
        base.eraseColor(0xFFFFFFFF.toInt())
        MaskOverlayUtils.drawOverlayOnto(base, maskBitmap, threshold = threshold, alpha = alpha)

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
        assertTrue(
            "Overlay should tint some pixels (thr=$threshold, alpha=$alpha)",
            tintedCount in 1 until total
        )

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
            "Lower-left should have more red overlay than upper-right (thr=$threshold, alpha=$alpha, ll=$llTint, ur=$urTint)",
            llTint > urTint * 2
        )

        // Cleanup
        base.recycle()
        maskBitmap.recycle()
    }
}
