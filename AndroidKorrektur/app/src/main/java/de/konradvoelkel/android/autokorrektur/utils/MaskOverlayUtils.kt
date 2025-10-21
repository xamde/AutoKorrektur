package de.konradvoelkel.android.autokorrektur.utils

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale

/**
 * Utilities to convert a grayscale mask bitmap (0=car/foreground, 255=background)
 * into a red translucent overlay, where only the masked area is tinted.
 */
object MaskOverlayUtils {
    /**
     * Creates a red overlay bitmap matching the given output size. The overlay will be
     * transparent everywhere except where the mask intensity is below [threshold], where
     * it will be drawn as red with [alpha] transparency (0..255).
     *
     * @param maskBitmap Grayscale mask bitmap (any config). Black (near 0) denotes masked area.
     * @param outWidth Target overlay width
     * @param outHeight Target overlay height
     * @param threshold Intensity threshold to consider as masked (default 128)
     * @param alpha Alpha for masked pixels (default 128)
     */
    fun createRedOverlayBitmap(
        maskBitmap: Bitmap,
        outWidth: Int,
        outHeight: Int,
        threshold: Int = 128,
        alpha: Int = 128
    ): Bitmap {
        // Scale mask to target size first
        val scaledMask = if (maskBitmap.width != outWidth || maskBitmap.height != outHeight) {
            maskBitmap.scale(outWidth, outHeight)
        } else maskBitmap

        // Prepare output ARGB bitmap
        val overlay = createBitmap(outWidth, outHeight)

        // Extract pixels from scaled mask
        val count = outWidth * outHeight
        val maskPixels = IntArray(count)
        scaledMask.getPixels(maskPixels, 0, outWidth, 0, 0, outWidth, outHeight)

        val outPixels = IntArray(count)
        val redColor = Color.red(Color.RED) // 255
        val greenColor = Color.green(Color.RED) // 0
        val blueColor = Color.blue(Color.RED) // 0

        for (i in 0 until count) {
            // Grayscale mask -> any channel is fine
            val p = maskPixels[i]
            val intensity = p and 0xFF // blue channel, but grayscale means R=G=B
            if (intensity < threshold) {
                // Masked area -> semi-transparent red
                outPixels[i] = (alpha.coerceIn(0, 255) shl 24) or
                        (redColor shl 16) or (greenColor shl 8) or blueColor
            } else {
                // Fully transparent elsewhere
                outPixels[i] = 0x00000000
            }
        }

        overlay.setPixels(outPixels, 0, outWidth, 0, 0, outWidth, outHeight)

        // If we created a temporary scaled mask, recycle it to free memory
        if (scaledMask !== maskBitmap) {
            scaledMask.recycle()
        }

        return overlay
    }

    /**
     * Draws the overlay generated from [maskBitmap] onto [baseBitmap] in-place.
     */
    fun drawOverlayOnto(
        baseBitmap: Bitmap,
        maskBitmap: Bitmap,
        threshold: Int = 128,
        alpha: Int = 128
    ) {
        val overlay = createRedOverlayBitmap(
            maskBitmap,
            baseBitmap.width,
            baseBitmap.height,
            threshold,
            alpha
        )
        val canvas = Canvas(baseBitmap)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        canvas.drawBitmap(overlay, 0f, 0f, paint)
        overlay.recycle()
    }
}
