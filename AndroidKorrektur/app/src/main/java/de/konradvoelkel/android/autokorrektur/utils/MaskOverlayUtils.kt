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

        // Process row-by-row to avoid large temporary arrays that can cause OOM on big images
        val redColor = Color.red(Color.RED) // 255
        val greenColor = Color.green(Color.RED) // 0
        val blueColor = Color.blue(Color.RED) // 0
        val clampedAlpha = alpha.coerceIn(0, 255)

        val maskRow = IntArray(outWidth)
        val outRow = IntArray(outWidth)
        for (y in 0 until outHeight) {
            // Read one row from the mask
            scaledMask.getPixels(maskRow, 0, outWidth, 0, y, outWidth, 1)
            // Convert to overlay row
            for (x in 0 until outWidth) {
                val p = maskRow[x]
                val intensity = p and 0xFF // grayscale -> any channel
                outRow[x] = if (intensity < threshold) {
                    (clampedAlpha shl 24) or (redColor shl 16) or (greenColor shl 8) or blueColor
                } else 0x00000000
            }
            // Write the row into the overlay
            overlay.setPixels(outRow, 0, outWidth, 0, y, outWidth, 1)
        }

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
