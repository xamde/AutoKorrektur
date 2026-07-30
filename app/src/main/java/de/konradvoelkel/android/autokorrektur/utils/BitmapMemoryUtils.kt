package de.konradvoelkel.android.autokorrektur.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Utility functions for memory-safe Bitmap loading, downscaling, and recycling
 * to prevent OutOfMemory (OOM) exceptions on high-resolution camera photos.
 */
object BitmapMemoryUtils {

    const val DEFAULT_MAX_DISPLAY_DIMENSION = 1920

    /**
     * Scales down a Bitmap to fit within [maxDimension] while maintaining aspect ratio.
     * Returns the original bitmap if its dimensions are already within [maxDimension].
     */
    fun createScaledBitmapForDisplay(
        bitmap: Bitmap,
        maxDimension: Int = DEFAULT_MAX_DISPLAY_DIMENSION
    ): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val currentMax = max(width, height)

        if (currentMax <= maxDimension) {
            return bitmap
        }

        val scale = maxDimension.toFloat() / currentMax.toFloat()
        val newWidth = (width * scale).roundToInt()
        val newHeight = (height * scale).roundToInt()

        AppLogger.info("Downscaling bitmap for memory safety: ${width}x${height} -> ${newWidth}x${newHeight}")
        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }

    /**
     * Safely decodes a Bitmap from [uri] downsampled to [maxDimension] to prevent OOM errors.
     */
    fun decodeSampledBitmapFromUri(
        context: Context,
        uri: Uri,
        maxDimension: Int = DEFAULT_MAX_DISPLAY_DIMENSION
    ): Bitmap {
        return try {
            val source = ImageDecoder.createSource(context.contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                val width = info.size.width
                val height = info.size.height
                val largest = max(width, height)
                if (largest > maxDimension) {
                    val sampleSize = (largest.toFloat() / maxDimension.toFloat()).roundToInt()
                    if (sampleSize > 1) {
                        decoder.setTargetSampleSize(sampleSize)
                    }
                }
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = false
            }
        } catch (e: Exception) {
            AppLogger.warn("ImageDecoder failed, falling back to BitmapFactory options: ${e.message}")
            decodeBitmapWithOptions(context, uri, maxDimension)
        }
    }

    private fun decodeBitmapWithOptions(
        context: Context,
        uri: Uri,
        maxDimension: Int
    ): Bitmap {
        val boundsOptions = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri)?.use { stream ->
            BitmapFactory.decodeStream(stream, null, boundsOptions)
        }

        val width = boundsOptions.outWidth
        val height = boundsOptions.outHeight
        val largest = max(width, height)
        var sampleSize = 1
        if (largest > maxDimension) {
            sampleSize = (largest.toFloat() / maxDimension.toFloat()).roundToInt()
        }

        val decodeOptions = BitmapFactory.Options().apply {
            inSampleSize = max(1, sampleSize)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }

        return context.contentResolver.openInputStream(uri)?.use { stream ->
            BitmapFactory.decodeStream(stream, null, decodeOptions)
        } ?: throw java.io.IOException("Failed to open stream for URI: $uri")
    }
}
