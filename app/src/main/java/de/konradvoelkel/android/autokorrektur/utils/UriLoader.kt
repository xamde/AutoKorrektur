package de.konradvoelkel.android.autokorrektur.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import android.net.Uri
import android.os.Build
import java.io.File
import java.io.IOException
import kotlin.math.sqrt

/**
 * Utility to load Bitmaps from Android URIs with safe downsampling and EXIF rotation correction.
 */
class UriLoader(private val context: Context) {

    companion object {
        private const val MEGAPIXEL = 1_000_000f
    }

    /**
     * Loads a [Bitmap] from a given [Uri], applying safe downsampling to fit [maxMegapixels]
     * and correcting orientation based on EXIF metadata.
     */
    @Throws(IOException::class)
    fun loadRotatedBitmap(imageUri: Uri, maxMegapixels: Float): Bitmap {
        val loaded = when (val scheme = imageUri.scheme?.lowercase()) {
            "file" -> loadBitmapFromFile(imageUri, maxMegapixels)
            "content" -> loadBitmapFromContentProvider(imageUri, maxMegapixels)
            else -> throw IOException("Unsupported URI scheme: $scheme")
        }
        val rotated = rotateBitmapIfRequired(loaded, imageUri)
        return ensureStandardSoftwareArgb8888(rotated)
    }

    private fun ensureStandardSoftwareArgb8888(bitmap: Bitmap): Bitmap {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            bitmap.gainmap = null
        }
        val standard = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            standard.gainmap = null
        }
        val canvas = android.graphics.Canvas(standard)
        canvas.drawBitmap(bitmap, 0f, 0f, null)
        if (standard !== bitmap) {
            bitmap.recycle()
        }
        return standard
    }

    private fun rotateBitmapIfRequired(bitmap: Bitmap, uri: Uri): Bitmap {
        val inputStream = try {
            when (uri.scheme?.lowercase()) {
                "file" -> uri.path?.let { java.io.FileInputStream(it) }
                "content" -> context.contentResolver.openInputStream(uri)
                else -> null
            }
        } catch (e: Exception) {
            null
        } ?: return bitmap

        val exif = try {
            ExifInterface(inputStream)
        } catch (e: Exception) {
            return bitmap
        } finally {
            try {
                inputStream.close()
            } catch (_: Exception) {
            }
        }

        val orientation = exif.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        )

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            else -> return bitmap
        }

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            bitmap.gainmap = null
        }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            rotated.gainmap = null
        }
        if (rotated != bitmap) {
            bitmap.recycle()
        }
        return rotated
    }

    private fun loadBitmapFromFile(uri: Uri, maxMegapixels: Float): Bitmap {
        val path = uri.path ?: throw IOException("File path is null for URI: $uri")
        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, options)

        val imageWidth = options.outWidth
        val imageHeight = options.outHeight
        if (imageWidth <= 0 || imageHeight <= 0) {
            throw IOException("Invalid image dimensions from file: ${imageWidth}x${imageHeight}")
        }

        val decodeOptions = BitmapFactory.Options().apply {
            inSampleSize = calculateInSampleSize(imageWidth, imageHeight, maxMegapixels)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bmp = BitmapFactory.decodeFile(path, decodeOptions)
            ?: throw IOException("BitmapFactory.decodeFile failed for path: $path")
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            bmp.gainmap = null
        }
        return bmp
    }

    private fun loadBitmapFromContentProvider(uri: Uri, maxMegapixels: Float): Bitmap {
        try {
            val source = ImageDecoder.createSource(context.contentResolver, uri)
            val bmp = ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                val sampleSize =
                    calculateInSampleSize(info.size.width, info.size.height, maxMegapixels)
                if (sampleSize > 1) {
                    decoder.setTargetSampleSize(sampleSize)
                }
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = true
            }
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                bmp.gainmap = null
            }
            return bmp
        } catch (e: Exception) {
            AppLogger.warn("UriLoader: ImageDecoder failed for $uri, falling back. ${e.message}")
        }

        context.contentResolver.openFileDescriptor(uri, "r")?.use { pfd ->
            val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFileDescriptor(pfd.fileDescriptor, null, options)

            val decodeOptions = BitmapFactory.Options().apply {
                inSampleSize =
                    calculateInSampleSize(options.outWidth, options.outHeight, maxMegapixels)
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            return BitmapFactory.decodeFileDescriptor(pfd.fileDescriptor, null, decodeOptions)
                ?: throw IOException("BitmapFactory.decodeFileDescriptor failed for URI: $uri")
        } ?: throw IOException("Could not get FileDescriptor for URI: $uri")
    }

    private fun calculateInSampleSize(width: Int, height: Int, maxMegapixels: Float): Int {
        if (width <= 0 || height <= 0) return 1
        val targetMaxPixels = (maxMegapixels * MEGAPIXEL).toLong()
        var sampleSize = 1
        while ((width / sampleSize).toLong() * (height / sampleSize) > targetMaxPixels) {
            sampleSize *= 2
        }
        AppLogger.info("UriLoader: Downsampling by ${sampleSize}x to fit ${maxMegapixels}MP limit (${width / sampleSize}x${height / sampleSize})")
        return sampleSize
    }
}
