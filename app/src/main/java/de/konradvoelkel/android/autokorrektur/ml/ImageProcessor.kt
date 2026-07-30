package de.konradvoelkel.android.autokorrektur.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import androidx.core.graphics.createBitmap
import de.konradvoelkel.android.autokorrektur.ml.preprocess.DefaultPreprocessor
import de.konradvoelkel.android.autokorrektur.ml.preprocess.Preprocessor
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.IOException
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Handles image processing operations for ML inference.
 * This class is responsible for loading, transforming, and preparing images
 * to be fed into a machine learning model.
 */
class ImageProcessor(private val context: Context) {

    private val preprocessor: Preprocessor = DefaultPreprocessor()

    companion object {
        /**
         * Maximum initial image size in megapixels to prevent OOM errors when loading
         * very high-resolution images. Images larger than this will be downsampled.
         */
        private const val MAX_INITIAL_MEGAPIXELS = 16.0f
        private const val MEGAPIXEL = 1_000_000f
    }

    /**
     * Processes an input image URI for ML inference.
     *
     * @param imageUri The URI of the image to process.
     * @param modelWidth The target width for the model input.
     * @param modelHeight The target height for the model input.
     * @param downscaleMp Optional megapixels limit to downscale the image for performance.
     * @return A [ProcessedImage] containing the original bitmap, transformed data, and metadata.
     */
    @Throws(IOException::class)
    fun processInputImage(
        imageUri: Uri,
        modelWidth: Int,
        modelHeight: Int,
        downscaleMp: Float? = null
    ): ProcessedImage {
        // 1. Load bitmap with safe downsampling to avoid OOM errors.
        val originalBitmap = loadBitmapFromUri(imageUri)
        AppLogger.debug("ImageProcessor: Loaded original bitmap (${originalBitmap.width}x${originalBitmap.height})")

        // 2. Convert to OpenCV Mat and correct color space.
        // Android Bitmaps are ARGB, but OpenCV's bitmapToMat converts them to BGRA.
        val bgraMat = Mat()
        Utils.bitmapToMat(originalBitmap, bgraMat)
        val rgbMat = Mat()
        Imgproc.cvtColor(bgraMat, rgbMat, Imgproc.COLOR_BGRA2RGB)
        bgraMat.release() // Release intermediate BGRA Mat.

        // 3. Optionally downscale the image further for performance.
        val workingMat = downscaleMatIfLarge(rgbMat, downscaleMp)
        if (workingMat !== rgbMat) {
            rgbMat.release() // Release original if a new downscaled Mat was created.
        }
        AppLogger.debug("ImageProcessor: Working Mat dimensions ${workingMat.width()}x${workingMat.height()}")


        // 4. Preprocess the image for the model (padding, resizing, normalizing to 8U here).
        val prep = preprocessor.prepare(workingMat, modelWidth, modelHeight)

        // 5. Convert the preprocessed Mat back to a Bitmap for display, fixing color channels.
        val transformedBitmap = createDisplayBitmapFromPreprocessedMat(prep.forBitmap)
        AppLogger.debug("ImageProcessor: Created transformed bitmap (${transformedBitmap.width}x${transformedBitmap.height})")

        // 6. Build the normalized float Mat for the model from the 8-bit engine Mat.
        val transformedMat = Mat()
        prep.forEngine.convertTo(transformedMat, CvType.CV_32FC3, 1.0 / 255.0)

        // 7. Release 8-bit mats that are no longer needed on the native side.
        prep.forBitmap.release()
        prep.forEngine.release()

        return ProcessedImage(
            originalBitmap = originalBitmap,
            transformedBitmap = transformedBitmap,
            originalMat = workingMat, // This is the final, possibly downscaled, Mat.
            transformedMat = transformedMat, // This is the normalized Mat for the model.
            xRatio = prep.xRatio,
            yRatio = prep.yRatio
        )
    }

    /**
     * Loads a [Bitmap] from a given [Uri], handling different schemes and applying
     * safe downsampling to prevent out-of-memory errors.
     */
    @Throws(IOException::class)
    private fun loadBitmapFromUri(imageUri: Uri): Bitmap {
        val loaded = when (val scheme = imageUri.scheme?.lowercase()) {
            "file" -> loadBitmapFromFile(imageUri)
            "content" -> loadBitmapFromContentProvider(imageUri)
            else -> throw IOException("Unsupported URI scheme: $scheme")
        }
        return rotateBitmapIfRequired(loaded, imageUri)
    }

    /**
     * Inspects EXIF metadata and rotates the bitmap if it contains an orientation tag.
     */
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
            android.media.ExifInterface(inputStream)
        } catch (e: Exception) {
            return bitmap
        } finally {
            try { inputStream.close() } catch (_: Exception) {}
        }

        val orientation = exif.getAttributeInt(
            android.media.ExifInterface.TAG_ORIENTATION,
            android.media.ExifInterface.ORIENTATION_NORMAL
        )

        val matrix = android.graphics.Matrix()
        when (orientation) {
            android.media.ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            android.media.ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            android.media.ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            android.media.ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            android.media.ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            else -> return bitmap
        }

        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (rotated != bitmap) {
            bitmap.recycle()
        }
        return rotated
    }

    /** Loads a bitmap from a `file://` URI. */
    private fun loadBitmapFromFile(uri: Uri): Bitmap {
        val path = uri.path ?: throw IOException("File URI has no path: $uri")
        if (!File(path).exists()) throw IOException("File not found: $path")

        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, options)

        val imageWidth = options.outWidth
        val imageHeight = options.outHeight
        if (imageWidth <= 0 || imageHeight <= 0) {
            throw IOException("Invalid image dimensions from file: ${imageWidth}x${imageHeight}")
        }

        val decodeOptions = BitmapFactory.Options().apply {
            inSampleSize = calculateInSampleSize(imageWidth, imageHeight, MAX_INITIAL_MEGAPIXELS)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        return BitmapFactory.decodeFile(path, decodeOptions)
            ?: throw IOException("BitmapFactory.decodeFile failed for path: $path")
    }

    /** Loads a bitmap from a `content://` URI, using modern and fallback methods. */
    private fun loadBitmapFromContentProvider(uri: Uri): Bitmap {
        // On modern Android (API 28+), ImageDecoder is the preferred, safest method.
        try {
            val source = ImageDecoder.createSource(context.contentResolver, uri)
            return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                val width = info.size.width
                val height = info.size.height
                val sampleSize = calculateInSampleSize(width, height, MAX_INITIAL_MEGAPIXELS)
                if (sampleSize > 1) {
                    decoder.setTargetSampleSize(sampleSize)
                }
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = true // Required for drawing on it later
            }
        } catch (e: Exception) {
            AppLogger.warn("ImageProcessor: ImageDecoder failed for $uri, falling back. Error: ${e.message}")
            // Fall through to BitmapFactory-based methods.
        }

        // Fallback for older APIs or if ImageDecoder fails.
        context.contentResolver.openFileDescriptor(uri, "r")?.use { pfd ->
            val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFileDescriptor(pfd.fileDescriptor, null, options)

            val imageWidth = options.outWidth
            val imageHeight = options.outHeight
            if (imageWidth <= 0 || imageHeight <= 0) {
                throw IOException("Invalid image dimensions from content URI: ${imageWidth}x${imageHeight}")
            }
            val decodeOptions = BitmapFactory.Options().apply {
                inSampleSize =
                    calculateInSampleSize(imageWidth, imageHeight, MAX_INITIAL_MEGAPIXELS)
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            return BitmapFactory.decodeFileDescriptor(pfd.fileDescriptor, null, decodeOptions)
                ?: throw IOException("BitmapFactory.decodeFileDescriptor failed for URI: $uri")
        } ?: throw IOException("Could not get FileDescriptor for URI: $uri")
    }

    /**
     * Calculates the `inSampleSize` for BitmapFactory to load a downsampled image
     * that is roughly under the megapixel limit.
     */
    private fun calculateInSampleSize(width: Int, height: Int, maxMegapixels: Float): Int {
        if (width <= 0 || height <= 0) return 1
        val imageMegapixels = (width.toLong() * height.toLong()) / MEGAPIXEL
        if (imageMegapixels <= maxMegapixels) {
            return 1
        }

        val scaleFactor = sqrt(imageMegapixels / maxMegapixels)
        // Sample size must be a power of 2.
        var sampleSize = 1
        while (sampleSize * 2 <= scaleFactor) {
            sampleSize *= 2
        }
        AppLogger.info("ImageProcessor: Image is ${imageMegapixels}MP, downsampling by ${sampleSize}x to fit ${maxMegapixels}MP limit")
        return sampleSize
    }

    /**
     * Downscales a Mat if its megapixel count exceeds the specified limit.
     * Returns a new, downscaled Mat, or the original Mat if no scaling was needed.
     */
    private fun downscaleMatIfLarge(mat: Mat, maxMegapixels: Float?): Mat {
        if (maxMegapixels == null) return mat

        val currentMegapixels = (mat.rows() * mat.cols()) / MEGAPIXEL
        if (currentMegapixels <= maxMegapixels) return mat

        val scale = sqrt(maxMegapixels.toDouble() / currentMegapixels)
        val newSize = Size(
            (mat.cols() * scale).roundToInt().toDouble(),
            (mat.rows() * scale).roundToInt().toDouble()
        )

        val downscaledMat = Mat()
        Imgproc.resize(mat, downscaledMat, newSize, 0.0, 0.0, Imgproc.INTER_AREA)
        AppLogger.debug("ImageProcessor: Downscaled Mat to ${downscaledMat.width()}x${downscaledMat.height()}")
        return downscaledMat
    }

    /**
     * Correctly converts a preprocessed 3-channel RGB Mat into a displayable ARGB Bitmap.
     */
    private fun createDisplayBitmapFromPreprocessedMat(rgbMat: Mat): Bitmap {
        // To convert to an ARGB_8888 bitmap, we need a 4-channel BGRA Mat.
        val bgraMat = Mat()
        Imgproc.cvtColor(rgbMat, bgraMat, Imgproc.COLOR_RGB2BGRA)

        val bitmap = createBitmap(bgraMat.cols(), bgraMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(bgraMat, bitmap)

        bgraMat.release() // Release the intermediate Mat.
        return bitmap
    }


    /**
     * Holds all processed image data. Call [release] when done to free native resources.
     */
    data class ProcessedImage(
        val originalBitmap: Bitmap,
        val transformedBitmap: Bitmap,
        val originalMat: Mat,      // The final working Mat (possibly downscaled), in RGB.
        val transformedMat: Mat,   // The normalized Mat for model input, in RGB.
        val xRatio: Float,
        val yRatio: Float
    ) {
        /** Releases the native OpenCV [Mat] objects to prevent memory leaks. */
        fun release() {
            originalMat.release()
            transformedMat.release()
            AppLogger.debug("ImageProcessor: Released Mat objects in ProcessedImage.")
        }
    }
}
