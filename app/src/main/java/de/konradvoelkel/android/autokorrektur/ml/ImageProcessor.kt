package de.konradvoelkel.android.autokorrektur.ml

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.core.graphics.createBitmap
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.IOException
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Handles image processing operations for ML inference.
 * Equivalent to processInput.js in the web app.
 */
class ImageProcessor(private val context: Context) {

    /**
     * Processes an input image for ML inference.
     *
     * @param imageUri The URI of the image to process
     * @param modelWidth The width of the model input
     * @param modelHeight The height of the model input
     * @param downscaleMp The maximum megapixels to downscale to, or null for no downscaling
     * @return A triple containing:
     * - The original RGB bitmap
     * - The transformed bitmap for model input
     * - The x ratio of the image
     * - The y ratio of the image
     */
    @Throws(IOException::class)
    fun processInputImage(
        imageUri: Uri,
        modelWidth: Int,
        modelHeight: Int,
        downscaleMp: Float? = null
    ): ProcessedImage {
        // Load the image from URI
        val originalBitmap = loadBitmapFromUri(imageUri)
        AppLogger.debug("ImageProcessor: originalBitmap loaded with dimensions ${originalBitmap.width}x${originalBitmap.height}")


        // Convert to OpenCV Mat
        val rgbMat = Mat()
        Utils.bitmapToMat(originalBitmap, rgbMat)
        // CRITICAL CHANGE: Convert from RGBA (Android Bitmap default) to RGB, not BGR.
        // Most YOLO models expect RGB channel order.
        Imgproc.cvtColor(rgbMat, rgbMat, Imgproc.COLOR_RGBA2RGB)

        // Optionally downscale the image
        if (downscaleMp != null) {
            val currentMegapixels = (rgbMat.rows() * rgbMat.cols()) / 1000000f

            if (currentMegapixels > downscaleMp) {
                val scaleFactor = sqrt(downscaleMp.toDouble() / currentMegapixels.toDouble())

                val newWidth = (rgbMat.cols() * scaleFactor).roundToInt()
                val newHeight = (rgbMat.rows() * scaleFactor).roundToInt()

                Imgproc.resize(
                    rgbMat,
                    rgbMat,
                    Size(newWidth.toDouble(), newHeight.toDouble()),
                    0.0,
                    0.0,
                    Imgproc.INTER_AREA
                )
            }
        }

        // Preprocess the image for model input
        val preprocessingResult = preprocessing(rgbMat, modelWidth, modelHeight)

        // Convert Mats back to Bitmaps
        val transformedBitmap = createBitmap(
            preprocessingResult.transformedMatForBitmap.cols(),
            preprocessingResult.transformedMatForBitmap.rows()
        )
        Utils.matToBitmap(preprocessingResult.transformedMatForBitmap, transformedBitmap)
        AppLogger.debug("ImageProcessor: transformedBitmap created with dimensions ${transformedBitmap.width}x${transformedBitmap.height}")

        return ProcessedImage(
            originalBitmap = originalBitmap,
            transformedBitmap = transformedBitmap,
            originalMat = rgbMat,
            transformedMat = preprocessingResult.transformedMat,
            xRatio = preprocessingResult.xRatio,
            yRatio = preprocessingResult.yRatio
        )
    }

    /**
     * Data class to hold preprocessing results.
     */
    private data class PreprocessingResult(
        val transformedMat: Mat,      // Normalized Mat for ML inference (CV_32FC3)
        val transformedMatForBitmap: Mat,  // 8-bit Mat for bitmap conversion (CV_8UC3)
        val xRatio: Float,
        val yRatio: Float
    )

    /**
     * Preprocesses an image for model input.
     *
     * @param rgbMat The RGB image matrix
     * @param modelWidth The width of the model input
     * @param modelHeight The height of the model input
     * @param stride The stride value for dimension adjustment
     * @return A PreprocessingResult containing the transformed matrices and ratios
     */
    private fun preprocessing(
        rgbMat: Mat,
        modelWidth: Int,
        modelHeight: Int,
        stride: Int = 32
    ): PreprocessingResult {
        // Resize to dimensions divisible by stride (pure helper moved to ImageProcessingUtils)
        val (w, h) = ImageProcessingUtils.divStride(stride, rgbMat.cols(), rgbMat.rows())
        val resizedMat = Mat()
        Imgproc.resize(
            rgbMat,
            resizedMat,
            Size(w.toDouble(), h.toDouble()),
            0.0,
            0.0,
            Imgproc.INTER_LANCZOS4
        )

        // Padding image to square dimensions
        val maxSize = max(resizedMat.rows(), resizedMat.cols())
        val xPad = maxSize - resizedMat.cols()
        val xRatio = maxSize.toFloat() / resizedMat.cols()
        val yPad = maxSize - resizedMat.rows()
        val yRatio = maxSize.toFloat() / resizedMat.rows()

        val paddedMat = Mat()
        Core.copyMakeBorder(
            resizedMat, paddedMat, 0, yPad, 0, xPad, Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0)
        )

        // Resize to model input size
        val transformedMatForBitmap = Mat()
        Imgproc.resize(
            paddedMat,
            transformedMatForBitmap,
            Size(modelWidth.toDouble(), modelHeight.toDouble())
        )

        // Create normalized version for ML inference
        val transformedMat = Mat()

        // log first 10 bytes in first row of transformedMatForBitmap
        AppLogger.debug(
            "ImageProcessor: transformedMatForBitmap[0] = ${
                transformedMatForBitmap.get(
                    0,
                    0
                ).contentToString()
            }"
        )

        transformedMatForBitmap.convertTo(transformedMat, CvType.CV_32FC3, 1.0 / 255.0)

        AppLogger.debug(
            "ImageProcessor: transformedMat[0] = ${
                transformedMat.get(0, 0).contentToString()
            }"
        )


        // Release intermediate Mats
        resizedMat.release()
        paddedMat.release()

        return PreprocessingResult(transformedMat, transformedMatForBitmap, xRatio, yRatio)
    }


    /**
     * Loads a bitmap from a URI with intelligent downsampling to prevent OOM errors.
     *
     * This method first decodes the image dimensions without loading the full bitmap,
     * then calculates an appropriate sample size to ensure the image fits within
     * reasonable memory constraints.
     *
     * @param imageUri The URI of the image
     * @return The loaded bitmap, potentially downsampled to fit memory constraints
     */
    @Throws(IOException::class)
    private fun loadBitmapFromUri(imageUri: Uri): Bitmap {
        // Prefer FileDescriptor-based decoding. Some Photo Picker providers return null streams
        // for content:// URIs but do support file descriptors. On Android 9+ we will prefer
        // ImageDecoder for content URIs which is more compatible with modern providers.
        fun openFileDescriptorCompat(uri: Uri): android.os.ParcelFileDescriptor? {
            return try {
                when (uri.scheme?.lowercase()) {
                    null, "file" -> {
                        val path = uri.path ?: return null
                        val file = File(path)
                        if (!file.exists()) return null
                        android.os.ParcelFileDescriptor.open(
                            file,
                            android.os.ParcelFileDescriptor.MODE_READ_ONLY
                        )
                    }
                    else -> context.contentResolver.openFileDescriptor(uri, "r")
                }
            } catch (_: Exception) {
                null
            }
        }

        // Maximum pixels to load initially (before any downscaling).
        // This prevents OOM when loading very high-resolution images.
        // 16MP is a reasonable limit that most modern devices can handle.
        val maxInitialMegapixels = 16.0f
        val pixelsToMegapixels = 1_000_000f

        val scheme = imageUri.scheme?.lowercase()
        val authority = imageUri.authority
        AppLogger.debug("ImageProcessor: Decoding URI scheme=$scheme authority=$authority uri=$imageUri")

        // For file URIs, keep using BitmapFactory fast-path.
        if (scheme == null || scheme == "file") {
            val options = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }
            val path = imageUri.path ?: throw IOException("File URI has no path: $imageUri")
            if (!File(path).exists()) throw IOException("File not found: $path")
            android.graphics.BitmapFactory.decodeFile(path, options)

            val imageWidth = options.outWidth
            val imageHeight = options.outHeight
            if (imageWidth <= 0 || imageHeight <= 0) throw IOException("Invalid image dimensions: ${imageWidth}x${imageHeight}")
            AppLogger.debug("ImageProcessor: Image dimensions from file URI: ${imageWidth}x${imageHeight}")

            var inSampleSize = 1
            val imageMegapixels = (imageWidth * imageHeight) / pixelsToMegapixels
            if (imageMegapixels > maxInitialMegapixels) {
                val scaleFactor = sqrt(imageMegapixels / maxInitialMegapixels)
                while (inSampleSize * 2 <= scaleFactor) inSampleSize *= 2
                AppLogger.info("ImageProcessor: Image is ${imageMegapixels}MP, downsampling by ${inSampleSize}x to fit ${maxInitialMegapixels}MP limit")
            }

            val decodeOptions = android.graphics.BitmapFactory.Options().apply {
                this.inSampleSize = inSampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            val bitmap = android.graphics.BitmapFactory.decodeFile(path, decodeOptions)
                ?: throw IOException("Could not decode bitmap from file path: $path")
            AppLogger.debug("ImageProcessor: Loaded bitmap with dimensions ${bitmap.width}x${bitmap.height} (inSampleSize=$inSampleSize)")
            return bitmap
        }

        // For content URIs, prefer ImageDecoder on API 28+, as some providers do not support raw
        // openInputStream or file descriptors depending on OEM implementation.
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            try {
                val source = android.graphics.ImageDecoder.createSource(context.contentResolver, imageUri)
                var headerW = -1
                var headerH = -1
                // Decode using header to compute safe target sample
                val bitmap = android.graphics.ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                    headerW = info.size.width
                    headerH = info.size.height
                    if (headerW <= 0 || headerH <= 0) {
                        // Let it throw later if decode fails
                        return@decodeBitmap
                    }
                    val imageMp = (headerW.toFloat() * headerH.toFloat()) / pixelsToMegapixels
                    var sample = 1
                    if (imageMp > maxInitialMegapixels) {
                        val scaleFactor = sqrt(imageMp / maxInitialMegapixels)
                        while (sample * 2 <= scaleFactor) sample *= 2
                        AppLogger.info("ImageProcessor: (ImageDecoder) Image is ${imageMp}MP, setTargetSampleSize=${sample}")
                        decoder.setTargetSampleSize(sample)
                    }
                    decoder.allocator = android.graphics.ImageDecoder.ALLOCATOR_SOFTWARE
                    decoder.isMutableRequired = false
                }
                AppLogger.debug("ImageProcessor: Loaded bitmap via ImageDecoder with dimensions ${bitmap.width}x${bitmap.height}")
                return bitmap
            } catch (e: Exception) {
                AppLogger.warn("ImageProcessor: ImageDecoder path failed for $imageUri: ${e.message}")
                // fall through to FD/stream attempts
            }
        }

        // Fallback for content URIs on older APIs or if ImageDecoder failed
        // First, try a file descriptor
        openFileDescriptorCompat(imageUri)?.use { pfd ->
            // Decode bounds
            val bounds = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }
            android.graphics.BitmapFactory.decodeFileDescriptor(pfd.fileDescriptor, null, bounds)
            val imageWidth = bounds.outWidth
            val imageHeight = bounds.outHeight
            if (imageWidth <= 0 || imageHeight <= 0) {
                AppLogger.warn("ImageProcessor: Could not read dimensions from FD for $imageUri")
            }
            var inSampleSize = 1
            val imageMegapixels = if (imageWidth > 0 && imageHeight > 0) (imageWidth * imageHeight) / pixelsToMegapixels else 0f
            if (imageMegapixels > maxInitialMegapixels) {
                val scaleFactor = sqrt(imageMegapixels / maxInitialMegapixels)
                while (inSampleSize * 2 <= scaleFactor) inSampleSize *= 2
            }
            val decodeOptions = android.graphics.BitmapFactory.Options().apply {
                this.inSampleSize = inSampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            val bmp = android.graphics.BitmapFactory.decodeFileDescriptor(pfd.fileDescriptor, null, decodeOptions)
            if (bmp != null) return bmp
        }

        // Last resort: InputStream
        val inputStream = try { context.contentResolver.openInputStream(imageUri) } catch (_: Exception) { null }
        inputStream?.use { stream ->
            // We cannot do a true two-pass with the same stream easily; just decode directly.
            val bmp = android.graphics.BitmapFactory.decodeStream(stream)
            if (bmp != null) return bmp
        }

        throw IOException("Could not open input stream or file descriptor for URI: $imageUri")
    }

    /**
     * Data class to hold processed image data.
     */
    data class ProcessedImage(
        val originalBitmap: Bitmap,
        val transformedBitmap: Bitmap,
        val originalMat: Mat,
        val transformedMat: Mat,
        val xRatio: Float,
        val yRatio: Float
    )
}
