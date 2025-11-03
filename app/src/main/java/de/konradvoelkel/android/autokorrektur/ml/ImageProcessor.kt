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
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream
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
        // Resize to dimensions divisible by stride
        val (w, h) = divStride(stride, rgbMat.cols(), rgbMat.rows())
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
     * Get dimensions divisible by stride.
     *
     * @param stride The stride value
     * @param width The original width
     * @param height The original height
     * @return A pair of width and height divisible by stride
     */
    private fun divStride(stride: Int, width: Int, height: Int): Pair<Int, Int> {
        var widthDivisibleByStride = width
        var heightDivisibleByStride = height

        if (widthDivisibleByStride % stride != 0) {
            widthDivisibleByStride = if (widthDivisibleByStride % stride >= stride / 2) {
                (widthDivisibleByStride / stride + 1) * stride
            } else {
                (widthDivisibleByStride / stride) * stride
            }
        }

        if (heightDivisibleByStride % stride != 0) {
            heightDivisibleByStride = if (heightDivisibleByStride % stride >= stride / 2) {
                (heightDivisibleByStride / stride + 1) * stride
            } else {
                (heightDivisibleByStride / stride) * stride
            }
        }

        return Pair(widthDivisibleByStride, heightDivisibleByStride)
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
        // for content:// URIs but do support file descriptors.
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
            } catch (e: Exception) {
                null
            }
        }

        // Maximum pixels to load initially (before any downscaling).
        // This prevents OOM when loading very high-resolution images.
        // 16MP is a reasonable limit that most modern devices can handle.
        val maxInitialMegapixels = 16.0f

        val scheme = imageUri.scheme?.lowercase()
        val authority = imageUri.authority
        AppLogger.debug("ImageProcessor: Decoding URI scheme=$scheme authority=$authority uri=$imageUri")

        // First pass: decode image dimensions without loading the full bitmap
        val options = android.graphics.BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }

        if (scheme == null || scheme == "file") {
            val path = imageUri.path ?: throw IOException("File URI has no path: $imageUri")
            if (!File(path).exists()) throw IOException("File not found: $path")
            android.graphics.BitmapFactory.decodeFile(path, options)
        } else {
            openFileDescriptorCompat(imageUri)?.use { pfd ->
                android.graphics.BitmapFactory.decodeFileDescriptor(pfd.fileDescriptor, null, options)
            } ?: run {
                // Fallback to stream if FD could not be opened
                val inputStream = try { context.contentResolver.openInputStream(imageUri) } catch (_: Exception) { null }
                inputStream?.use { stream ->
                    android.graphics.BitmapFactory.decodeStream(stream, null, options)
                } ?: throw IOException("Could not open input stream or file descriptor for URI: $imageUri")
            }
        }

        val imageWidth = options.outWidth
        val imageHeight = options.outHeight
        if (imageWidth <= 0 || imageHeight <= 0) {
            throw IOException("Invalid image dimensions: ${imageWidth}x${imageHeight}")
        }
        AppLogger.debug("ImageProcessor: Image dimensions from URI: ${imageWidth}x${imageHeight}")

        // Calculate the megapixels (pixels to megapixels conversion factor)
        val pixelsToMegapixels = 1_000_000f
        val imageMegapixels = (imageWidth * imageHeight) / pixelsToMegapixels

        // Calculate inSampleSize to reduce memory usage for very large images
        // inSampleSize is a power of 2 that reduces both dimensions by that factor
        var inSampleSize = 1
        if (imageMegapixels > maxInitialMegapixels) {
            val scaleFactor = sqrt(imageMegapixels / maxInitialMegapixels)
            while (inSampleSize * 2 <= scaleFactor) {
                inSampleSize *= 2
            }
            AppLogger.info("ImageProcessor: Image is ${imageMegapixels}MP, downsampling by ${inSampleSize}x to fit ${maxInitialMegapixels}MP limit")
        }

        // Second pass: decode the bitmap with the calculated sample size
        val decodeOptions = android.graphics.BitmapFactory.Options().apply {
            this.inSampleSize = inSampleSize
            inPreferredConfig = android.graphics.Bitmap.Config.ARGB_8888
        }

        val bitmap = if (scheme == null || scheme == "file") {
            val path = imageUri.path ?: throw IOException("File URI has no path: $imageUri")
            if (!File(path).exists()) throw IOException("File not found: $path")
            android.graphics.BitmapFactory.decodeFile(path, decodeOptions)
        } else {
            openFileDescriptorCompat(imageUri)?.use { pfd ->
                android.graphics.BitmapFactory.decodeFileDescriptor(pfd.fileDescriptor, null, decodeOptions)
            } ?: run {
                // Fallback to stream if FD could not be opened
                val inputStream = try { context.contentResolver.openInputStream(imageUri) } catch (_: Exception) { null }
                inputStream?.use { stream ->
                    android.graphics.BitmapFactory.decodeStream(stream, null, decodeOptions)
                }
            }
        } ?: throw IOException("Could not decode bitmap from URI: $imageUri")

        AppLogger.debug("ImageProcessor: Loaded bitmap with dimensions ${bitmap.width}x${bitmap.height} (inSampleSize=$inSampleSize)")
        return bitmap
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
