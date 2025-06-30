package de.konradvoelkel.android.autokorrektur.ml

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.IOException
import androidx.core.graphics.createBitmap
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Handles image processing operations for ML inference.
 * Equivalent to processInput.js in the web app.
 */
class ImageProcessor(private val context: Context) {

    /**
     * Processes an input image for ML inference.
     *
     * @param uri The URI of the image to process
     * @param modelWidth The width of the model input
     * @param modelHeight The height of the model input
     * @param downscaleMp The maximum megapixels to downscale to, or null for no downscaling
     * @return A triple containing:
     *   - The original RGB bitmap
     *   - The transformed bitmap for model input
     *   - The x ratio of the image
     *   - The y ratio of the image
     */
    @Throws(IOException::class)
    fun processInputImage(
        uri: Uri,
        modelWidth: Int,
        modelHeight: Int,
        downscaleMp: Float? = null
    ): ProcessedImage {
        // Load the image from URI
        val originalBitmap = loadBitmapFromUri(uri)

        // Convert to OpenCV Mat
        val rgbMat = Mat()
        Utils.bitmapToMat(originalBitmap, rgbMat)
        Imgproc.cvtColor(rgbMat, rgbMat, Imgproc.COLOR_RGBA2RGB)

        // Optionally downscale the image
        if (downscaleMp != null) {
            val currentMegapixels = (rgbMat.rows() * rgbMat.cols()) / 1000000f

            if (currentMegapixels > downscaleMp) {
                val scaleFactor = sqrt(downscaleMp.toDouble() / currentMegapixels.toDouble())

                val newWidth = Math.round(rgbMat.cols() * scaleFactor)
                val newHeight = Math.round(rgbMat.rows() * scaleFactor)

                Imgproc.resize(rgbMat, rgbMat, Size(newWidth.toDouble(), newHeight.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            }
        }

        // Preprocess the image for model input
        val preprocessingResult = preprocessing(rgbMat, modelWidth, modelHeight)

        // Convert Mats back to Bitmaps
        val transformedBitmap = createBitmap(preprocessingResult.transformedMatForBitmap.cols(), preprocessingResult.transformedMatForBitmap.rows())
        Utils.matToBitmap(preprocessingResult.transformedMatForBitmap, transformedBitmap)

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
        Imgproc.resize(rgbMat, resizedMat, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LANCZOS4)

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
        Imgproc.resize(paddedMat, transformedMatForBitmap, Size(modelWidth.toDouble(), modelHeight.toDouble()))

        // Create normalized version for ML inference
        val transformedMat = Mat()
        transformedMatForBitmap.convertTo(transformedMat, CvType.CV_32FC3, 1.0/255.0)

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
        var w = width
        var h = height

        if (w % stride != 0) {
            w = if (w % stride >= stride / 2) {
                (w / stride + 1) * stride
            } else {
                (w / stride) * stride
            }
        }

        if (h % stride != 0) {
            h = if (h % stride >= stride / 2) {
                (h / stride + 1) * stride
            } else {
                (h / stride) * stride
            }
        }

        return Pair(w, h)
    }

    /**
     * Loads a bitmap from a URI.
     *
     * @param uri The URI of the image
     * @return The loaded bitmap
     */
    @Throws(IOException::class)
    private fun loadBitmapFromUri(uri: Uri): Bitmap {
        val inputStream = context.contentResolver.openInputStream(uri)
            ?: throw IOException("Could not open input stream for URI: $uri")

        val bitmap = android.graphics.BitmapFactory.decodeStream(inputStream)
        inputStream.close()

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
