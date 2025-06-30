package com.autokorrektur.android.data.processing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import timber.log.Timber
import java.io.InputStream
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.*

@Singleton
class ImageProcessingService @Inject constructor(
    @ApplicationContext private val context: Context
) {

    companion object {
        private const val MODEL_WIDTH = 640
        private const val MODEL_HEIGHT = 640
        private const val STRIDE = 32
    }

    data class ProcessingResult(
        val originalMat: Mat,
        val transformedMat: Mat,
        val xRatio: Double,
        val yRatio: Double
    )

    /**
     * Process input image from URI, similar to processInputImage in the web version
     */
    suspend fun processInputImage(
        imageUri: Uri,
        maxMegapixels: Double? = null
    ): ProcessingResult = withContext(Dispatchers.Default) {

        try {
            Timber.d("Processing input image from URI: $imageUri")

            // Load image from URI
            val bitmap = loadBitmapFromUri(imageUri)

            // Convert bitmap to OpenCV Mat
            val originalMat = Mat()
            Utils.bitmapToMat(bitmap, originalMat)

            // Convert RGBA to RGB
            val rgbMat = Mat()
            Imgproc.cvtColor(originalMat, rgbMat, Imgproc.COLOR_RGBA2RGB)
            originalMat.release()

            // Apply downscaling if specified
            val processedMat = if (maxMegapixels != null) {
                applyDownscaling(rgbMat, maxMegapixels)
            } else {
                rgbMat
            }

            // Preprocess for model input
            val (transformedMat, xRatio, yRatio) = preprocessing(processedMat, MODEL_WIDTH, MODEL_HEIGHT)

            Timber.d("Image processing completed. Original size: ${processedMat.cols()}x${processedMat.rows()}")

            return@withContext ProcessingResult(processedMat, transformedMat, xRatio, yRatio)

        } catch (e: Exception) {
            Timber.e(e, "Failed to process input image")
            throw e
        }
    }

    /**
     * Load bitmap from URI
     */
    private fun loadBitmapFromUri(uri: Uri): Bitmap {
        val inputStream: InputStream = context.contentResolver.openInputStream(uri)
            ?: throw IllegalArgumentException("Cannot open input stream for URI: $uri")

        val bitmap = BitmapFactory.decodeStream(inputStream)
        inputStream.close()

        return bitmap ?: throw IllegalArgumentException("Cannot decode bitmap from URI: $uri")
    }

    /**
     * Apply downscaling based on megapixel limit, similar to web version
     */
    private fun applyDownscaling(mat: Mat, maxMegapixels: Double): Mat {
        val currentMegapixels = (mat.rows() * mat.cols()) / 1_000_000.0

        if (currentMegapixels <= maxMegapixels) {
            return mat
        }

        val scaleFactor = sqrt(maxMegapixels / currentMegapixels)
        val newWidth = (mat.cols() * scaleFactor).roundToInt()
        val newHeight = (mat.rows() * scaleFactor).roundToInt()

        val resizedMat = Mat()
        Imgproc.resize(mat, resizedMat, Size(newWidth.toDouble(), newHeight.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)

        mat.release()
        Timber.d("Downscaled image from ${currentMegapixels}MP to ${maxMegapixels}MP. New size: ${newWidth}x${newHeight}")

        return resizedMat
    }

    /**
     * Preprocessing image to model shape, ported from web version
     */
    private fun preprocessing(mat: Mat, modelWidth: Int, modelHeight: Int): Triple<Mat, Double, Double> {
        // Resize to be divisible by stride
        val (newWidth, newHeight) = divStride(STRIDE, mat.cols(), mat.rows())
        val resizedMat = Mat()
        Imgproc.resize(mat, resizedMat, Size(newWidth.toDouble(), newHeight.toDouble()), 0.0, 0.0, Imgproc.INTER_LANCZOS4)

        // Padding image to square dimensions
        val maxSize = max(resizedMat.rows(), resizedMat.cols())
        val xPad = maxSize - resizedMat.cols()
        val yPad = maxSize - resizedMat.rows()
        val xRatio = maxSize.toDouble() / resizedMat.cols()
        val yRatio = maxSize.toDouble() / resizedMat.rows()

        val paddedMat = Mat()
        Core.copyMakeBorder(
            resizedMat, paddedMat,
            0, yPad, 0, xPad,
            Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0)
        )

        // Resize to model input size and normalize
        val finalMat = Mat()
        Imgproc.resize(paddedMat, finalMat, Size(modelWidth.toDouble(), modelHeight.toDouble()))

        // Convert to float and normalize to [0, 1]
        val normalizedMat = Mat()
        finalMat.convertTo(normalizedMat, CvType.CV_32F, 1.0 / 255.0)

        // Convert from HWC to CHW format for ONNX
        val chwMat = convertHWCtoCHW(normalizedMat)

        // Cleanup intermediate matrices
        resizedMat.release()
        paddedMat.release()
        finalMat.release()
        normalizedMat.release()

        return Triple(chwMat, xRatio, yRatio)
    }

    /**
     * Convert image from HWC (Height-Width-Channel) to CHW (Channel-Height-Width) format
     */
    private fun convertHWCtoCHW(mat: Mat): Mat {
        val channels = mutableListOf<Mat>()
        Core.split(mat, channels)

        val chwMat = Mat(1, 3 * mat.rows() * mat.cols(), CvType.CV_32F)
        val chwData = FloatArray(3 * mat.rows() * mat.cols())

        val channelSize = mat.rows() * mat.cols()

        for (c in 0 until 3) {
            val channelData = FloatArray(channelSize)
            channels[c].get(0, 0, channelData)

            System.arraycopy(channelData, 0, chwData, c * channelSize, channelSize)
        }

        chwMat.put(0, 0, chwData)

        // Cleanup
        channels.forEach { it.release() }

        return chwMat
    }

    /**
     * Get divisible image size by stride, ported from web version
     */
    private fun divStride(stride: Int, width: Int, height: Int): Pair<Int, Int> {
        var newWidth = width
        var newHeight = height

        if (newWidth % stride != 0) {
            newWidth = if (newWidth % stride >= stride / 2) {
                (newWidth / stride + 1) * stride
            } else {
                (newWidth / stride) * stride
            }
        }

        if (newHeight % stride != 0) {
            newHeight = if (newHeight % stride >= stride / 2) {
                (newHeight / stride + 1) * stride
            } else {
                (newHeight / stride) * stride
            }
        }

        return Pair(newWidth, newHeight)
    }

    /**
     * Convert OpenCV Mat to Bitmap for display
     */
    fun matToBitmap(mat: Mat): Bitmap {
        val bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bitmap)
        return bitmap
    }

    /**
     * Resize mask to match original image dimensions
     */
    fun resizeMask(mask: Mat, targetWidth: Int, targetHeight: Int): Mat {
        val resizedMask = Mat()
        Imgproc.resize(mask, resizedMask, Size(targetWidth.toDouble(), targetHeight.toDouble()), 0.0, 0.0, Imgproc.INTER_LANCZOS4)
        return resizedMask
    }

    /**
     * Apply downshift to mask (move mask down by percentage of image height)
     */
    fun shiftMaskDown(mask: Mat, downshiftPercentage: Double): Mat {
        val shiftPixels = (mask.rows() * downshiftPercentage).toInt()

        if (shiftPixels <= 0) return mask.clone()

        val shiftedMask = Mat.zeros(mask.size(), mask.type())

        // Copy the mask shifted down
        val srcRect = Rect(0, 0, mask.cols(), mask.rows() - shiftPixels)
        val dstRect = Rect(0, shiftPixels, mask.cols(), mask.rows() - shiftPixels)

        mask.submat(srcRect).copyTo(shiftedMask.submat(dstRect))

        return shiftedMask
    }

    /**
     * Combine two masks using bitwise AND operation
     */
    fun combineMasks(mask1: Mat, mask2: Mat): Mat {
        val combinedMask = Mat()
        Core.bitwise_and(mask1, mask2, combinedMask)
        return combinedMask
    }

    /**
     * Create overlay visualization of mask on original image
     */
    fun createMaskOverlay(originalMat: Mat, mask: Mat): Mat {
        val overlay = originalMat.clone()
        val colorMask = Mat()

        // Convert grayscale mask to 3-channel
        Imgproc.cvtColor(mask, colorMask, Imgproc.COLOR_GRAY2RGB)

        // Apply red tint to masked areas
        val redMask = Mat.zeros(originalMat.size(), originalMat.type())
        redMask.setTo(Scalar(255.0, 0.0, 0.0), mask)

        // Blend with original image
        Core.addWeighted(overlay, 0.7, redMask, 0.3, 0.0, overlay)

        colorMask.release()
        redMask.release()

        return overlay
    }
}
