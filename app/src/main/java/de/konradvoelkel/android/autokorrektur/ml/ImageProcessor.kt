package de.konradvoelkel.android.autokorrektur.ml

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import de.konradvoelkel.android.autokorrektur.ml.preprocess.DefaultPreprocessor
import de.konradvoelkel.android.autokorrektur.ml.preprocess.Preprocessor
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import de.konradvoelkel.android.autokorrektur.utils.UriLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.IOException

/**
 * Orchestrates image processing for ML inference by coordinating Uri loading,
 * megapixel scaling, and model-specific preprocessing.
 */
class ImageProcessor(context: Context) : ImagePreprocessingService {

    private val uriLoader = UriLoader(context)
    private val preprocessor: Preprocessor = DefaultPreprocessor()

    companion object {
        private const val DEFAULT_MAX_MEGAPIXELS = 8.0f
    }

    /**
     * Processes an input image URI for ML inference.
     */
    @Throws(IOException::class)
    override fun processInputImage(
        imageUri: Uri,
        modelWidth: Int,
        modelHeight: Int,
        downscaleMp: Float?
    ): ProcessedImage {
        val matsToRelease = mutableListOf<Mat>()
        var originalBitmap: Bitmap? = null
        var transformedBitmap: Bitmap? = null

        try {
            // 1. Load bitmap with safe downsampling to avoid OOM
            originalBitmap = uriLoader.loadRotatedBitmap(imageUri, DEFAULT_MAX_MEGAPIXELS)
            AppLogger.debug("ImageProcessor: Loaded original bitmap (${originalBitmap.width}x${originalBitmap.height})")

            // 2. Convert to OpenCV Mat RGB
            val bgraMat = Mat().also { matsToRelease.add(it) }
            Utils.bitmapToMat(originalBitmap, bgraMat)
            val rgbMat = Mat().also { matsToRelease.add(it) }
            Imgproc.cvtColor(bgraMat, rgbMat, Imgproc.COLOR_RGBA2RGB)

            // 3. Optional further downscaling for performance
            val workingMat = MatScaler.downscaleIfLarge(rgbMat, downscaleMp).also {
                if (it !== rgbMat) matsToRelease.add(it)
            }

            // 4. Preprocess for model
            val prep = preprocessor.prepare(workingMat, modelWidth, modelHeight)
            matsToRelease.add(prep.forBitmap)
            matsToRelease.add(prep.forEngine)

            // 5. Create transformed bitmap for UI
            transformedBitmap = MatScaler.createDisplayBitmap(prep.forBitmap)

            // 6. Build normalized float Mat for engine
            val transformedMat = Mat().also { matsToRelease.add(it) }
            prep.forEngine.convertTo(transformedMat, CvType.CV_32F, 1.0 / 255.0)

            // Cleanup local refs we are returning
            matsToRelease.remove(workingMat)
            matsToRelease.remove(transformedMat)

            return ProcessedImage(
                originalBitmap = originalBitmap,
                transformedBitmap = transformedBitmap,
                originalMat = workingMat,
                transformedMat = transformedMat,
                xRatio = prep.xRatio,
                yRatio = prep.yRatio
            )
        } catch (e: Exception) {
            originalBitmap?.recycle()
            transformedBitmap?.recycle()
            if (e is kotlinx.coroutines.CancellationException) throw e
            throw if (e is IOException) e else IOException(
                "Image processing failed: ${e.message}",
                e
            )
        } finally {
            matsToRelease.forEach { it.release() }
        }
    }

    /** Holds all processed image data. Call [release] when done to free native resources. */
    data class ProcessedImage(
        val originalBitmap: Bitmap,
        val transformedBitmap: Bitmap,
        val originalMat: Mat,
        val transformedMat: Mat,
        val xRatio: Float,
        val yRatio: Float
    ) {
        fun release(recycleBitmaps: Boolean = false) {
            originalMat.release()
            transformedMat.release()
            if (recycleBitmaps) {
                originalBitmap.recycle()
                transformedBitmap.recycle()
            }
            AppLogger.debug("ImageProcessor: Released native Mats (bitmapsRecycled=$recycleBitmaps).")
        }
    }
}
