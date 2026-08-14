package de.konradvoelkel.android.autokorrektur.shared

import android.content.Context
import android.graphics.Bitmap
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream

/**
 * Centralized test assertion utility that executes a second-pass YOLO vehicle detection
 * on inpainted images to assert that all vehicles have been completely and cleanly eliminated.
 */
object PostInpaintingVehicleAssertionUtils {

    private const val DEFAULT_CONFIDENCE_THRESHOLD = 0.25f

    /**
     * Asserts that no vehicles remain in an inpainted Bitmap.
     *
     * @param inpaintedBitmap The post-inpainting output Bitmap
     * @param context Application/Test Context
     * @param yoloService Initialized YOLO detection service
     * @param imageProcessor Initialized ImageProcessor
     * @param confidenceThreshold Confidence threshold to flag remaining vehicles (default 0.25)
     * @param message Custom failure message prefix
     */
    suspend fun assertNoVehiclesRemain(
        inpaintedBitmap: Bitmap,
        context: Context,
        yoloService: YoloService,
        imageProcessor: ImageProcessor,
        confidenceThreshold: Float = DEFAULT_CONFIDENCE_THRESHOLD,
        message: String = "Post-inpainting verification failed: vehicles still detected in output"
    ) {
        val tempFile = File(context.cacheDir, "post_inpaint_verify_${System.currentTimeMillis()}.png")
        try {
            FileOutputStream(tempFile).use { fos ->
                inpaintedBitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
            }
            val tempUri = android.net.Uri.fromFile(tempFile)
            val processed = imageProcessor.processInputImage(
                imageUri = tempUri,
                modelWidth = 640,
                modelHeight = 640,
                downscaleMp = null
            )

            try {
                val config = YoloConfig(scoreThreshold = confidenceThreshold)
                val yoloResult = yoloService.inferDetailed(
                    transformedMat = processed.transformedMat,
                    xRatio = processed.xRatio,
                    yRatio = processed.yRatio,
                    upscaleFactor = 1.02f,
                    originalWidth = processed.originalMat.cols(),
                    originalHeight = processed.originalMat.rows(),
                    overrideConfig = config
                )

                try {
                    val remainingVehicles = yoloResult.detections.filter { det ->
                        det.classId in config.vehicleClassIndices && det.confidence >= confidenceThreshold
                    }

                    if (remainingVehicles.isNotEmpty()) {
                        val details = remainingVehicles.joinToString(separator = "\n") { det ->
                            val label = if (det.classId in config.labels.indices) config.labels[det.classId] else "class_${det.classId}"
                            " - [$label] conf=${String.format("%.3f", det.confidence)}, box=[x=${String.format("%.3f", det.x)}, y=${String.format("%.3f", det.y)}, w=${String.format("%.3f", det.width)}, h=${String.format("%.3f", det.height)}]"
                        }
                        AppLogger.error("$message:\n$details")
                        fail("$message. Found ${remainingVehicles.size} residual vehicle(s):\n$details")
                    }
                } finally {
                    yoloResult.mask.release()
                }
            } finally {
                processed.release(recycleBitmaps = true)
            }
        } finally {
            tempFile.delete()
        }
    }

    /**
     * Asserts that no vehicles remain in an inpainted OpenCV Mat (CV_8UC3 RGB).
     */
    suspend fun assertNoVehiclesRemain(
        inpaintedMat: Mat,
        context: Context,
        yoloService: YoloService,
        imageProcessor: ImageProcessor,
        confidenceThreshold: Float = DEFAULT_CONFIDENCE_THRESHOLD,
        message: String = "Post-inpainting verification failed: vehicles still detected in Mat"
    ) {
        val rgbaMat = Mat()
        try {
            when (inpaintedMat.channels()) {
                3 -> Imgproc.cvtColor(inpaintedMat, rgbaMat, Imgproc.COLOR_RGB2RGBA)
                4 -> inpaintedMat.copyTo(rgbaMat)
                else -> Imgproc.cvtColor(inpaintedMat, rgbaMat, Imgproc.COLOR_GRAY2RGBA)
            }
            val bitmap = Bitmap.createBitmap(rgbaMat.cols(), rgbaMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(rgbaMat, bitmap)
            try {
                assertNoVehiclesRemain(
                    inpaintedBitmap = bitmap,
                    context = context,
                    yoloService = yoloService,
                    imageProcessor = imageProcessor,
                    confidenceThreshold = confidenceThreshold,
                    message = message
                )
            } finally {
                bitmap.recycle()
            }
        } finally {
            rgbaMat.release()
        }
    }
}
