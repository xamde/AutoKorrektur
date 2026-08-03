package de.konradvoelkel.android.autokorrektur.ml

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.MediumTest
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine
import de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.io.InputStream
import kotlin.math.max
import kotlin.math.min

/**
 * Isolated Benchmark & Evaluation Suite for Segmentation Mask Quality.
 *
 * Computes quantitative metrics:
 *  - Intersection over Union (IoU)
 *  - Dice Similarity Coefficient (F1 Score)
 *
 * Compares model predicted mask against ground-truth mask assets.
 */
@RunWith(AndroidJUnit4::class)
@MediumTest
class MaskQualityBenchmarkTest : AndroidInstrumentedBaseTest() {

    private lateinit var yoloService: YoloServiceImpl
    private lateinit var imageProcessor: ImageProcessor

    @Before
    fun setUp() = kotlinx.coroutines.runBlocking {
        assertTrue("OpenCV initialization failed", OpenCVLoader.initLocal())
        yoloService = YoloServiceImpl(YoloTFLiteEngine(appContext))
        yoloService.initialize()
        imageProcessor = ImageProcessor(appContext)
    }

    data class SegmentationMetrics(
        val sampleName: String,
        val iou: Float,
        val dice: Float,
        val predictedMaskArea: Int,
        val gtMaskArea: Int
    )

    @Test
    fun benchmarkSegmentationMaskQuality_calculatesQuantitativeIoUAndDice() =
        kotlinx.coroutines.runBlocking {
        val benchmarkSamples = listOf(
            Pair("photo_with_car_1.png", "photo_with_car_1_mask.png")
        )

        val results = mutableListOf<SegmentationMetrics>()

        val testAssets = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().context.assets
        for ((imgName, maskName) in benchmarkSamples) {
            val imgInputStream: InputStream = testAssets.open(imgName)
            val inputBitmap = BitmapFactory.decodeStream(imgInputStream)
            imgInputStream.close()

            val maskInputStream: InputStream = testAssets.open(maskName)
            val gtMaskBitmap = BitmapFactory.decodeStream(maskInputStream)
            maskInputStream.close()

            // Run ImageProcessor + YOLO inference
            val tempFile = java.io.File(appContext.cacheDir, "benchmark_$imgName")
            java.io.FileOutputStream(tempFile).use { out ->
                inputBitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            val uri = android.net.Uri.fromFile(tempFile)

            val processedImage = imageProcessor.processInputImage(uri, modelWidth = 640, modelHeight = 640)
            val yoloResult = yoloService.inferDetailed(
                transformedMat = processedImage.transformedMat,
                xRatio = processedImage.xRatio,
                yRatio = processedImage.yRatio,
                upscaleFactor = 1.0f,
                originalWidth = processedImage.originalMat.cols(),
                originalHeight = processedImage.originalMat.rows(),
                overrideConfig = YoloConfig(scoreThreshold = 0.5f, iouThreshold = 0.45f)
            )

            val predMaskMat = yoloResult.mask
            val metrics = computeIoUAndDice(predMaskMat, gtMaskBitmap, sampleName = imgName)
            results.add(metrics)

            AppLogger.info("BENCHMARK [$imgName] -> IoU: ${"%.4f".format(metrics.iou)}, Dice: ${"%.4f".format(metrics.dice)}")

            processedImage.release()
            tempFile.delete()
            inputBitmap.recycle()
            gtMaskBitmap.recycle()
        }

        val meanIoU = results.map { it.iou }.average().toFloat()
        val meanDice = results.map { it.dice }.average().toFloat()

        AppLogger.info("=== SEGMENTATION BENCHMARK SUMMARY ===")
        AppLogger.info("Mean IoU: ${"%.4f".format(meanIoU)}")
        AppLogger.info("Mean Dice: ${"%.4f".format(meanDice)}")

        // Quantitative Regression Check: Mean IoU must be >= 0.70 for baseline asset
        assertTrue("Segmentation Mean IoU must be >= 0.70 (got ${"%.4f".format(meanIoU)})", meanIoU >= 0.70f)
    }

    private fun computeIoUAndDice(predMaskMat: Mat, gtBitmap: Bitmap, sampleName: String): SegmentationMetrics {
        val width = predMaskMat.cols()
        val height = predMaskMat.rows()

        val gtScaled = if (gtBitmap.width != width || gtBitmap.height != height) {
            Bitmap.createScaledBitmap(gtBitmap, width, height, true)
        } else gtBitmap

        var intersection = 0
        var union = 0
        var predCount = 0
        var gtCount = 0

        val predBytes = ByteArray(width * height)
        val pred8u = Mat()
        predMaskMat.convertTo(pred8u, CvType.CV_8UC1)
        pred8u.get(0, 0, predBytes)
        pred8u.release()

        val gtPixels = IntArray(width * height)
        gtScaled.getPixels(gtPixels, 0, width, 0, 0, width, height)

        for (i in 0 until width * height) {
            val predVal = predBytes[i].toInt() and 0xFF
            // Pred mask: 0 = masked (car), 255 = background
            val isPredCar = predVal < 128

            val gtColor = gtPixels[i]
            // GT mask: black / low intensity = masked (car)
            val isGtCar = Color.red(gtColor) < 128 || Color.alpha(gtColor) < 128

            if (isPredCar) predCount++
            if (isGtCar) gtCount++

            if (isPredCar && isGtCar) intersection++
            if (isPredCar || isGtCar) union++
        }

        if (gtScaled !== gtBitmap) gtScaled.recycle()

        val iou = if (union > 0) intersection.toFloat() / union.toFloat() else 1.0f
        val dice = if (predCount + gtCount > 0) (2.0f * intersection) / (predCount + gtCount).toFloat() else 1.0f

        return SegmentationMetrics(
            sampleName = sampleName,
            iou = iou,
            dice = dice,
            predictedMaskArea = predCount,
            gtMaskArea = gtCount
        )
    }
}
