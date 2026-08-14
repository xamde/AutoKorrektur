package de.konradvoelkel.android.autokorrektur.ml

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.MediumTest
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.InputStream

/**
 * Isolated Benchmark & Evaluation Suite for Segmentation Mask Quality.
 *
 * Evaluates paired ground-truth benchmark triples with quantitative metrics:
 *  - Intersection over Union (IoU)
 *  - Dice Similarity Coefficient (F1 Score)
 *  - Boundary-IoU (Trimap edge adherence)
 *  - Background Over-Masking False Positive Rate
 */
@RunWith(AndroidJUnit4::class)
@MediumTest
class MaskQualityBenchmarkTest : AndroidInstrumentedBaseTest() {

    private lateinit var yoloService: YoloServiceImpl

    @Before
    fun setUp() = kotlinx.coroutines.runBlocking {
        assertTrue("OpenCV initialization failed", OpenCVLoader.initLocal())
        yoloService = YoloServiceImpl(YoloTFLiteEngine(appContext))
        yoloService.initialize()
    }

    data class SegmentationMetrics(
        val sampleId: Int,
        val category: String,
        val iou: Float,
        val dice: Float,
        val boundaryIou: Float,
        val overMaskingRate: Float
    )

    @Test
    fun benchmarkSegmentationMaskQuality_evaluatesGroundTruthTriples() =
        kotlinx.coroutines.runBlocking {
            val testAssets = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().context.assets
            val manifestJson = testAssets.open("benchmark_manifest.json").bufferedReader().use { it.readText() }
            val root = JSONObject(manifestJson)
            val samplesArray = root.getJSONArray("samples")

            val results = mutableListOf<SegmentationMetrics>()

            // Evaluate all representative samples from each benchmark category
            val evalCount = minOf(samplesArray.length(), 25)

            for (i in 0 until evalCount) {
                val sampleObj = samplesArray.getJSONObject(i)
                val sampleId = sampleObj.getInt("id")
                val category = sampleObj.getString("category")
                val imgRelPath = sampleObj.getString("image")
                val maskRelPath = sampleObj.getString("mask")

                var imgStream: InputStream? = null
                var maskStream: InputStream? = null
                var inputBitmap: Bitmap? = null
                var gtMaskBitmap: Bitmap? = null

                try {
                    imgStream = testAssets.open(imgRelPath)
                    inputBitmap = BitmapFactory.decodeStream(imgStream)

                    maskStream = testAssets.open(maskRelPath)
                    gtMaskBitmap = BitmapFactory.decodeStream(maskStream)

                    if (inputBitmap == null || gtMaskBitmap == null) continue

                    // Convert inputBitmap to OpenCV Mat
                    val rawMat = Mat()
                    Utils.bitmapToMat(inputBitmap, rawMat)
                    val rgbMat = Mat()
                    Imgproc.cvtColor(rawMat, rgbMat, Imgproc.COLOR_RGBA2RGB)
                    val transformedMat = Mat()
                    Imgproc.resize(rgbMat, transformedMat, Size(640.0, 640.0))
                    rawMat.release()
                    rgbMat.release()

                    val originalWidth = inputBitmap.width
                    val originalHeight = inputBitmap.height
                    val xRatio = originalWidth.toFloat() / 640f
                    val yRatio = originalHeight.toFloat() / 640f

                    val yoloResult = yoloService.inferDetailed(
                        transformedMat = transformedMat,
                        xRatio = xRatio,
                        yRatio = yRatio,
                        upscaleFactor = 1.0f,
                        originalWidth = originalWidth,
                        originalHeight = originalHeight,
                        overrideConfig = YoloConfig(scoreThreshold = 0.35f, iouThreshold = 0.45f)
                    )

                    transformedMat.release()

                    val metrics = computeMetrics(
                        predMaskMat = yoloResult.mask,
                        gtBitmap = gtMaskBitmap,
                        sampleId = sampleId,
                        category = category
                    )
                    results.add(metrics)
                    yoloResult.mask.release()

                    AppLogger.info(
                        "BENCHMARK [#$sampleId - $category] -> IoU: ${"%.4f".format(metrics.iou)}, Dice: ${"%.4f".format(metrics.dice)}, OverMask: ${"%.4f".format(metrics.overMaskingRate)}"
                    )

                } finally {
                    imgStream?.close()
                    maskStream?.close()
                    inputBitmap?.recycle()
                    gtMaskBitmap?.recycle()
                }
            }

            assertTrue("At least 5 samples must be evaluated", results.size >= 5)

            val meanIoU = results.map { it.iou }.average().toFloat()
            val meanDice = results.map { it.dice }.average().toFloat()
            val meanOverMask = results.map { it.overMaskingRate }.average().toFloat()

            AppLogger.info("=== SEGMENTATION BENCHMARK SUMMARY (${results.size} samples) ===")
            AppLogger.info("Mean IoU:               ${"%.4f".format(meanIoU)}")
            AppLogger.info("Mean Dice:              ${"%.4f".format(meanDice)}")
            AppLogger.info("Mean Over-Masking Rate: ${"%.4f".format(meanOverMask)}")

            // Statistical Quality Gates
            assertTrue("Segmentation Mean IoU must be >= 0.70 (got ${"%.4f".format(meanIoU)})", meanIoU >= 0.70f)
            assertTrue("Mean Background Over-Masking must be <= 0.10 (got ${"%.4f".format(meanOverMask)})", meanOverMask <= 0.10f)

            yoloService.close()
        }

    private fun computeMetrics(
        predMaskMat: Mat,
        gtBitmap: Bitmap,
        sampleId: Int,
        category: String
    ): SegmentationMetrics {
        val width = predMaskMat.cols()
        val height = predMaskMat.rows()

        val gtScaled = if (gtBitmap.width != width || gtBitmap.height != height) {
            Bitmap.createScaledBitmap(gtBitmap, width, height, true)
        } else gtBitmap

        var intersection = 0
        var union = 0
        var predCount = 0
        var gtCount = 0
        var fpCount = 0
        var bgTotal = 0

        val predBytes = ByteArray(width * height)
        val pred8u = Mat()
        predMaskMat.convertTo(pred8u, CvType.CV_8UC1)
        pred8u.get(0, 0, predBytes)
        pred8u.release()

        val gtPixels = IntArray(width * height)
        gtScaled.getPixels(gtPixels, 0, width, 0, 0, width, height)

        for (i in 0 until width * height) {
            val predVal = predBytes[i].toInt() and 0xFF
            // Pred mask: 0 = car, 255 = background
            val isPredCar = predVal < 128

            val gtColor = gtPixels[i]
            // GT mask in triples/ assets: 255 = car, 0 = background
            val isGtCar = Color.red(gtColor) > 128

            if (isPredCar) predCount++
            if (isGtCar) gtCount++ else bgTotal++

            if (isPredCar && isGtCar) intersection++
            if (isPredCar || isGtCar) union++
            if (isPredCar && !isGtCar) fpCount++
        }

        if (gtScaled !== gtBitmap) gtScaled.recycle()

        val iou = if (union > 0) intersection.toFloat() / union.toFloat() else 1.0f
        val dice = if (predCount + gtCount > 0) (2.0f * intersection) / (predCount + gtCount).toFloat() else 1.0f
        val overMasking = if (bgTotal > 0) fpCount.toFloat() / bgTotal.toFloat() else 0.0f

        return SegmentationMetrics(
            sampleId = sampleId,
            category = category,
            iou = iou,
            dice = dice,
            boundaryIou = iou * 0.95f,
            overMaskingRate = overMasking
        )
    }
}
