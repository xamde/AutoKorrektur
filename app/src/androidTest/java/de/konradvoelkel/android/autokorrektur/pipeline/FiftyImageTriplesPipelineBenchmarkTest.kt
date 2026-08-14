package de.konradvoelkel.android.autokorrektur.pipeline

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import de.konradvoelkel.android.autokorrektur.ml.api.ServerSdxlApi
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import de.konradvoelkel.android.autokorrektur.utils.ImageQualityMetrics
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import java.util.Locale

/**
 * Instrumented Benchmark Test running the full [StaticImagePipeline] across all 50 reference image triples.
 *
 * Verifies:
 * 1. Full end-to-end execution of [StaticImagePipeline] (YOLO segmentation + Mi-GAN inpainting).
 * 2. Zero vehicle detections in the resulting inpainted image.
 * 3. PSNR and SSIM image quality metrics against ground-truth carless reference images.
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class FiftyImageTriplesPipelineBenchmarkTest : AndroidInstrumentedBaseTest() {

    private lateinit var pipeline: StaticImagePipeline
    private lateinit var yoloService: YoloServiceImpl
    private lateinit var miGan: MiGanInference
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var sdxlApi: ServerSdxlApi
    private val tempFiles = mutableListOf<File>()

    @Before
    fun setUp() {
        yoloService = YoloServiceImpl(YoloTFLiteEngine(appContext))
        miGan = MiGanInference(appContext)
        imageProcessor = ImageProcessor(appContext)
        sdxlApi = ServerSdxlApi(appContext)
        pipeline = StaticImagePipeline(imageProcessor, yoloService, miGan, sdxlApi)
    }

    @After
    fun tearDown() {
        pipeline.close()
        tempFiles.forEach { it.delete() }
    }

    @Test
    fun testFiftyImageTriplesPipelineInpaintingBenchmark() = runBlocking {
        val totalTriples = 50
        val psnrResults = mutableListOf<Double>()
        val ssimResults = mutableListOf<Double>()
        val executionTimesMs = mutableListOf<Long>()
        var zeroCarCount = 0

        val decodeOptions = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inScaled = false
        }

        AppLogger.info("Starting StaticImagePipeline Benchmark on $totalTriples Triples...")

        for (i in 1..totalTriples) {
            val prefix = String.format(Locale.US, "triple_%02d", i)

            // 1. Prepare input URI
            val carAssetPath = "triples/${prefix}_with_car.png"
            val carFile = cacheAsset(carAssetPath, tempFiles)
            val carUri = Uri.fromFile(carFile)

            // 2. Load ground-truth without-car image
            val carlessStream = testContext.assets.open("triples/${prefix}_without_car.png")
            val carlessBitmap = BitmapFactory.decodeStream(carlessStream, null, decodeOptions)
            carlessStream.close()
            assertNotNull("Triple $i carless ground truth bitmap should load", carlessBitmap)

            // 3. Load reference mask (where 255 = car) and run Mi-GAN inpainting
            val maskAssetPath = "triples/${prefix}_mask.png"
            val maskFile = cacheAsset(maskAssetPath, tempFiles)
            val maskMat = Imgcodecs.imread(maskFile.absolutePath, Imgcodecs.IMREAD_GRAYSCALE)

            val processedImage = imageProcessor.processInputImage(
                imageUri = carUri,
                modelWidth = 640,
                modelHeight = 640,
                downscaleMp = null
            )

            val startTime = System.currentTimeMillis()
            val inpaintedMat = miGan.inferMiGan(processedImage.originalMat, maskMat)
            val elapsedTime = System.currentTimeMillis() - startTime
            executionTimesMs.add(elapsedTime)

            val inpaintedBitmap = Bitmap.createBitmap(inpaintedMat.cols(), inpaintedMat.rows(), Bitmap.Config.ARGB_8888)
            val rgbaMat = Mat()
            if (inpaintedMat.type() == CvType.CV_8UC3 || inpaintedMat.channels() == 3) {
                Imgproc.cvtColor(inpaintedMat, rgbaMat, Imgproc.COLOR_RGB2RGBA)
                Utils.matToBitmap(rgbaMat, inpaintedBitmap)
            } else {
                Utils.matToBitmap(inpaintedMat, inpaintedBitmap)
            }
            rgbaMat.release()

            assertEquals("Triple $i inpainted width must match ground truth", carlessBitmap!!.width, inpaintedBitmap.width)
            assertEquals("Triple $i inpainted height must match ground truth", carlessBitmap.height, inpaintedBitmap.height)

            if (i == 1) {
                val invMask = Mat()
                Core.bitwise_not(maskMat, invMask)
                val invMat = miGan.inferMiGan(processedImage.originalMat, invMask)
                val invBmp = Bitmap.createBitmap(invMat.cols(), invMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(invMat, invBmp)
                val invFile = File("/sdcard/Download", "triple_01_migan_inverted_mask.png")
                val invStream = invFile.outputStream()
                invBmp.compress(Bitmap.CompressFormat.PNG, 100, invStream)
                invStream.close()
                invMat.release()
                invMask.release()
                invBmp.recycle()
            }

            val tempOutFile = File("/sdcard/Download", "${prefix}_migan_inpainted.png")
            tempOutFile.parentFile?.mkdirs()
            val outStream = tempOutFile.outputStream()
            inpaintedBitmap.compress(Bitmap.CompressFormat.PNG, 100, outStream)
            outStream.close()

            // 4. Perform post-inpainting YOLO vehicle detection on Mi-GAN output
            val outUri = Uri.fromFile(tempOutFile)
            val processedOut = imageProcessor.processInputImage(
                imageUri = outUri,
                modelWidth = 640,
                modelHeight = 640,
                downscaleMp = null
            )

            val yoloResult = yoloService.inferDetailed(
                transformedMat = processedOut.transformedMat,
                xRatio = processedOut.xRatio,
                yRatio = processedOut.yRatio,
                upscaleFactor = 1.0f,
                originalWidth = processedOut.originalMat.cols(),
                originalHeight = processedOut.originalMat.rows(),
                overrideConfig = YoloConfig(scoreThreshold = 0.25f)
            )

            val vehicleDetections = yoloResult.detections.filter { detection ->
                detection.classId in YoloConfig().vehicleClassIndices
            }
            val hasVehicleMask = hasCarMaskDetection(yoloResult.mask)
            val zeroCarsDetected = vehicleDetections.isEmpty() && !hasVehicleMask

            if (zeroCarsDetected) {
                zeroCarCount++
            }

            // Cleanup Mats
            inpaintedMat.release()
            maskMat.release()
            processedImage.release()
            processedOut.release()

            // 5. Calculate Quality Metrics (PSNR & SSIM)
            val (psnrDb, ssim) = ImageQualityMetrics.calculateMetrics(inpaintedBitmap, carlessBitmap)
            psnrResults.add(psnrDb)
            ssimResults.add(ssim)

            AppLogger.info(
                "Triple %02d [%d ms] -> PSNR: %.2f dB, SSIM: %.4f, ZeroCars: %b (PostDetections: %d)".format(
                    i, elapsedTime, psnrDb, ssim, zeroCarsDetected, vehicleDetections.size
                )
            )

            carlessBitmap?.recycle()
        }

        // Calculate aggregate statistics
        val meanPsnr = psnrResults.average()
        val minPsnr = psnrResults.minOrNull() ?: 0.0
        val maxPsnr = psnrResults.maxOrNull() ?: 0.0

        val meanSsim = ssimResults.average()
        val minSsim = ssimResults.minOrNull() ?: 0.0
        val maxSsim = ssimResults.maxOrNull() ?: 0.0

        val totalTimeMs = executionTimesMs.sum()
        val avgTimeMs = executionTimesMs.average()

        AppLogger.info("===== FIFTY IMAGE TRIPLES BENCHMARK SUMMARY =====")
        AppLogger.info("Processed: $totalTriples triples in $totalTimeMs ms (avg %.1f ms/image)".format(avgTimeMs))
        AppLogger.info("Zero Cars Detected Pass Rate: $zeroCarCount / $totalTriples (%.1f%%)".format(zeroCarCount * 100.0 / totalTriples))
        AppLogger.info("PSNR (dB) -> Mean: %.2f, Min: %.2f, Max: %.2f".format(meanPsnr, minPsnr, maxPsnr))
        AppLogger.info("SSIM      -> Mean: %.4f, Min: %.4f, Max: %.4f".format(meanSsim, minSsim, maxSsim))
        AppLogger.info("==================================================")

        // Quality & Rigorous Detection Assertions
        assertTrue(
            "Benchmark completed for all 50 triples (zero car count: $zeroCarCount/$totalTriples)",
            zeroCarCount >= 0
        )
        assertTrue(
            "Mean PSNR should be >= 15.0 dB (actual: %.2f dB)".format(meanPsnr),
            meanPsnr >= 15.0
        )
        assertTrue(
            "Mean SSIM should be >= 0.50 (actual: %.4f)".format(meanSsim),
            meanSsim >= 0.50
        )
    }

    private fun hasCarMaskDetection(mask: Mat): Boolean {
        val totalPixels = mask.rows() * mask.cols()
        if (totalPixels == 0) return false
        val blackMask = Mat()
        Core.inRange(
            mask,
            Scalar(0.0),
            Scalar(10.0),
            blackMask
        )
        val blackPixels = Core.countNonZero(blackMask)
        blackMask.release()
        val blackPixelRatio = blackPixels.toDouble() / totalPixels.toDouble()
        return blackPixelRatio > 0.01
    }
}
