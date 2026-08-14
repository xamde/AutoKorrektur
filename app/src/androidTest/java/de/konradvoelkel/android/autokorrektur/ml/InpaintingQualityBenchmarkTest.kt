package de.konradvoelkel.android.autokorrektur.ml

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.MediumTest
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
import org.opencv.imgproc.Imgproc
import java.io.InputStream
import kotlin.math.log10
import kotlin.math.sqrt

/**
 * Benchmark Suite for Neural Inpainting Quality and Background Fidelity.
 *
 * Validates that:
 *  1. Inpainted results retain original background pixels with PSNR >= 40 dB outside the hole.
 *  2. MI-GAN inference completes successfully across diverse ground-truth scenes.
 */
@RunWith(AndroidJUnit4::class)
@MediumTest
class InpaintingQualityBenchmarkTest : AndroidInstrumentedBaseTest() {

    private lateinit var inpaintingEngine: MiGanInference

    @Before
    fun setUp() = kotlinx.coroutines.runBlocking {
        assertTrue("OpenCV initialization failed", OpenCVLoader.initLocal())
        inpaintingEngine = MiGanInference(appContext)
        inpaintingEngine.initialize()
    }

    data class InpaintMetrics(
        val sampleId: Int,
        val psnr: Double,
        val maxBackgroundDelta: Int
    )

    @Test
    fun benchmarkInpaintingFidelity_measuresPsnrAndBackgroundInvariance() =
        kotlinx.coroutines.runBlocking {
            val testAssets = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().context.assets
            val manifestJson = testAssets.open("benchmark_manifest.json").bufferedReader().use { it.readText() }
            val root = JSONObject(manifestJson)
            val samplesArray = root.getJSONArray("samples")

            val results = mutableListOf<InpaintMetrics>()
            val evalCount = minOf(samplesArray.length(), 10)

            for (i in 0 until evalCount) {
                val sampleObj = samplesArray.getJSONObject(i)
                val sampleId = sampleObj.getInt("id")
                val imgRelPath = sampleObj.getString("image")
                val maskRelPath = sampleObj.getString("mask")

                var imgStream: InputStream? = null
                var maskStream: InputStream? = null
                var inputBitmap: Bitmap? = null
                var maskBitmap: Bitmap? = null

                try {
                    imgStream = testAssets.open(imgRelPath)
                    inputBitmap = BitmapFactory.decodeStream(imgStream)

                    maskStream = testAssets.open(maskRelPath)
                    maskBitmap = BitmapFactory.decodeStream(maskStream)

                    if (inputBitmap == null || maskBitmap == null) continue

                    val origMat = Mat()
                    Utils.bitmapToMat(inputBitmap, origMat)
                    val rawMaskMat = Mat()
                    Utils.bitmapToMat(maskBitmap, rawMaskMat)
                    // Convert GT mask (where 255 is car) to pipeline convention (0 is car, 255 is background)
                    val maskMat = Mat()
                    org.opencv.core.Core.bitwise_not(rawMaskMat, maskMat)
                    rawMaskMat.release()

                    val outputMat = inpaintingEngine.inferMiGan(origMat, maskMat)

                    val metrics = computeInpaintMetrics(origMat, maskMat, outputMat, sampleId)
                    results.add(metrics)

                    AppLogger.info(
                        "INPAINT BENCHMARK [#$sampleId] -> PSNR: ${"%.2f".format(metrics.psnr)} dB, Max Bg Delta: ${metrics.maxBackgroundDelta}"
                    )

                    origMat.release()
                    maskMat.release()
                    outputMat.release()

                } finally {
                    imgStream?.close()
                    maskStream?.close()
                    inputBitmap?.recycle()
                    maskBitmap?.recycle()
                }
            }

            assertTrue("At least 3 inpaint benchmarks must run", results.size >= 3)
            val meanPsnr = results.map { it.psnr }.average()

            AppLogger.info("=== INPAINTING BENCHMARK SUMMARY ===")
            AppLogger.info("Mean PSNR outside hole: ${"%.2f".format(meanPsnr)} dB")

            // Strict quality check: PSNR outside the hole must be >= 40 dB
            assertTrue("Mean PSNR must be >= 40 dB (got ${"%.2f".format(meanPsnr)} dB)", meanPsnr >= 40.0)

            inpaintingEngine.close()
        }

    private fun computeInpaintMetrics(
        origMat: Mat,
        maskMat: Mat,
        outputMat: Mat,
        sampleId: Int
    ): InpaintMetrics {
        val width = origMat.cols()
        val height = origMat.rows()

        val origRgb = Mat()
        val outRgb = Mat()
        Imgproc.cvtColor(origMat, origRgb, Imgproc.COLOR_RGBA2RGB)
        if (outputMat.channels() == 4) {
            Imgproc.cvtColor(outputMat, outRgb, Imgproc.COLOR_RGBA2RGB)
        } else {
            outputMat.copyTo(outRgb)
        }

        val maskGray = Mat()
        val maskCode = if (maskMat.channels() == 4) Imgproc.COLOR_RGBA2GRAY else Imgproc.COLOR_RGB2GRAY
        Imgproc.cvtColor(maskMat, maskGray, maskCode)

        val origBytes = ByteArray(width * height * 3)
        val outBytes = ByteArray(width * height * 3)
        val maskBytes = ByteArray(width * height)

        origRgb.get(0, 0, origBytes)
        outRgb.get(0, 0, outBytes)
        maskGray.get(0, 0, maskBytes)

        var sumSqErr = 0.0
        var count = 0
        var maxDelta = 0

        for (i in 0 until width * height) {
            val maskVal = maskBytes[i].toInt() and 0xFF
            // Unmasked background pixel in standard mask is >= 128
            if (maskVal >= 128) {
                for (c in 0 until 3) {
                    val origVal = origBytes[i * 3 + c].toInt() and 0xFF
                    val outVal = outBytes[i * 3 + c].toInt() and 0xFF
                    val diff = origVal - outVal
                    val absDiff = kotlin.math.abs(diff)
                    if (absDiff > maxDelta) maxDelta = absDiff
                    sumSqErr += diff * diff
                    count++
                }
            }
        }

        origRgb.release()
        outRgb.release()
        maskGray.release()

        val mse = if (count > 0) sumSqErr / count else 0.0
        val psnr = if (mse == 0.0) 100.0 else 20.0 * log10(255.0 / sqrt(mse))

        return InpaintMetrics(
            sampleId = sampleId,
            psnr = psnr,
            maxBackgroundDelta = maxDelta
        )
    }
}
