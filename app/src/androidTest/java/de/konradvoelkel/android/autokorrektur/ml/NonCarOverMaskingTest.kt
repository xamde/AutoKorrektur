package de.konradvoelkel.android.autokorrektur.ml

import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
import org.opencv.core.CvType
import java.io.InputStream

/**
 * Instrumented Test Suite for Non-Car Over-Masking Prevention.
 *
 * Verifies that background regions (sky, trees, building facades, signposts, tables)
 * are NOT falsely masked by YOLO segmentation.
 */
@RunWith(AndroidJUnit4::class)
@MediumTest
class NonCarOverMaskingTest : AndroidInstrumentedBaseTest() {

    private lateinit var yoloService: YoloServiceImpl
    private lateinit var imageProcessor: ImageProcessor

    @Before
    fun setUp() = kotlinx.coroutines.runBlocking {
        assertTrue("OpenCV initialization failed", OpenCVLoader.initLocal())
        yoloService = YoloServiceImpl(YoloTFLiteEngine(appContext))
        yoloService.initialize()
        imageProcessor = ImageProcessor(appContext)
    }

    @Test
    fun testNonCarBackgroundRegions_areNotFalselyMasked() = kotlinx.coroutines.runBlocking {
        val testImages = listOf(
            "sample_street_with_car.jpg",
            "sample_suburb_with_car.jpg",
            "sample_corpus_landscape_multicar.jpg",
            "photo_with_car_1.png"
        )

        val testAssets = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().context.assets

        for (imgName in testImages) {
            val imgInputStream: InputStream = testAssets.open(imgName)
            val inputBitmap = BitmapFactory.decodeStream(imgInputStream)
            imgInputStream.close()

            val tempFile = java.io.File(appContext.cacheDir, "overmask_$imgName")
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
                overrideConfig = YoloConfig()
            )

            val maskMat = yoloResult.mask
            val width = maskMat.cols()
            val height = maskMat.rows()

            val maskBytes = ByteArray(width * height)
            val mask8u = org.opencv.core.Mat()
            maskMat.convertTo(mask8u, CvType.CV_8UC1)
            mask8u.get(0, 0, maskBytes)
            mask8u.release()

            // Analyze upper 15% of the image (pure sky / trees / building roofs above any car)
            val topRegionHeight = (height * 0.15f).toInt()
            var totalTopPixels = 0
            var falselyMaskedTopPixels = 0

            for (y in 0 until topRegionHeight) {
                for (x in 0 until width) {
                    totalTopPixels++
                    val maskVal = maskBytes[y * width + x].toInt() and 0xFF
                    // 0 = masked, 255 = unmasked
                    if (maskVal < 128) {
                        falselyMaskedTopPixels++
                    }
                }
            }

            val falseMaskRatio = falselyMaskedTopPixels.toFloat() / totalTopPixels.toFloat()
            AppLogger.info("OVER-MASKING [$imgName] -> Top region false mask ratio: ${"%.2f".format(falseMaskRatio * 100)}%")

            processedImage.release()
            tempFile.delete()
            inputBitmap.recycle()

            // Rigorous Assertion: Non-car top background regions must have < 10% false masking
            assertTrue(
                "Background top region in [$imgName] was falsely masked! Ratio: ${"%.2f".format(falseMaskRatio * 100)}% (expected < 10%)",
                falseMaskRatio < 0.10f
            )
        }
    }
}
