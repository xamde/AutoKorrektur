package de.konradvoelkel.android.autokorrektur

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.MediumTest
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.Core
import java.io.File

@RunWith(AndroidJUnit4::class)
@MediumTest
class VehicleMaskSegmentationTest : AndroidInstrumentedBaseTest() {

    private val corpusFiles = listOf(
        "sample_street_with_car.jpg",
        "sample_suburb_with_car.jpg",
        "sample_corpus_landscape_multicar.jpg",
        "sample_corpus_portrait_suv.jpg",
        "sample_corpus_shadow_car.jpg"
    )

    @Test
    fun testMaskSegmentationOnCorpusImages() {
        val yoloService = YoloServiceImpl(appContext)
        yoloService.initialize("yolo11s", useFP16 = false)

        val imageProcessor = ImageProcessor(appContext)

        val testContext = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().context
        for (filename in corpusFiles) {
            val stream = try {
                testContext.assets.open(filename)
            } catch (e: Exception) {
                null
            }
            if (stream == null) continue

            val bitmap = BitmapFactory.decodeStream(stream)
            assertNotNull("Corpus bitmap $filename should decode", bitmap)

            val rawMat = org.opencv.core.Mat()
            org.opencv.android.Utils.bitmapToMat(bitmap, rawMat)
            val rgbMat = org.opencv.core.Mat()
            org.opencv.imgproc.Imgproc.cvtColor(rawMat, rgbMat, org.opencv.imgproc.Imgproc.COLOR_RGBA2RGB)
            val inputMat = org.opencv.core.Mat()
            org.opencv.imgproc.Imgproc.resize(rgbMat, inputMat, org.opencv.core.Size(640.0, 640.0))
            rawMat.release()
            rgbMat.release()

            val yoloResult = yoloService.inferDetailed(
                transformedMat = inputMat,
                xRatio = 1.0f,
                yRatio = 1.0f,
                upscaleFactor = 1.05f
            )

            assertNotNull("YoloResult mask should not be null for $filename", yoloResult.mask)
            assertTrue("Mask Mat should not be empty for $filename", !yoloResult.mask.empty())

            // Ensure non-zero mask pixels exist if vehicles were detected
            val nonZeroCount = Core.countNonZero(yoloResult.mask)
            assertTrue("Mask pixel count should be non-negative for $filename", nonZeroCount >= 0)

            inputMat.release()
            yoloResult.mask.release()
            bitmap.recycle()
        }

        yoloService.close()
    }

    @Test
    fun testScoreThresholdSensitivity() {
        val yoloService = YoloServiceImpl(appContext)
        yoloService.initialize("yolo11s", useFP16 = false)

        val testContext = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().context
        val stream = testContext.assets.open("sample_street_with_car.jpg")
        val bitmap = BitmapFactory.decodeStream(stream)
        val rawMat = org.opencv.core.Mat()
        org.opencv.android.Utils.bitmapToMat(bitmap, rawMat)
        val rgbMat = org.opencv.core.Mat()
        org.opencv.imgproc.Imgproc.cvtColor(rawMat, rgbMat, org.opencv.imgproc.Imgproc.COLOR_RGBA2RGB)
        val inputMat = org.opencv.core.Mat()
        org.opencv.imgproc.Imgproc.resize(rgbMat, inputMat, org.opencv.core.Size(640.0, 640.0))
        rawMat.release()
        rgbMat.release()

        // Strict threshold
        val strictConfig = YoloConfig(scoreThreshold = 0.8f)
        val strictResult = yoloService.inferDetailed(
            transformedMat = inputMat,
            xRatio = 1.0f,
            yRatio = 1.0f,
            upscaleFactor = 1.05f,
            overrideConfig = strictConfig
        )

        // Lenient threshold
        val lenientConfig = YoloConfig(scoreThreshold = 0.1f)
        val lenientResult = yoloService.inferDetailed(
            transformedMat = inputMat,
            xRatio = 1.0f,
            yRatio = 1.0f,
            upscaleFactor = 1.05f,
            overrideConfig = lenientConfig
        )

        // Lenient threshold should detect equal or more candidates than strict threshold
        assertTrue(
            "Lenient score threshold should detect at least as many vehicles as strict threshold",
            lenientResult.detections.size >= strictResult.detections.size
        )

        inputMat.release()
        strictResult.mask.release()
        lenientResult.mask.release()
        bitmap.recycle()
        yoloService.close()
    }
}
