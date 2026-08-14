package de.konradvoelkel.android.autokorrektur.ml

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.MediumTest
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.InputStream

/**
 * Validates segmentation and mask edge adherence on real-world scenes with harsh directional asphalt cast shadows.
 */
@RunWith(AndroidJUnit4::class)
@MediumTest
class VehicleShadowSegmentationTest : AndroidInstrumentedBaseTest() {

    private lateinit var yoloService: YoloServiceImpl

    @Before
    fun setUp() = kotlinx.coroutines.runBlocking {
        assertTrue("OpenCV initialization failed", OpenCVLoader.initLocal())
        yoloService = YoloServiceImpl(YoloTFLiteEngine(appContext))
        yoloService.initialize()
    }

    @Test
    fun testVehicleWithCastShadow_detectsVehicleAndProducesRefinedMask() =
        kotlinx.coroutines.runBlocking {
            val testAssets = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().context.assets
            val stream: InputStream = testAssets.open("triples/triple_21_with_car.png")
            val bitmap = BitmapFactory.decodeStream(stream)
            stream.close()

            assertNotNull("Shadow test image should decode", bitmap)

            val tempFile = java.io.File(appContext.cacheDir, "test_shadow_car.jpg")
            java.io.FileOutputStream(tempFile).use { out ->
                bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 95, out)
            }
            val uri = android.net.Uri.fromFile(tempFile)
            val imageProcessor = ImageProcessor(appContext)
            val processedImage = imageProcessor.processInputImage(uri, modelWidth = 640, modelHeight = 640)

            val yoloResult = yoloService.inferDetailed(
                transformedMat = processedImage.transformedMat,
                xRatio = processedImage.xRatio,
                yRatio = processedImage.yRatio,
                upscaleFactor = 1.05f,
                originalWidth = processedImage.originalMat.cols(),
                originalHeight = processedImage.originalMat.rows(),
                overrideConfig = YoloConfig(scoreThreshold = 0.25f, iouThreshold = 0.45f)
            )

            processedImage.release()
            tempFile.delete()
            bitmap.recycle()

            // 1. Must detect at least 1 vehicle in the scene
            assertTrue("Should detect vehicle with shadow", yoloResult.detections.isNotEmpty())
            val topDetection = yoloResult.detections.maxByOrNull { it.confidence }!!
            AppLogger.info("Detected shadow car with confidence: ${topDetection.confidence} at (${topDetection.x}, ${topDetection.y}, ${topDetection.width}, ${topDetection.height})")
            assertTrue("Detection confidence should be >= 0.35", topDetection.confidence >= 0.35f)

            // 2. Output mask must not be empty
            assertNotNull("Mask Mat must exist", yoloResult.mask)
            assertTrue("Mask must not be empty", !yoloResult.mask.empty())

            // 3. Count masked car pixels (0 in pipeline convention)
            val totalPixels = yoloResult.mask.rows() * yoloResult.mask.cols()
            val backgroundPixels = Core.countNonZero(yoloResult.mask)
            val carPixels = totalPixels - backgroundPixels

            AppLogger.info("Shadow scene total pixels: $totalPixels, car pixels: $carPixels, background: $backgroundPixels")
            assertTrue("Car mask must have positive area", carPixels > 500)
            assertTrue("Car mask must not cover whole image", carPixels < totalPixels * 0.85)

            yoloResult.mask.release()
            yoloService.close()
        }
}
