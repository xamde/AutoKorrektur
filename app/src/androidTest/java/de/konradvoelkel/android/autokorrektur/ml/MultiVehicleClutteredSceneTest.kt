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
 * Validates vehicle segmentation in cluttered scenes containing multiple parked cars and foreground/background elements.
 */
@RunWith(AndroidJUnit4::class)
@MediumTest
class MultiVehicleClutteredSceneTest : AndroidInstrumentedBaseTest() {

    private lateinit var yoloService: YoloServiceImpl

    @Before
    fun setUp() = kotlinx.coroutines.runBlocking {
        assertTrue("OpenCV initialization failed", OpenCVLoader.initLocal())
        yoloService = YoloServiceImpl(YoloTFLiteEngine(appContext))
        yoloService.initialize()
    }

    @Test
    fun testMultiCarScene_detectsMultipleVehiclesAndSeparatesBackground() =
        kotlinx.coroutines.runBlocking {
            val testAssets = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().context.assets
            val stream: InputStream = testAssets.open("triples/triple_41_with_car.png")
            val bitmap = BitmapFactory.decodeStream(stream)
            stream.close()

            assertNotNull("Multicar image should decode", bitmap)

            val tempFile = java.io.File(appContext.cacheDir, "test_multicar.jpg")
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
                overrideConfig = YoloConfig(scoreThreshold = 0.25f, iouThreshold = 0.40f)
            )

            processedImage.release()
            tempFile.delete()
            bitmap.recycle()

            AppLogger.info("Multi-car detections count: ${yoloResult.detections.size}")
            for ((idx, det) in yoloResult.detections.withIndex()) {
                AppLogger.info("  Vehicle #$idx: class=${det.classId}, conf=${det.confidence}, bbox=(${det.x}, ${det.y}, ${det.width}, ${det.height})")
            }

            // 1. In a multi-car scene, should detect at least 1-2 candidate vehicles
            assertTrue("Should detect vehicles in multi-car scene", yoloResult.detections.isNotEmpty())

            // 2. Output mask must exist and have non-trivial car area
            assertNotNull("Mask Mat must exist", yoloResult.mask)
            val totalPixels = yoloResult.mask.rows() * yoloResult.mask.cols()
            val backgroundPixels = Core.countNonZero(yoloResult.mask)
            val carPixels = totalPixels - backgroundPixels

            AppLogger.info("Multicar total: $totalPixels, car masked: $carPixels, bg: $backgroundPixels")
            assertTrue("Car mask area must be positive", carPixels > 100)
            assertTrue("Background must be preserved (> 20% background area)", backgroundPixels > totalPixels * 0.20)

            yoloResult.mask.release()
            yoloService.close()
        }
}
