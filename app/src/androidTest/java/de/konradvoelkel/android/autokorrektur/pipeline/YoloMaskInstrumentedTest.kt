package de.konradvoelkel.android.autokorrektur.pipeline

import android.content.pm.ApplicationInfo
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.MediumTest
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.YoloInferenceTFLite
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import de.konradvoelkel.android.autokorrektur.shared.OpenCvTestUtils
import de.konradvoelkel.android.autokorrektur.shared.PipelineTestFixtures
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import java.io.File

/**
 * YOLO mask related tests: presence/absence and basic properties.
 */
@RunWith(AndroidJUnit4::class)
@MediumTest
class YoloMaskInstrumentedTest : AndroidInstrumentedBaseTest() {

    private lateinit var yoloInference: YoloInferenceTFLite
    private lateinit var imageProcessor: ImageProcessor
    private val tempFiles = mutableListOf<File>()

    companion object {
        @org.junit.AfterClass
        @JvmStatic
        fun tearDownFixtures() {
            PipelineTestFixtures.closeAll()
        }
    }

    private fun isDebug(): Boolean {
        val debuggable = (appContext.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0
        return debuggable && OpenCvTestUtils.shouldWriteDebugArtifacts(appContext)
    }

    @Before
    fun setUp() {
        yoloInference = PipelineTestFixtures.yolo()
        imageProcessor = PipelineTestFixtures.imageProcessor()
    }

    @After
    fun tearDown() {
        tempFiles.forEach { it.delete() }
    }

    private fun hasCarDetection(mask: Mat): Boolean {
        val totalPixels = mask.rows() * mask.cols()
        if (totalPixels == 0) return false
        val blackMask = Mat()
        Core.inRange(mask, Scalar(0.0), Scalar(10.0), blackMask)
        val blackPixels = Core.countNonZero(blackMask)
        blackMask.release()
        val blackPixelRatio = blackPixels.toDouble() / totalPixels.toDouble()
        return blackPixelRatio > 0.01
    }

    @Test
    fun testImageSelectionSimulation() {
        val mockupImageUri = de.konradvoelkel.android.autokorrektur.shared.AndroidTestUtils
            .copyAssetToCache(appContext, "image_1_with_car_640x640.png")
            .let { Uri.fromFile(it) }
        assertNotNull("Mockup image URI should not be null", mockupImageUri)

        val processedImage = imageProcessor.processInputImage(
            imageUri = mockupImageUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = 2.0f
        )

        assertNotNull("Processed image should not be null", processedImage)
        assertEquals(640, processedImage.transformedBitmap.width)
        assertEquals(640, processedImage.transformedBitmap.height)
        assertTrue("X ratio should be positive", processedImage.xRatio > 0)
        assertTrue("Y ratio should be positive", processedImage.yRatio > 0)
    }

    @Test
    fun testCarDetectionOnExampleImage() {
        val tempFile = cacheAsset("image_1_with_car_640x640.png", tempFiles)
        val fileUri = Uri.fromFile(tempFile)

        val processedImage = imageProcessor.processInputImage(
            imageUri = fileUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )

        val resultMask = yoloInference.inferYolo(
            transformedMat = processedImage.transformedMat,
            xRatio = processedImage.xRatio,
            yRatio = processedImage.yRatio,
            upscaleFactor = 1.2f,
            downshiftFactor = 0.0f,
            originalWidth = processedImage.originalMat.cols(),
            originalHeight = processedImage.originalMat.rows()
        )

        if (isDebug()) {
            val outputFileName = "debug_mask_car.png"
            val outputFile = File(appContext.getExternalFilesDir(null), outputFileName)
            Imgcodecs.imwrite(outputFile.absolutePath, resultMask)
        }

        assertTrue(
            "Car should be detected in image_1_with_car_640x640.png",
            hasCarDetection(resultMask)
        )
        resultMask.release()
    }

    @Test
    fun testNoCarDetectionOnResultImage() {
        val tempFile = cacheAsset("image_1_without_car_640x640.png", tempFiles)
        val fileUri = Uri.fromFile(tempFile)

        val processedImage = imageProcessor.processInputImage(
            imageUri = fileUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )

        val resultMask = yoloInference.inferYolo(
            transformedMat = processedImage.transformedMat,
            xRatio = processedImage.xRatio,
            yRatio = processedImage.yRatio,
            upscaleFactor = 1.2f,
            downshiftFactor = 0.0f,
            originalWidth = processedImage.originalMat.cols(),
            originalHeight = processedImage.originalMat.rows()
        )

        if (isDebug()) {
            val outputFileName = "debug_mask_no_car.png"
            val outputFile = File(appContext.getExternalFilesDir(null), outputFileName)
            Imgcodecs.imwrite(outputFile.absolutePath, resultMask)
        }

        assertFalse(
            "Car should NOT be detected in image_1_without_car_640x640.png",
            hasCarDetection(resultMask)
        )
        resultMask.release()
    }

    @Test
    fun testYoloMaskCreationProperties() {
        val tempFile = cacheAsset("image_1_with_car_640x640.png", tempFiles)
        val fileUri = Uri.fromFile(tempFile)

        val processedImage = imageProcessor.processInputImage(
            imageUri = fileUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )

        val resultMask = yoloInference.inferYolo(
            transformedMat = processedImage.transformedMat,
            xRatio = processedImage.xRatio,
            yRatio = processedImage.yRatio,
            upscaleFactor = 1.2f,
            downshiftFactor = 0.0f,
            originalWidth = processedImage.originalMat.cols(),
            originalHeight = processedImage.originalMat.rows()
        )

        assertNotNull("Result mask should not be null", resultMask)
        assertTrue("Result mask should not be empty", !resultMask.empty())
        assertEquals(processedImage.originalMat.rows(), resultMask.rows())
        assertEquals(processedImage.originalMat.cols(), resultMask.cols())

        resultMask.release()
    }

    @Test
    fun testCarDetectionMaskIsSensible() {
        val tempFile = cacheAsset("image_1_with_car_640x640.png", tempFiles)
        val fileUri = Uri.fromFile(tempFile)

        val processedImage = imageProcessor.processInputImage(
            imageUri = fileUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )

        val resultMask = yoloInference.inferYolo(
            transformedMat = processedImage.transformedMat,
            xRatio = processedImage.xRatio,
            yRatio = processedImage.yRatio,
            upscaleFactor = 1.2f,
            downshiftFactor = 0.0f,
            originalWidth = processedImage.originalMat.cols(),
            originalHeight = processedImage.originalMat.rows()
        )

        val totalPixelsInMask = resultMask.rows() * resultMask.cols()
        val blackMask = Mat()
        Core.inRange(resultMask, Scalar(0.0), Scalar(10.0), blackMask)
        val detectedPixels = Core.countNonZero(blackMask)
        blackMask.release()

        assertTrue("A car should be detected", detectedPixels > 0)

        val detectedRatio = detectedPixels.toDouble() / totalPixelsInMask.toDouble()
        assertTrue("Detected mask should be smaller than the whole image", detectedRatio < 0.95)
        assertTrue("Detected mask should be of a sensible size", detectedRatio > 0.001)

        resultMask.release()
    }
}
