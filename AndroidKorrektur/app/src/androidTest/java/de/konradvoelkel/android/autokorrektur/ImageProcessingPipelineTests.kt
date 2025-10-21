package de.konradvoelkel.android.autokorrektur

import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.YoloInferenceTFLite
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

@RunWith(AndroidJUnit4::class)
class ImageProcessingPipelineTests {

    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    private lateinit var yoloInference: YoloInferenceTFLite
    private lateinit var imageProcessor: ImageProcessor
    private val tempFiles = mutableListOf<File>()

    @Before
    fun setUp() {
        TestUtils.initOpenCV()
        yoloInference = YoloInferenceTFLite(appContext)
        yoloInference.initialize("yolo11s")
        imageProcessor = ImageProcessor(appContext)
    }

    @After
    fun tearDown() {
        yoloInference.close()
        tempFiles.forEach { it.delete() }
    }

    private fun copyAssetToCache(assetFileName: String): File {
        val file = TestUtils.copyAssetToCache(appContext, assetFileName)
        tempFiles.add(file)
        return file
    }

    private fun hasCarDetection(mask: Mat): Boolean {
        val totalPixels = mask.rows() * mask.cols()
        if (totalPixels == 0) return false
        val blackMask = Mat()
        Core.inRange(mask, Scalar(0.0), Scalar(10.0), blackMask)
        val blackPixels = Core.countNonZero(blackMask)
        blackMask.release()
        val blackPixelRatio = blackPixels.toDouble() / totalPixels.toDouble()
        println(
            "[DEBUG_LOG] hasCarDetection: Black pixels: $blackPixels / $totalPixels (${
                String.format(
                    "%.4f",
                    blackPixelRatio * 100
                )
            }%)"
        )
        return blackPixelRatio > 0.01
    }

    @Test
    fun testImageSelectionSimulation() {
        val mockupImageUri = TestUtils.copyAssetToCache(appContext, "image_1_with_car_640x640.png")
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
        val tempFile = copyAssetToCache("image_1_with_car_640x640.png")
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

        // for debugging, how does the no-car-but-still-some-mask look like?
        val outputFileName = "debug_mask_car.png"
        val outputFile = File(appContext.getExternalFilesDir(null), outputFileName)
        Imgcodecs.imwrite(outputFile.absolutePath, resultMask)
        println("[DEBUG_LOG] Saved debug mask to: ${outputFile.absolutePath}")

        assertTrue(
            "Car should be detected in image_1_with_car_640x640.png",
            hasCarDetection(resultMask)
        )
        resultMask.release()
    }

    @Test
    fun testNoCarDetectionOnResultImage() {
        val tempFile = copyAssetToCache("image_1_without_car_640x640.png")
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

        // for debugging, how does the no-car-but-still-some-mask look like?
        val outputFileName = "debug_mask_no_car.png"
        val outputFile = File(appContext.getExternalFilesDir(null), outputFileName)
        Imgcodecs.imwrite(outputFile.absolutePath, resultMask)
        println("[DEBUG_LOG] Saved debug mask to: ${outputFile.absolutePath}")

        assertFalse(
            "Car should NOT be detected in image_1_without_car_640x640.png",
            hasCarDetection(resultMask)
        )
        resultMask.release()
    }

    @Test
    fun testYoloMaskCreationProperties() {
        val tempFile = copyAssetToCache("image_1_with_car_640x640.png")
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
        val tempFile = copyAssetToCache("image_1_with_car_640x640.png")
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

    @Test
    fun testCarMaskMatchesReference() {
        val photoFile = copyAssetToCache("photo_with_car_1.png")
        val photoUri = Uri.fromFile(photoFile)

        val processedImage = imageProcessor.processInputImage(
            imageUri = photoUri,
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

        val refMaskFile = copyAssetToCache("photo_with_car_1_mask.png")
        val referenceMask = Imgcodecs.imread(refMaskFile.absolutePath, Imgcodecs.IMREAD_GRAYSCALE)

        assertNotNull("Reference mask should load", referenceMask)
        assertTrue("Reference mask should not be empty", !referenceMask.empty())

        assertEquals(
            "Mask height must match original image height",
            processedImage.originalMat.rows(), resultMask.rows()
        )
        assertEquals(
            "Mask width must match original image width",
            processedImage.originalMat.cols(), resultMask.cols()
        )
        assertEquals(
            "Reference mask height must match result height",
            resultMask.rows(), referenceMask.rows()
        )
        assertEquals(
            "Reference mask width must match result width",
            resultMask.cols(), referenceMask.cols()
        )

        // Binarize: treat near-black as car (255), others 0
        val binResult = Mat()
        val binReference = Mat()
        Core.inRange(resultMask, Scalar(0.0), Scalar(10.0), binResult)
        Core.inRange(referenceMask, Scalar(0.0), Scalar(10.0), binReference)

        // Compute XOR to find mismatches
        val diff = Mat()
        Core.bitwise_xor(binResult, binReference, diff)
        val mismatches = Core.countNonZero(diff)
        val totalPixels = diff.rows() * diff.cols()
        val agreement = 1.0 - (mismatches.toDouble() / totalPixels.toDouble())

        println(
            "[DEBUG_LOG] Mask agreement: ${String.format("%.4f", agreement * 100)}% (mismatches=$mismatches / total=$totalPixels)"
        )

        // Optionally save debug images
        val outDebugDir = appContext.getExternalFilesDir(null)
        if (outDebugDir != null) {
            val resultPath = File(outDebugDir, "debug_mask_result.png").absolutePath
            val referencePath = File(outDebugDir, "debug_mask_reference.png").absolutePath
            val diffPath = File(outDebugDir, "debug_mask_diff.png").absolutePath
            Imgcodecs.imwrite(resultPath, resultMask)
            Imgcodecs.imwrite(referencePath, referenceMask)
            Imgcodecs.imwrite(diffPath, diff)
            println("[DEBUG_LOG] Saved debug images to: ${outDebugDir.absolutePath}")
            println("[DEBUG_LOG] - result: $resultPath")
            println("[DEBUG_LOG] - reference: $referencePath")
            println("[DEBUG_LOG] - diff: $diffPath")
        }

        assertTrue("Generated mask should agree with reference mask >= 80%", agreement >= 0.80)

        // Cleanup
        diff.release()
        binResult.release()
        binReference.release()
        referenceMask.release()
        resultMask.release()
    }
}
