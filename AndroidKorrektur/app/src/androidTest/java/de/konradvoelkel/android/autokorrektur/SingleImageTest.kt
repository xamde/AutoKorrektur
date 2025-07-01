package de.konradvoelkel.android.autokorrektur

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.*
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Core
import android.net.Uri
import de.konradvoelkel.android.autokorrektur.ml.YoloInferenceTFLite
import java.io.File
import java.io.FileOutputStream

@RunWith(AndroidJUnit4::class)
class SingleImageTest {

    @Test
    fun testSingleImageProcessing() {
        println("[DEBUG_LOG] Starting single image processing test")

        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val testContext = InstrumentationRegistry.getInstrumentation().context

        // Initialize OpenCV first
        try {
            println("[DEBUG_LOG] Initializing OpenCV")
            if (!org.opencv.android.OpenCVLoader.initDebug()) {
                println("[DEBUG_LOG] OpenCV initialization failed")
                fail("OpenCV initialization failed")
            } else {
                println("[DEBUG_LOG] OpenCV initialized successfully")
            }
        } catch (e: Exception) {
            println("[DEBUG_LOG] OpenCV initialization check failed: ${e.message}")
            fail("OpenCV initialization check failed: ${e.message}")
        }

        try {
            // Initialize YOLO inference
            println("[DEBUG_LOG] Creating YoloInference")
            val yoloInference = YoloInferenceTFLite(appContext)
            assertNotNull("YoloInference should not be null", yoloInference)

            println("[DEBUG_LOG] Initializing YOLO model")
            yoloInference.initialize("yolo11s")
            println("[DEBUG_LOG] YOLO model initialized successfully")

            // Initialize ImageProcessor
            println("[DEBUG_LOG] Creating ImageProcessor")
            val imageProcessor = ImageProcessor(appContext)
            assertNotNull("ImageProcessor should not be null", imageProcessor)

            // Test with one image file
            val mediaFile = "example1.jpeg"
            println("[DEBUG_LOG] Processing media file: $mediaFile")

            try {
                // Copy media file from test assets to internal storage for processing
                println("[DEBUG_LOG] Copying file from test assets")
                val mediaInputStream = testContext.assets.open(mediaFile)
                val tempFile = File(appContext.cacheDir, mediaFile)
                val outputStream = FileOutputStream(tempFile)

                mediaInputStream.use { input ->
                    outputStream.use { output ->
                        input.copyTo(output)
                    }
                }
                println("[DEBUG_LOG] File copied successfully")

                // Create URI from the temp file
                val fileUri = Uri.fromFile(tempFile)
                println("[DEBUG_LOG] Created URI: $fileUri")

                // Process the image through the pipeline
                val modelWidth = 640
                val modelHeight = 640

                // Process the image using ImageProcessor
                println("[DEBUG_LOG] Processing image with ImageProcessor")
                val processedImage = imageProcessor.processInputImage(
                    uri = fileUri,
                    modelWidth = modelWidth,
                    modelHeight = modelHeight,
                    downscaleMp = null
                )

                assertNotNull("Processed image should not be null", processedImage)
                assertNotNull("Transformed mat should not be null", processedImage.transformedMat)

                println("[DEBUG_LOG] Loaded image: ${processedImage.originalMat.rows()}x${processedImage.originalMat.cols()}")
                println("[DEBUG_LOG] Transformed mat: ${processedImage.transformedMat.rows()}x${processedImage.transformedMat.cols()}, type: ${processedImage.transformedMat.type()}")
                println("[DEBUG_LOG] Ratios - X: ${processedImage.xRatio}, Y: ${processedImage.yRatio}")

                // Run YOLO inference
                println("[DEBUG_LOG] Running YOLO inference")
                val resultMask = yoloInference.inferYolo(
                    transformedMat = processedImage.transformedMat,
                    xRatio = processedImage.xRatio,
                    yRatio = processedImage.yRatio,
                    modelWidth = modelWidth,
                    modelHeight = modelHeight,
                    upscaleFactor = 1.2f,
                    scoreThreshold = 0.5f,
                    downshiftFactor = 0.0f
                )
                println("[DEBUG_LOG] YOLO inference completed")

                assertNotNull("Result mask should not be null", resultMask)

                // Check if cars were detected by analyzing the mask
                val carsDetected = hasCarDetection(resultMask)
                println("[DEBUG_LOG] Cars detected: $carsDetected")

                // Clean up
                processedImage.originalMat.release()
                processedImage.transformedMat.release()
                resultMask.release()
                tempFile.delete()

                println("[DEBUG_LOG] Successfully processed $mediaFile")

            } catch (e: Exception) {
                println("[DEBUG_LOG] Error processing $mediaFile: ${e.message}")
                e.printStackTrace()
                fail("Failed to process $mediaFile: ${e.message}")
            }

            // Clean up YOLO inference
            yoloInference.close()

            println("[DEBUG_LOG] Single image processing test completed successfully")

        } catch (e: Exception) {
            println("[DEBUG_LOG] Unexpected error during test: ${e.message}")
            e.printStackTrace()
            fail("Test should not crash: ${e.message}")
        }
    }

    private fun hasCarDetection(mask: Mat): Boolean {
        // Count black pixels (car detections)
        val totalPixels = mask.rows() * mask.cols()
        var blackPixels = 0

        // Create a mask for black pixels (value = 0)
        val blackMask = Mat()
        Core.inRange(mask, Scalar(0.0), Scalar(10.0), blackMask) // Allow for small variations

        // Count non-zero pixels in the black mask (these represent detected cars)
        blackPixels = Core.countNonZero(blackMask)
        blackMask.release()

        // Consider cars detected if more than 0.1% of pixels are black (car pixels)
        val blackPixelRatio = blackPixels.toDouble() / totalPixels.toDouble()
        val threshold = 0.001 // 0.1% threshold

        println("[DEBUG_LOG] Black pixels: $blackPixels / $totalPixels (${String.format("%.4f", blackPixelRatio * 100)}%)")

        return blackPixelRatio > threshold
    }
}
