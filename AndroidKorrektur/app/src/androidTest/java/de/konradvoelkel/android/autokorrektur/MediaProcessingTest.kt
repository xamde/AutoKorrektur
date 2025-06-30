package de.konradvoelkel.android.autokorrektur

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.YoloInference
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.*
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Core
import android.net.Uri
import java.io.File
import java.io.FileOutputStream

@RunWith(AndroidJUnit4::class)
class MediaProcessingTest {

    @Test
    fun testMediaFilesCarDetection() {
        println("[DEBUG_LOG] Starting media files car detection test")

        val appContext = InstrumentationRegistry.getInstrumentation().targetContext

        // Initialize OpenCV first
        try {
            println("[DEBUG_LOG] Initializing OpenCV for media processing test")
            if (!org.opencv.android.OpenCVLoader.initDebug()) {
                println("[DEBUG_LOG] OpenCV initialization failed")
                fail("OpenCV initialization failed - required for media processing test")
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
            val yoloInference = YoloInference(appContext)
            assertNotNull("YoloInference should not be null", yoloInference)

            println("[DEBUG_LOG] Initializing YOLO model")
            try {
                yoloInference.initialize("yolo11s")
                println("[DEBUG_LOG] YOLO model initialized successfully")
            } catch (e: Exception) {
                println("[DEBUG_LOG] YOLO initialization failed: ${e.message}")
                e.printStackTrace()
                fail("YOLO initialization failed: ${e.message}")
            }

            // Initialize ImageProcessor
            println("[DEBUG_LOG] Creating ImageProcessor")
            val imageProcessor = ImageProcessor(appContext)
            assertNotNull("ImageProcessor should not be null", imageProcessor)

            // Define the media files to test
            val mediaFiles = listOf(
                "example1.jpeg",      // Should contain cars
                "example2.png",       // Should contain cars
                "example2Other.png",  // Should NOT contain cars
                "example2This.png"    // Should NOT contain cars
                // Note: example1Result.jpeg is excluded as it's a processed result
            )

            val expectedCarDetections = mapOf(
                "example1.jpeg" to true,      // Should contain cars
                "example2.png" to true,       // Should contain cars
                "example2Other.png" to false, // Should NOT contain cars
                "example2This.png" to false   // Should NOT contain cars
            )

            // Process each media file
            for (mediaFile in mediaFiles) {
                println("[DEBUG_LOG] Processing media file: $mediaFile")

                try {
                    // Copy media file from assets to internal storage for processing
                    val mediaInputStream = appContext.assets.open(mediaFile)
                    val tempFile = File(appContext.cacheDir, mediaFile)
                    val outputStream = FileOutputStream(tempFile)

                    mediaInputStream.use { input ->
                        outputStream.use { output ->
                            input.copyTo(output)
                        }
                    }

                    // Create URI from the temp file
                    val fileUri = Uri.fromFile(tempFile)

                    // Process the image through the pipeline
                    val modelWidth = 640
                    val modelHeight = 640

                    // Process the image using ImageProcessor
                    println("[DEBUG_LOG] Processing image with ImageProcessor for $mediaFile")
                    val processedImage = imageProcessor.processInputImage(
                        uri = fileUri,
                        modelWidth = modelWidth,
                        modelHeight = modelHeight,
                        downscaleMp = null
                    )

                    assertNotNull("Processed image should not be null for $mediaFile", processedImage)
                    assertNotNull("Transformed mat should not be null for $mediaFile", processedImage.transformedMat)

                    println("[DEBUG_LOG] Loaded image $mediaFile: ${processedImage.originalMat.rows()}x${processedImage.originalMat.cols()}")
                    println("[DEBUG_LOG] Transformed mat: ${processedImage.transformedMat.rows()}x${processedImage.transformedMat.cols()}, type: ${processedImage.transformedMat.type()}")
                    println("[DEBUG_LOG] Ratios - X: ${processedImage.xRatio}, Y: ${processedImage.yRatio}")

                    // Run YOLO inference
                    println("[DEBUG_LOG] Running YOLO inference for $mediaFile")
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
                    println("[DEBUG_LOG] YOLO inference completed for $mediaFile")

                    assertNotNull("Result mask should not be null for $mediaFile", resultMask)

                    // Check if cars were detected by analyzing the mask
                    // The mask has black pixels (0) where cars are detected, white pixels (255) for background
                    val carsDetected = hasCarDetection(resultMask)

                    println("[DEBUG_LOG] $mediaFile - Cars detected: $carsDetected")

                    // Verify against expected results
                    val expectedDetection = expectedCarDetections[mediaFile] ?: false
                    assertEquals(
                        "Car detection mismatch for $mediaFile. Expected: $expectedDetection, Got: $carsDetected",
                        expectedDetection,
                        carsDetected
                    )

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
            }

            // Clean up YOLO inference
            yoloInference.close()

            println("[DEBUG_LOG] Media files car detection test completed successfully")

        } catch (e: Exception) {
            println("[DEBUG_LOG] Unexpected error during media processing test: ${e.message}")
            e.printStackTrace()
            fail("Media processing test should not crash: ${e.message}")
        }
    }

    /**
     * Analyzes a mask to determine if cars were detected.
     * Cars are represented by black pixels (0), background by white pixels (255).
     * Returns true if a significant number of black pixels are found.
     */
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
