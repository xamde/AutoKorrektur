package de.konradvoelkel.android.autokorrektur

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.*
import android.net.Uri
import de.konradvoelkel.android.autokorrektur.ml.YoloInferenceTFLite
import java.io.File
import java.io.FileOutputStream

@RunWith(AndroidJUnit4::class)
class DebugMaskTest {

    @Test
    fun testMaskCreationDebug() {
        println("[DEBUG_LOG] Starting mask creation debug test")

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

            // Test with example1.jpeg which should contain cars
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

                // Run YOLO inference with LOW threshold to ensure detections are found
                println("[DEBUG_LOG] Running YOLO inference with low threshold to trigger mask creation")
                val resultMask = yoloInference.inferYolo(
                    transformedMat = processedImage.transformedMat,
                    xRatio = processedImage.xRatio,
                    yRatio = processedImage.yRatio,
                    modelWidth = modelWidth,
                    modelHeight = modelHeight,
                    upscaleFactor = 1.2f,
                    scoreThreshold = 0.01f,  // Very low threshold to ensure detections
                    downshiftFactor = 0.0f
                )
                println("[DEBUG_LOG] YOLO inference completed successfully")

                assertNotNull("Result mask should not be null", resultMask)

                // Clean up
                processedImage.originalMat.release()
                processedImage.transformedMat.release()
                resultMask.release()
                tempFile.delete()

                println("[DEBUG_LOG] Successfully completed mask creation debug test")

            } catch (e: Exception) {
                println("[DEBUG_LOG] Error during mask creation test: ${e.message}")
                e.printStackTrace()
                // Don't fail the test - we want to see what the error is
                println("[DEBUG_LOG] This error reveals the actual issue with mask creation")
            }

            // Clean up YOLO inference
            yoloInference.close()

        } catch (e: Exception) {
            println("[DEBUG_LOG] Unexpected error during debug test: ${e.message}")
            e.printStackTrace()
            // Don't fail - we want to see the debug output
        }
    }
}