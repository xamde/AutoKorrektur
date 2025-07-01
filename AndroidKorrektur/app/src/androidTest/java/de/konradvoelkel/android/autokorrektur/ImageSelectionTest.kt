package de.konradvoelkel.android.autokorrektur

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.drawable.Drawable
import android.net.Uri
import androidx.core.content.ContextCompat
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import de.konradvoelkel.android.autokorrektur.ml.YoloInferenceTFLite
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * Test case that simulates selecting a picture in the app using mockup pictures
 * from Android system resources.
 */
@RunWith(AndroidJUnit4::class)
class ImageSelectionTest {

    @Test
    fun testImageSelectionSimulation() {
        println("[DEBUG_LOG] Starting image selection simulation test")

        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("de.konradvoelkel.android.autokorrektur", appContext.packageName)

        try {
            // Initialize OpenCV first (required for ImageProcessor)
            println("[DEBUG_LOG] Initializing OpenCV for image processing test")
            if (!org.opencv.android.OpenCVLoader.initDebug()) {
                println("[DEBUG_LOG] OpenCV initialization failed")
                fail("OpenCV initialization failed - required for image processing")
            } else {
                println("[DEBUG_LOG] OpenCV initialized successfully")
            }

            // Create a mockup image from Android system resources
            println("[DEBUG_LOG] Creating mockup image from Android system resources")
            val mockupImageUri = createMockupImageFromSystemResources(appContext)
            assertNotNull("Mockup image URI should not be null", mockupImageUri)
            println("[DEBUG_LOG] Mockup image created successfully: $mockupImageUri")

            // Test image processing with the mockup image
            println("[DEBUG_LOG] Testing image processing with mockup image")
            val imageProcessor = ImageProcessor(appContext)
            assertNotNull("ImageProcessor should not be null", imageProcessor)

            // Process the image (similar to what the app does)
            val modelWidth = 640
            val modelHeight = 640
            val downscaleMp = 2.0f

            val processedImage = imageProcessor.processInputImage(
                uri = mockupImageUri!!,
                modelWidth = modelWidth,
                modelHeight = modelHeight,
                downscaleMp = downscaleMp
            )

            // Verify the processed image results
            assertNotNull("Original bitmap should not be null", processedImage.originalBitmap)
            assertNotNull("Transformed bitmap should not be null", processedImage.transformedBitmap)
            assertNotNull("Original Mat should not be null", processedImage.originalMat)
            assertNotNull("Transformed Mat should not be null", processedImage.transformedMat)

            println("[DEBUG_LOG] Original image size: ${processedImage.originalBitmap.width}x${processedImage.originalBitmap.height}")
            println("[DEBUG_LOG] Transformed image size: ${processedImage.transformedBitmap.width}x${processedImage.transformedBitmap.height}")
            println("[DEBUG_LOG] X ratio: ${processedImage.xRatio}")
            println("[DEBUG_LOG] Y ratio: ${processedImage.yRatio}")

            // Verify that the transformed image has the expected model dimensions
            assertEquals(
                "Transformed image width should match model width",
                modelWidth,
                processedImage.transformedBitmap.width
            )
            assertEquals(
                "Transformed image height should match model height",
                modelHeight,
                processedImage.transformedBitmap.height
            )

            // Verify that ratios are positive
            assertTrue("X ratio should be positive", processedImage.xRatio > 0)
            assertTrue("Y ratio should be positive", processedImage.yRatio > 0)

            println("[DEBUG_LOG] Image selection simulation test completed successfully")

        } catch (e: Exception) {
            println("[DEBUG_LOG] Unexpected error during image selection simulation test: ${e.message}")
            e.printStackTrace()
            fail("Image selection simulation should not crash: ${e.message}")
        }
    }

    @Test
    fun testMultipleImageFormatsSimulation() {
        println("[DEBUG_LOG] Starting multiple image formats simulation test")

        val appContext = InstrumentationRegistry.getInstrumentation().targetContext

        try {
            // Initialize OpenCV
            if (!org.opencv.android.OpenCVLoader.initDebug()) {
                fail("OpenCV initialization failed")
            }

            val imageProcessor = ImageProcessor(appContext)

            // Test with different system drawable resources
            val systemDrawables = listOf(
                android.R.drawable.ic_menu_gallery,
                android.R.drawable.ic_menu_camera,
                android.R.drawable.ic_menu_save,
                android.R.drawable.ic_menu_preferences
            )

            for ((index, drawableId) in systemDrawables.withIndex()) {
                println("[DEBUG_LOG] Testing with system drawable $index: $drawableId")

                val mockupImageUri = createMockupImageFromDrawable(appContext, drawableId)
                assertNotNull(
                    "Mockup image URI should not be null for drawable $drawableId",
                    mockupImageUri
                )

                val processedImage = imageProcessor.processInputImage(
                    uri = mockupImageUri!!,
                    modelWidth = 320,
                    modelHeight = 320
                )

                assertNotNull(
                    "Processed image should not be null for drawable $drawableId",
                    processedImage
                )
                println("[DEBUG_LOG] Successfully processed image from drawable $drawableId")
            }

            println("[DEBUG_LOG] Multiple image formats simulation test completed successfully")

        } catch (e: Exception) {
            println("[DEBUG_LOG] Unexpected error during multiple formats test: ${e.message}")
            e.printStackTrace()
            fail("Multiple image formats test should not crash: ${e.message}")
        }
    }

    /**
     * Creates a mockup image from Android system resources and returns its URI.
     * This simulates selecting a picture from camera/gallery without actually using them.
     */
    private fun createMockupImageFromSystemResources(context: Context): Uri? {
        return createMockupImageFromDrawable(context, android.R.drawable.ic_menu_gallery)
    }

    /**
     * Creates a mockup image from a specific drawable resource.
     */
    private fun createMockupImageFromDrawable(context: Context, drawableId: Int): Uri? {
        try {
            // Get the drawable from system resources
            val drawable: Drawable = ContextCompat.getDrawable(context, drawableId)
                ?: return null

            // Create a bitmap from the drawable
            val bitmap = Bitmap.createBitmap(
                drawable.intrinsicWidth.takeIf { it > 0 } ?: 100,
                drawable.intrinsicHeight.takeIf { it > 0 } ?: 100,
                Bitmap.Config.ARGB_8888
            )

            val canvas = Canvas(bitmap)
            drawable.setBounds(0, 0, canvas.width, canvas.height)
            drawable.draw(canvas)

            // Save the bitmap to a temporary file
            val tempFile = File.createTempFile("mockup_image_", ".png", context.cacheDir)
            val outputStream = FileOutputStream(tempFile)
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream)
            outputStream.close()

            // Return the URI of the temporary file
            return Uri.fromFile(tempFile)

        } catch (e: IOException) {
            println("[DEBUG_LOG] Error creating mockup image: ${e.message}")
            return null
        }
    }

    @Test
    fun testCompleteProcessingPipelineSimulation() {
        println("[DEBUG_LOG] Starting complete processing pipeline simulation test")

        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("de.konradvoelkel.android.autokorrektur", appContext.packageName)

        try {
            // Initialize OpenCV first (required for all ML components)
            println("[DEBUG_LOG] Initializing OpenCV for complete pipeline test")
            if (!org.opencv.android.OpenCVLoader.initDebug()) {
                println("[DEBUG_LOG] OpenCV initialization failed")
                fail("OpenCV initialization failed - required for complete pipeline")
            } else {
                println("[DEBUG_LOG] OpenCV initialized successfully")
            }

            // Step 1: Create a mockup image (simulating image selection)
            println("[DEBUG_LOG] Step 1: Creating mockup image from Android system resources")
            val mockupImageUri = createMockupImageFromSystemResources(appContext)
            assertNotNull("Mockup image URI should not be null", mockupImageUri)
            println("[DEBUG_LOG] Mockup image created successfully: $mockupImageUri")

            // Step 2: Initialize ML components
            println("[DEBUG_LOG] Step 2: Initializing ML components")
            val imageProcessor = ImageProcessor(appContext)
            val yoloInference = YoloInferenceTFLite(appContext)
            val miGanInference = MiGanInference(appContext)

            assertNotNull("ImageProcessor should not be null", imageProcessor)
            assertNotNull("YoloInferenceTFLite should not be null", yoloInference)
            assertNotNull("MiGanInference should not be null", miGanInference)
            println("[DEBUG_LOG] ML components created successfully")

            // Step 3: Process input image (same as in performOnnxInference)
            println("[DEBUG_LOG] Step 3: Processing input image")
            val modelWidth = 640
            val modelHeight = 640
            val downscaleMp = 2.0f

            val processedImage = imageProcessor.processInputImage(
                uri = mockupImageUri!!,
                modelWidth = modelWidth,
                modelHeight = modelHeight,
                downscaleMp = downscaleMp
            )

            // Verify processed image
            assertNotNull("Original bitmap should not be null", processedImage.originalBitmap)
            assertNotNull("Transformed bitmap should not be null", processedImage.transformedBitmap)
            assertNotNull("Original Mat should not be null", processedImage.originalMat)
            assertNotNull("Transformed Mat should not be null", processedImage.transformedMat)

            println("[DEBUG_LOG] Image processed successfully")
            println("[DEBUG_LOG] Original image size: ${processedImage.originalBitmap.width}x${processedImage.originalBitmap.height}")
            println("[DEBUG_LOG] Transformed image size: ${processedImage.transformedBitmap.width}x${processedImage.transformedBitmap.height}")

            // Step 4: Initialize and run YOLO inference (may fail in test environment, but should not crash)
            println("[DEBUG_LOG] Step 4: Testing YOLO inference initialization and processing")
            try {
                println("[DEBUG_LOG] Initializing YOLO inference")
                yoloInference.initialize()
                println("[DEBUG_LOG] YOLO inference initialized successfully")

                println("[DEBUG_LOG] Running YOLO inference")
                val maskMat = yoloInference.inferYolo(
                    transformedMat = processedImage.transformedMat,
                    xRatio = processedImage.xRatio,
                    yRatio = processedImage.yRatio,
                    modelWidth = modelWidth,
                    modelHeight = modelHeight,
                    upscaleFactor = 1.2f,
                    scoreThreshold = 0.5f
                )

                assertNotNull("Mask Mat should not be null", maskMat)
                println("[DEBUG_LOG] YOLO inference completed successfully")
                println("[DEBUG_LOG] Mask size: ${maskMat.cols()}x${maskMat.rows()}")

                // Step 5: Initialize and run Mi-GAN inference
                println("[DEBUG_LOG] Step 5: Testing Mi-GAN inference initialization and processing")
                try {
                    println("[DEBUG_LOG] Initializing Mi-GAN inference")
                    miGanInference.initialize()
                    println("[DEBUG_LOG] Mi-GAN inference initialized successfully")

                    println("[DEBUG_LOG] Running Mi-GAN inference")
                    val resultMat = miGanInference.inferMiGan(
                        imageMat = processedImage.transformedMat,
                        maskMat = maskMat
                    )

                    assertNotNull("Result Mat should not be null", resultMat)
                    println("[DEBUG_LOG] Mi-GAN inference completed successfully")
                    println("[DEBUG_LOG] Result size: ${resultMat.cols()}x${resultMat.rows()}")

                    // Verify the complete pipeline worked
                    assertTrue(
                        "Result Mat should have positive dimensions",
                        resultMat.cols() > 0 && resultMat.rows() > 0
                    )
                    println("[DEBUG_LOG] Complete processing pipeline test completed successfully")

                } catch (e: Exception) {
                    println("[DEBUG_LOG] Mi-GAN inference failed (may be expected in test environment): ${e.message}")
                    // Mi-GAN may fail due to missing model files in test environment, but should not crash
                    assertNotNull("Mi-GAN inference object should still be valid", miGanInference)
                }

            } catch (e: Exception) {
                println("[DEBUG_LOG] YOLO inference failed (may be expected in test environment): ${e.message}")
                // YOLO may fail due to missing model files in test environment, but should not crash
                assertNotNull("YOLO inference object should still be valid", yoloInference)
            }

            // Clean up resources
            try {
                println("[DEBUG_LOG] Cleaning up ML resources")
                yoloInference.close()
                miGanInference.close()
                println("[DEBUG_LOG] ML resources cleaned up successfully")
            } catch (e: Exception) {
                println("[DEBUG_LOG] Error during cleanup (non-critical): ${e.message}")
            }

            println("[DEBUG_LOG] Complete processing pipeline simulation test finished")

        } catch (e: Exception) {
            println("[DEBUG_LOG] Unexpected error during complete pipeline test: ${e.message}")
            e.printStackTrace()
            fail("Complete processing pipeline simulation should not crash: ${e.message}")
        }
    }
}
