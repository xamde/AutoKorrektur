package de.konradvoelkel.android.autokorrektur

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.YoloInferenceTFLite
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.*

@Suppress("DEPRECATION")
@RunWith(AndroidJUnit4::class)
class MinimalTest {

    @Test
    fun testImageProcessorCreation() {
        println("[DEBUG_LOG] Starting minimal ImageProcessor test")

        val appContext = InstrumentationRegistry.getInstrumentation().targetContext

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
            // Test ImageProcessor creation
            println("[DEBUG_LOG] Creating ImageProcessor")
            val imageProcessor = ImageProcessor(appContext)
            assertNotNull("ImageProcessor should not be null", imageProcessor)
            println("[DEBUG_LOG] ImageProcessor created successfully")

            // Test YOLO creation
            println("[DEBUG_LOG] Creating YoloInference")
            val yoloInference = YoloInferenceTFLite(appContext)
            assertNotNull("YoloInference should not be null", yoloInference)
            println("[DEBUG_LOG] YoloInference created successfully")

            println("[DEBUG_LOG] Initializing YOLO model")
            yoloInference.initialize("yolo11s")
            println("[DEBUG_LOG] YOLO model initialized successfully")

            // Clean up
            yoloInference.close()
            println("[DEBUG_LOG] YOLO inference closed successfully")

            println("[DEBUG_LOG] Minimal test completed successfully")

        } catch (e: Exception) {
            println("[DEBUG_LOG] Minimal test failed: ${e.message}")
            e.printStackTrace()
            fail("Minimal test failed: ${e.message}")
        }
    }
}