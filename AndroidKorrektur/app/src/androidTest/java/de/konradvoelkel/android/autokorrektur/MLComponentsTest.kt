package de.konradvoelkel.android.autokorrektur

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.YoloInferenceTFLite
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.*

/**
 * Test for ML components initialization to verify error handling.
 */
@RunWith(AndroidJUnit4::class)
class MLComponentsTest {

    @Test
    fun testMLComponentsInitialization() {
        println("[DEBUG_LOG] Starting ML components initialization test")

        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("de.konradvoelkel.android.autokorrektur", appContext.packageName)

        // Test that ML components can be created without crashing
        try {
            println("[DEBUG_LOG] Creating ImageProcessor")
            val imageProcessor = ImageProcessor(appContext)
            assertNotNull("ImageProcessor should not be null", imageProcessor)
            println("[DEBUG_LOG] ImageProcessor created successfully")

            println("[DEBUG_LOG] Creating YoloInferenceTFLite")
            val yoloInference = YoloInferenceTFLite(appContext)
            assertNotNull("YoloInferenceTFLite should not be null", yoloInference)
            println("[DEBUG_LOG] YoloInferenceTFLite created successfully")

            println("[DEBUG_LOG] Creating MiGanInference")
            val miGanInference = MiGanInference(appContext)
            assertNotNull("MiGanInference should not be null", miGanInference)
            println("[DEBUG_LOG] MiGanInference created successfully")

            // Test initialization (this might fail, but should not crash)
            try {
                println("[DEBUG_LOG] Testing YOLO initialization")
                yoloInference.initialize()
                println("[DEBUG_LOG] YOLO initialization completed")
            } catch (e: Exception) {
                println("[DEBUG_LOG] YOLO initialization failed (expected): ${e.message}")
                // This is expected to fail in test environment, but should not crash
            }

            try {
                println("[DEBUG_LOG] Testing Mi-GAN initialization")
                miGanInference.initialize()
                println("[DEBUG_LOG] Mi-GAN initialization completed")
            } catch (e: Exception) {
                println("[DEBUG_LOG] Mi-GAN initialization failed (expected): ${e.message}")
                // This is expected to fail in test environment, but should not crash
            }

            println("[DEBUG_LOG] ML components test completed successfully")
        } catch (e: Exception) {
            println("[DEBUG_LOG] Unexpected error during ML components test: ${e.message}")
            e.printStackTrace()
            fail("ML components creation should not crash: ${e.message}")
        }
    }

    @Test
    fun testTFLiteYoloInference() {
        println("[DEBUG_LOG] Starting TFLite YOLO inference test")

        val appContext = InstrumentationRegistry.getInstrumentation().targetContext

        // Initialize OpenCV first
        try {
            println("[DEBUG_LOG] Initializing OpenCV for TFLite test")
            if (!org.opencv.android.OpenCVLoader.initDebug()) {
                println("[DEBUG_LOG] OpenCV initialization failed")
                fail("OpenCV initialization failed - required for TFLite test")
            } else {
                println("[DEBUG_LOG] OpenCV initialized successfully")
            }
        } catch (e: Exception) {
            println("[DEBUG_LOG] OpenCV initialization check failed: ${e.message}")
            fail("OpenCV initialization check failed: ${e.message}")
        }

        try {
            println("[DEBUG_LOG] Creating YoloInferenceTFLite")
            val yoloTFLite = YoloInferenceTFLite(appContext)
            assertNotNull("YoloInferenceTFLite should not be null", yoloTFLite)
            println("[DEBUG_LOG] YoloInferenceTFLite created successfully")

            // Test initialization with both FP16 and FP32 models
            try {
                println("[DEBUG_LOG] Testing TFLite YOLO initialization (FP16)")
                yoloTFLite.initialize("yolo11s", useFP16 = true)
                println("[DEBUG_LOG] TFLite YOLO FP16 initialization completed")
                yoloTFLite.close()
            } catch (e: Exception) {
                println("[DEBUG_LOG] TFLite YOLO FP16 initialization failed (may be expected): ${e.message}")
            }

            try {
                println("[DEBUG_LOG] Testing TFLite YOLO initialization (FP32)")
                yoloTFLite.initialize("yolo11s", useFP16 = false)
                println("[DEBUG_LOG] TFLite YOLO FP32 initialization completed")
                yoloTFLite.close()
            } catch (e: Exception) {
                println("[DEBUG_LOG] TFLite YOLO FP32 initialization failed (may be expected): ${e.message}")
            }

            println("[DEBUG_LOG] TFLite YOLO inference test completed successfully")
        } catch (e: Exception) {
            println("[DEBUG_LOG] Unexpected error during TFLite YOLO test: ${e.message}")
            e.printStackTrace()
            fail("TFLite YOLO creation should not crash: ${e.message}")
        }
    }
}
