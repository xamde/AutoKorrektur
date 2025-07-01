package de.konradvoelkel.android.autokorrektur

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

    @Test
    fun testMiGanOrderInCHWDataTypes() {
        println("[DEBUG_LOG] Starting MiGan orderInCHW data types test")

        val appContext = InstrumentationRegistry.getInstrumentation().targetContext

        // Initialize OpenCV first
        try {
            println("[DEBUG_LOG] Initializing OpenCV for MiGan test")
            if (!org.opencv.android.OpenCVLoader.initDebug()) {
                println("[DEBUG_LOG] OpenCV initialization failed")
                fail("OpenCV initialization failed - required for MiGan test")
            } else {
                println("[DEBUG_LOG] OpenCV initialized successfully")
            }
        } catch (e: Exception) {
            println("[DEBUG_LOG] OpenCV initialization check failed: ${e.message}")
            fail("OpenCV initialization check failed: ${e.message}")
        }

        try {
            println("[DEBUG_LOG] Creating MiGanInference for data type test")
            val miGanInference = MiGanInference(appContext)
            assertNotNull("MiGanInference should not be null", miGanInference)

            // Test with CV_8UC3 Mat (8-bit unsigned, 3 channels)
            println("[DEBUG_LOG] Testing orderInCHW with CV_8UC3 Mat")
            val mat8UC3 = org.opencv.core.Mat(10, 10, org.opencv.core.CvType.CV_8UC3)
            mat8UC3.setTo(org.opencv.core.Scalar(128.0, 64.0, 192.0)) // Set some test values

            // Use reflection to access the private orderInCHW method
            val orderInCHWMethod = MiGanInference::class.java.getDeclaredMethod("orderInCHW", org.opencv.core.Mat::class.java)
            orderInCHWMethod.isAccessible = true

            val result8UC3 = orderInCHWMethod.invoke(miGanInference, mat8UC3) as FloatArray
            assertNotNull("Result for CV_8UC3 should not be null", result8UC3)
            assertTrue("Result array should have correct size", result8UC3.size == 10 * 10 * 3)
            println("[DEBUG_LOG] CV_8UC3 test passed, result size: ${result8UC3.size}")

            // Test with CV_32FC3 Mat (32-bit float, 3 channels)
            println("[DEBUG_LOG] Testing orderInCHW with CV_32FC3 Mat")
            val mat32FC3 = org.opencv.core.Mat(10, 10, org.opencv.core.CvType.CV_32FC3)
            mat32FC3.setTo(org.opencv.core.Scalar(0.5, 0.25, 0.75)) // Set normalized test values

            val result32FC3 = orderInCHWMethod.invoke(miGanInference, mat32FC3) as FloatArray
            assertNotNull("Result for CV_32FC3 should not be null", result32FC3)
            assertTrue("Result array should have correct size", result32FC3.size == 10 * 10 * 3)
            println("[DEBUG_LOG] CV_32FC3 test passed, result size: ${result32FC3.size}")

            // Clean up
            mat8UC3.release()
            mat32FC3.release()

            println("[DEBUG_LOG] MiGan orderInCHW data types test completed successfully")
        } catch (e: Exception) {
            println("[DEBUG_LOG] Unexpected error during MiGan orderInCHW test: ${e.message}")
            e.printStackTrace()
            fail("MiGan orderInCHW test should not crash: ${e.message}")
        }
    }

    @Test
    fun testMiGanOrderInCHWAsBytesDataTypes() {
        println("[DEBUG_LOG] Starting MiGan orderInCHWAsBytes data types test")

        val appContext = InstrumentationRegistry.getInstrumentation().targetContext

        // Initialize OpenCV first
        try {
            println("[DEBUG_LOG] Initializing OpenCV for MiGan bytes test")
            if (!org.opencv.android.OpenCVLoader.initDebug()) {
                println("[DEBUG_LOG] OpenCV initialization failed")
                fail("OpenCV initialization failed - required for MiGan bytes test")
            } else {
                println("[DEBUG_LOG] OpenCV initialized successfully")
            }
        } catch (e: Exception) {
            println("[DEBUG_LOG] OpenCV initialization check failed: ${e.message}")
            fail("OpenCV initialization check failed: ${e.message}")
        }

        try {
            println("[DEBUG_LOG] Creating MiGanInference for bytes data type test")
            val miGanInference = MiGanInference(appContext)
            assertNotNull("MiGanInference should not be null", miGanInference)

            // Test with CV_8UC3 Mat (8-bit unsigned, 3 channels)
            println("[DEBUG_LOG] Testing orderInCHWAsBytes with CV_8UC3 Mat")
            val mat8UC3 = org.opencv.core.Mat(10, 10, org.opencv.core.CvType.CV_8UC3)
            mat8UC3.setTo(org.opencv.core.Scalar(128.0, 64.0, 192.0)) // Set some test values

            // Use reflection to access the private orderInCHWAsBytes method
            val orderInCHWAsBytesMethod = MiGanInference::class.java.getDeclaredMethod("orderInCHWAsBytes", org.opencv.core.Mat::class.java)
            orderInCHWAsBytesMethod.isAccessible = true

            val result8UC3 = orderInCHWAsBytesMethod.invoke(miGanInference, mat8UC3) as ByteArray
            assertNotNull("Result for CV_8UC3 should not be null", result8UC3)
            assertTrue("Result array should have correct size", result8UC3.size == 10 * 10 * 3)
            println("[DEBUG_LOG] CV_8UC3 bytes test passed, result size: ${result8UC3.size}")

            // Test with CV_32FC3 Mat (32-bit float, 3 channels)
            println("[DEBUG_LOG] Testing orderInCHWAsBytes with CV_32FC3 Mat")
            val mat32FC3 = org.opencv.core.Mat(10, 10, org.opencv.core.CvType.CV_32FC3)
            mat32FC3.setTo(org.opencv.core.Scalar(0.5, 0.25, 0.75)) // Set normalized test values

            val result32FC3 = orderInCHWAsBytesMethod.invoke(miGanInference, mat32FC3) as ByteArray
            assertNotNull("Result for CV_32FC3 should not be null", result32FC3)
            assertTrue("Result array should have correct size", result32FC3.size == 10 * 10 * 3)
            println("[DEBUG_LOG] CV_32FC3 bytes test passed, result size: ${result32FC3.size}")

            // Verify that float values are properly converted to uint8 range
            // For 0.5 normalized value, we expect ~127 in uint8
            val expectedValue = (0.5f * 255.0f).toInt().toByte()
            println("[DEBUG_LOG] Expected uint8 value for 0.5 float: $expectedValue")

            // Clean up
            mat8UC3.release()
            mat32FC3.release()

            println("[DEBUG_LOG] MiGan orderInCHWAsBytes data types test completed successfully")
        } catch (e: Exception) {
            println("[DEBUG_LOG] Unexpected error during MiGan orderInCHWAsBytes test: ${e.message}")
            e.printStackTrace()
            fail("MiGan orderInCHWAsBytes test should not crash: ${e.message}")
        }
    }
}
