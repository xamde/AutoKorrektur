package de.konradvoelkel.android.autokorrektur

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import de.konradvoelkel.android.autokorrektur.ml.YoloInferenceTFLite
import org.junit.Assert.assertNotNull
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class SimpleYoloTest {

    @Test
    fun testYoloInitialization() {
        println("[DEBUG_LOG] Starting simple YOLO initialization test")

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
            // Initialize YOLO inference
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

            println("[DEBUG_LOG] Simple YOLO initialization test completed successfully")

        } catch (e: Exception) {
            println("[DEBUG_LOG] YOLO initialization failed: ${e.message}")
            e.printStackTrace()
            fail("YOLO initialization failed: ${e.message}")
        }
    }
}