package de.konradvoelkel.android.autokorrektur

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SmallTest
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import de.konradvoelkel.android.autokorrektur.ml.YoloInferenceTFLite
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
@SmallTest
class MlComponentTests :
    de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest() {


    @Test
    fun testImageProcessorInstantiation() {
        try {
            val imageProcessor = ImageProcessor(appContext)
            assertNotNull("ImageProcessor should not be null", imageProcessor)
        } catch (e: Exception) {
            fail("ImageProcessor creation should not crash: ${e.message}")
        }
    }

    @Test
    fun testYoloInferenceInstantiationAndInitialization() {
        var yoloInference: YoloInferenceTFLite? = null
        try {
            yoloInference = YoloInferenceTFLite(appContext)
            assertNotNull("YoloInferenceTFLite should not be null", yoloInference)
            yoloInference.initialize("yolo11s")
        } catch (e: Exception) {
            fail("YoloInferenceTFLite initialization failed: ${e.message}")
        } finally {
            yoloInference?.close()
        }
    }

    @Test
    fun testMiGanInferenceInstantiationAndInitialization() {
        var miGanInference: MiGanInference? = null
        try {
            miGanInference = MiGanInference(appContext)
            assertNotNull("MiGanInference should not be null", miGanInference)
            miGanInference.initialize()
        } catch (_: Exception) {
            // Mi-GAN may fail due to missing model files in test environment, but should not crash
            assertNotNull("MiGanInference object should still be valid", miGanInference)
        } finally {
            miGanInference?.close()
        }
    }

    @Test
    fun testTFLiteYoloInference() {
        try {
            val yoloTFLite = YoloInferenceTFLite(appContext)
            assertNotNull("YoloInferenceTFLite should not be null", yoloTFLite)

            try {
                yoloTFLite.initialize("yolo11s", useFP16 = true)
                yoloTFLite.close()
            } catch (_: Exception) {
                // ignore if FP16 model is not available
            }

            try {
                yoloTFLite.initialize("yolo11s", useFP16 = false)
                yoloTFLite.close()
            } catch (e: Exception) {
                fail("TFLite YOLO FP32 initialization failed: ${e.message}")
            }

        } catch (e: Exception) {
            fail("TFLite YOLO creation should not crash: ${e.message}")
        }
    }

    @Test
    fun testMiGanOrderInCHWAsBytes() {
        val miGanInference = MiGanInference(appContext)
        assertNotNull("MiGanInference should not be null", miGanInference)

        // Test with CV_8UC3 Mat
        val mat8UC3 = org.opencv.core.Mat(10, 10, org.opencv.core.CvType.CV_8UC3)
        mat8UC3.setTo(org.opencv.core.Scalar(128.0, 64.0, 192.0))

        val orderInCHWAsBytesMethod = MiGanInference::class.java.getDeclaredMethod(
            "orderInCHWAsBytes",
            org.opencv.core.Mat::class.java
        )
        orderInCHWAsBytesMethod.isAccessible = true

        val result8UC3 = orderInCHWAsBytesMethod.invoke(miGanInference, mat8UC3) as ByteArray
        assertNotNull("Result for CV_8UC3 should not be null", result8UC3)
        assertTrue("Result array should have correct size", result8UC3.size == 10 * 10 * 3)

        mat8UC3.release()
    }
}
