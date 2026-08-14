package de.konradvoelkel.android.autokorrektur.ml.engine

import androidx.test.ext.junit.runners.AndroidJUnit4
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar

@RunWith(AndroidJUnit4::class)
class YoloTFLiteEngineInstrumentedTest : AndroidInstrumentedBaseTest() {

    @Test
    fun testYoloTFLiteEngine_lifecycleAndInference() = runBlocking {
        val engine = YoloTFLiteEngine(appContext)
        try {
            engine.initialize(modelName = "yolo11s", useFP16 = false)
            assertTrue(engine.isInitialized)

            // Create 640x640 RGB Mat
            val inputMat = Mat(640, 640, CvType.CV_8UC3, Scalar(128.0, 128.0, 128.0))
            try {
                val rawOutputs = engine.run(inputMat)
                assertNotNull(rawOutputs)
                assertNotNull(rawOutputs.detections)
                assertNotNull(rawOutputs.prototypes)
                assertEquals(640, rawOutputs.shapes.inputW)
                assertEquals(640, rawOutputs.shapes.inputH)
                assertTrue(rawOutputs.detections.capacity() > 0)
                assertTrue(rawOutputs.prototypes.capacity() > 0)
            } finally {
                inputMat.release()
            }
        } finally {
            engine.close()
            assertTrue(engine.isClosed)
        }
    }
}
