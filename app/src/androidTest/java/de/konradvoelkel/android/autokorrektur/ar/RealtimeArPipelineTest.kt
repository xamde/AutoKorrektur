package de.konradvoelkel.android.autokorrektur.ar

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar

@RunWith(AndroidJUnit4::class)
class RealtimeArPipelineTest : AndroidInstrumentedBaseTest() {

    @Test
    fun testRealtimeArPipeline_initializationAndProcessFrame() = runBlocking {
        val yoloEngine = YoloTFLiteEngine(appContext)
        val yoloService = YoloServiceImpl(yoloEngine)
        val pipeline = RealtimeArPipeline(yoloService)

        try {
            pipeline.initialize(modelName = "yolo11s")
            assertTrue(pipeline.isInitialized)

            var frameRendered: Bitmap? = null
            var fpsReceived = 0f
            pipeline.onFrameRendered = { bitmap, fps ->
                frameRendered = bitmap
                fpsReceived = fps
            }

            // Create sample 640x480 RGBA frame
            val frameMat = Mat(480, 640, CvType.CV_8UC4, Scalar(120.0, 140.0, 160.0, 255.0))
            try {
                pipeline.processFrame(frameMat)
                // Wait briefly for background inference and accumulator blending
                var waited = 0
                while (frameRendered == null && waited < 2000) {
                    kotlinx.coroutines.delay(50)
                    waited += 50
                }

                assertNotNull("Expected pipeline to render blended output bitmap", frameRendered)
                assertTrue(pipeline.accumulator.hasAccumulatedBackground)
            } finally {
                frameMat.release()
            }
        } finally {
            pipeline.close()
            assertTrue(pipeline.isClosed)
        }
    }

    @Test
    fun testRealtimeArPipeline_reset_clearsAccumulator() = runBlocking {
        val yoloEngine = YoloTFLiteEngine(appContext)
        val yoloService = YoloServiceImpl(yoloEngine)
        val pipeline = RealtimeArPipeline(yoloService)

        try {
            pipeline.initialize(modelName = "yolo11s")
            val frameMat = Mat(480, 640, CvType.CV_8UC4, Scalar(100.0, 100.0, 100.0, 255.0))
            try {
                pipeline.processFrame(frameMat)
                var waited = 0
                while (!pipeline.accumulator.hasAccumulatedBackground && waited < 3000) {
                    kotlinx.coroutines.delay(50)
                    waited += 50
                }
                assertTrue(pipeline.accumulator.hasAccumulatedBackground)

                pipeline.reset()
                assertTrue(!pipeline.accumulator.hasAccumulatedBackground)
            } finally {
                frameMat.release()
            }
        } finally {
            pipeline.close()
        }
    }
}
