package de.konradvoelkel.android.autokorrektur.ml

import androidx.test.ext.junit.runners.AndroidJUnit4
import de.konradvoelkel.android.autokorrektur.ml.factory.InpaintingEngineFactory
import de.konradvoelkel.android.autokorrektur.ml.factory.InpaintingModelType
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

@RunWith(AndroidJUnit4::class)
class LamaInferenceInstrumentedTest : AndroidInstrumentedBaseTest() {

    @Test
    fun testLamaInference_inpaintAndPreserveDimensions() = runBlocking {
        val engine = InpaintingEngineFactory.createEngine(appContext, InpaintingModelType.LAMA)
        try {
            engine.initialize()

            // Non-multiple-of-8 test image (517x373)
            val imageMat = Mat(373, 517, CvType.CV_8UC4, Scalar(100.0, 150.0, 200.0, 255.0))
            val maskMat = Mat(373, 517, CvType.CV_8UC1, Scalar(255.0)) // 255 = keep background
            // Car hole (0 = inpaint hole)
            Imgproc.rectangle(maskMat, Rect(50, 50, 100, 100), Scalar(0.0), -1)

            val output = engine.inpaint(imageMat, maskMat)
            try {
                assertNotNull(output)
                assertEquals(517, output.cols())
                assertEquals(373, output.rows())
                assertEquals(CvType.CV_8UC4, output.type())
            } finally {
                output.release()
                imageMat.release()
                maskMat.release()
            }
        } finally {
            engine.close()
        }
    }
}
