package de.konradvoelkel.android.autokorrektur.ml

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import de.konradvoelkel.android.autokorrektur.ml.progressive.ProgressiveTileInpainter
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar

@RunWith(AndroidJUnit4::class)
class ProgressiveTileInpainterInstrumentedTest {

    @Before
    fun setUp() {
        OpenCVLoader.initLocal()
    }

    @Test
    fun testCreateFeatheredMask_dimensionsAndType() {
        val mask = ProgressiveTileInpainter.createFeatheredMask(400, 300, featherPx = 16)
        assertNotNull(mask)
        assertEquals(400, mask.cols())
        assertEquals(300, mask.rows())
        assertEquals(CvType.CV_8UC1, mask.type())
        mask.release()
    }

    @Test
    fun testInpaintProgressive_withMockEngine() = runTest {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val mockEngine = object : InpaintingEngine {
            override suspend fun initialize() {}
            override suspend fun inpaint(imageMat: Mat, maskMat: Mat): Mat = imageMat.clone()
            override fun close() {}
        }

        val inpainter = ProgressiveTileInpainter(mockEngine)
        val imageMat = Mat(600, 800, CvType.CV_8UC4, Scalar(120.0, 140.0, 160.0, 255.0))
        val maskMat = Mat(600, 800, CvType.CV_8UC1, Scalar(255.0))

        // Draw a simulated car box in the center (0 in subtractive mask)
        val carRoi = maskMat.submat(Rect(200, 150, 400, 300))
        carRoi.setTo(Scalar(0.0))
        carRoi.release()

        var progressReported = false
        val result = inpainter.inpaintProgressive(imageMat, maskMat) { stage, percent, preview ->
            progressReported = true
        }

        assertNotNull(result)
        assertEquals(800, result.cols())
        assertEquals(600, result.rows())
        assertEquals(imageMat.type(), result.type())

        imageMat.release()
        maskMat.release()
        result.release()
    }
}
