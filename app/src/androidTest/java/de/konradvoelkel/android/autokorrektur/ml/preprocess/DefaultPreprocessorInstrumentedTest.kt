package de.konradvoelkel.android.autokorrektur.ml.preprocess

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SmallTest
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar

@RunWith(AndroidJUnit4::class)
@SmallTest
class DefaultPreprocessorInstrumentedTest : AndroidInstrumentedBaseTest() {

    @Test
    fun testPrepare_scalingAndPadding() {
        val preprocessor = DefaultPreprocessor(stride = 32)

        // Create a 100x200 RGB Mat
        val input = Mat(200, 100, CvType.CV_8UC3, Scalar(255.0, 0.0, 0.0)) // Red

        val targetW = 640
        val targetH = 640

        val result = preprocessor.prepare(input, targetW, targetH)

        assertNotNull(result.forEngine)
        assertNotNull(result.forBitmap)
        assertEquals(targetW, result.forEngine.cols())
        assertEquals(targetH, result.forEngine.rows())

        // Initial resize to divStride(32, 100, 200) -> 96x192
        // Pad to square (192x192) -> xRatio = 192/96 = 2.0, yRatio = 192/192 = 1.0
        assertEquals(2.0f, result.xRatio, 0.001f)
        assertEquals(1.0f, result.yRatio, 0.001f)

        assertEquals(targetW, result.forBitmap.cols())
        assertEquals(targetH, result.forBitmap.rows())

        input.release()
        result.forEngine.release()
        result.forBitmap.release()
    }
}
