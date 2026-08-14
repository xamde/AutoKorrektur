package de.konradvoelkel.android.autokorrektur.ar

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SmallTest
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar

@RunWith(AndroidJUnit4::class)
@SmallTest
class TemporalBackgroundAccumulatorInstrumentedTest : AndroidInstrumentedBaseTest() {

    @Test
    fun accumulateAndBlend_blendsBackgroundIntoMaskRegion() {
        val accumulator = TemporalBackgroundAccumulator()

        // 1. Initial clean frame: all green (0, 255, 0)
        val cleanFrame = Mat(100, 100, CvType.CV_8UC3, Scalar(0.0, 255.0, 0.0))
        val noCarMask = Mat(100, 100, CvType.CV_8UC1, Scalar(0.0)) // 0 = no vehicle

        val out1 = accumulator.accumulateAndBlend(cleanFrame, noCarMask)
        assertNotNull(out1)
        assertEquals(100, out1.cols())
        assertEquals(100, out1.rows())

        // 2. Second frame: frame has red car in center, mask indicates vehicle in center (255)
        val carFrame = Mat(100, 100, CvType.CV_8UC3, Scalar(0.0, 0.0, 255.0))
        val carMask = Mat(100, 100, CvType.CV_8UC1, Scalar(255.0)) // 255 = vehicle to replace

        val blendedOut = accumulator.accumulateAndBlend(carFrame, carMask)

        // Pixel in replaced region should now be green (accumulated from first clean frame)
        val pixel = blendedOut.get(50, 50)
        assertNotNull(pixel)
        assertEquals(0.0, pixel[0], 1.0)   // Blue
        assertEquals(255.0, pixel[1], 1.0) // Green (from clean frame)
        assertEquals(0.0, pixel[2], 1.0)   // Red

        cleanFrame.release()
        noCarMask.release()
        carFrame.release()
        carMask.release()
        out1.release()
        blendedOut.release()
        accumulator.close()
    }

    @Test
    fun reset_clearsInternalBackgroundBuffer() {
        val accumulator = TemporalBackgroundAccumulator()
        val frame = Mat(50, 50, CvType.CV_8UC3, Scalar(100.0, 100.0, 100.0))
        val mask = Mat(50, 50, CvType.CV_8UC1, Scalar(0.0))

        val out = accumulator.accumulateAndBlend(frame, mask)
        assertNotNull(out)
        accumulator.reset()

        frame.release()
        mask.release()
        out.release()
        accumulator.close()
    }
}
