package de.konradvoelkel.android.autokorrektur.ar

import androidx.test.ext.junit.runners.AndroidJUnit4
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

@RunWith(AndroidJUnit4::class)
class TemporalBackgroundAccumulatorTest : AndroidInstrumentedBaseTest() {

    @Test
    fun testAccumulateAndBlend_singleFrame_preservesOutputDimensions() {
        val accumulator = TemporalBackgroundAccumulator()
        try {
            val frame = Mat(480, 640, CvType.CV_8UC4, Scalar(100.0, 150.0, 200.0, 255.0))
            val mask = Mat.zeros(480, 640, CvType.CV_8UC1)
            // Car mask at center (255 = car)
            Imgproc.rectangle(mask, Rect(200, 150, 240, 180), Scalar(255.0), -1)

            val output = accumulator.accumulateAndBlend(frame, mask)
            try {
                assertNotNull(output)
                assertEquals(640, output.cols())
                assertEquals(480, output.rows())
                assertEquals(CvType.CV_8UC4, output.type())
            } finally {
                output.release()
                frame.release()
                mask.release()
            }
        } finally {
            accumulator.close()
        }
    }

    @Test
    fun testAccumulateAndBlend_multiFrameMotion_fillsCarRegionWithBackground() {
        val accumulator = TemporalBackgroundAccumulator()
        try {
            val width = 640
            val height = 480

            // Frame 1: Road texture is Green (0, 255, 0), Car at left [50, 50, 200, 200] is Red (255, 0, 0)
            val frame1 = Mat(height, width, CvType.CV_8UC4, Scalar(0.0, 255.0, 0.0, 255.0))
            Imgproc.rectangle(frame1, Rect(50, 50, 200, 200), Scalar(255.0, 0.0, 0.0, 255.0), -1)
            val mask1 = Mat.zeros(height, width, CvType.CV_8UC1)
            Imgproc.rectangle(mask1, Rect(50, 50, 200, 200), Scalar(255.0), -1)

            val out1 = accumulator.accumulateAndBlend(frame1, mask1)
            out1.release()

            // Frame 2: Camera pans, car moves to right [350, 50, 200, 200].
            // The left region [50, 50, 200, 200] is now clean Green background!
            val frame2 = Mat(height, width, CvType.CV_8UC4, Scalar(0.0, 255.0, 0.0, 255.0))
            Imgproc.rectangle(frame2, Rect(350, 50, 200, 200), Scalar(255.0, 0.0, 0.0, 255.0), -1)
            val mask2 = Mat.zeros(height, width, CvType.CV_8UC1)
            Imgproc.rectangle(mask2, Rect(350, 50, 200, 200), Scalar(255.0), -1)

            val out2 = accumulator.accumulateAndBlend(frame2, mask2)
            out2.release()

            // Frame 3: Car moves back to left [50, 50, 200, 200].
            // Because left region was populated with Green in Frame 2, the accumulator replaces the Red car with Green!
            val frame3 = Mat(height, width, CvType.CV_8UC4, Scalar(0.0, 255.0, 0.0, 255.0))
            Imgproc.rectangle(frame3, Rect(50, 50, 200, 200), Scalar(255.0, 0.0, 0.0, 255.0), -1)
            val mask3 = Mat.zeros(height, width, CvType.CV_8UC1)
            Imgproc.rectangle(mask3, Rect(50, 50, 200, 200), Scalar(255.0), -1)

            val out3 = accumulator.accumulateAndBlend(frame3, mask3)
            try {
                // Check sample pixel inside the car region (100, 100)
                val pixel = out3.get(100, 100)
                // Pixel in out3 should be GREEN (0, 255, 0), NOT Red car
                assertEquals(0.0, pixel[0], 5.0)   // B/R channel
                assertEquals(255.0, pixel[1], 5.0) // G channel
                assertTrue(accumulator.hasAccumulatedBackground)
            } finally {
                out3.release()
                frame1.release()
                mask1.release()
                frame2.release()
                mask2.release()
                frame3.release()
                mask3.release()
            }
        } finally {
            accumulator.close()
        }
    }

    @Test
    fun testReset_clearsState() {
        val accumulator = TemporalBackgroundAccumulator()
        try {
            val frame = Mat(100, 100, CvType.CV_8UC4, Scalar(50.0, 50.0, 50.0, 255.0))
            val mask = Mat.zeros(100, 100, CvType.CV_8UC1)
            val out = accumulator.accumulateAndBlend(frame, mask)
            out.release()
            frame.release()
            mask.release()

            assertTrue(accumulator.hasAccumulatedBackground)
            accumulator.reset()
            assertTrue(!accumulator.hasAccumulatedBackground)
        } finally {
            accumulator.close()
        }
    }
}
