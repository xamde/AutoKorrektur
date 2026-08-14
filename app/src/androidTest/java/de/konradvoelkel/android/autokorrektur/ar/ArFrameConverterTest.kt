package de.konradvoelkel.android.autokorrektur.ar

import androidx.test.ext.junit.runners.AndroidJUnit4
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
class ArFrameConverterTest : AndroidInstrumentedBaseTest() {

    @Test
    fun testRotateMat_rotatesCorrectly() {
        val src = Mat(480, 640, CvType.CV_8UC4, Scalar(10.0, 20.0, 30.0, 255.0))
        try {
            // 90 degrees rotation (landscape to portrait)
            val rot90 = ArFrameConverter.rotateMat(src, 90)
            try {
                assertEquals(480, rot90.cols())
                assertEquals(640, rot90.rows())
            } finally {
                rot90.release()
            }

            // 180 degrees rotation
            val rot180 = ArFrameConverter.rotateMat(src, 180)
            try {
                assertEquals(640, rot180.cols())
                assertEquals(480, rot180.rows())
            } finally {
                rot180.release()
            }

            // 270 degrees rotation
            val rot270 = ArFrameConverter.rotateMat(src, 270)
            try {
                assertEquals(480, rot270.cols())
                assertEquals(640, rot270.rows())
            } finally {
                rot270.release()
            }

            // 0 degrees rotation
            val rot0 = ArFrameConverter.rotateMat(src, 0)
            try {
                assertEquals(640, rot0.cols())
                assertEquals(480, rot0.rows())
            } finally {
                rot0.release()
            }
        } finally {
            src.release()
        }
    }

    @Test
    fun testScaleAndPadForYolo_producesExact640x640() {
        val src = Mat(1080, 1920, CvType.CV_8UC3, Scalar(100.0, 100.0, 100.0))
        try {
            val yoloInput = ArFrameConverter.scaleAndPadForYolo(src, 640)
            try {
                assertNotNull(yoloInput)
                assertEquals(640, yoloInput.cols())
                assertEquals(640, yoloInput.rows())
                assertEquals(CvType.CV_8UC3, yoloInput.type())
            } finally {
                yoloInput.release()
            }
        } finally {
            src.release()
        }
    }
}
