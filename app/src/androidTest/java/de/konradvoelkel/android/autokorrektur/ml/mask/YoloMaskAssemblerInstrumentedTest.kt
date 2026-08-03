package de.konradvoelkel.android.autokorrektur.ml.mask

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SmallTest
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.nio.ByteBuffer
import java.nio.ByteOrder

@RunWith(AndroidJUnit4::class)
@SmallTest
class YoloMaskAssemblerInstrumentedTest : AndroidInstrumentedBaseTest() {

    @Test
    fun testExtractPrototypeMasks() {
        val h = 160
        val w = 160
        val c = 32
        val size = h * w * c
        val buffer = ByteBuffer.allocateDirect(size * 4).order(ByteOrder.nativeOrder())
        val floatBuffer = buffer.asFloatBuffer()
        for (i in 0 until size) {
            floatBuffer.put(i.toFloat())
        }
        buffer.rewind()

        val protoShape = intArrayOf(1, h, w, c)
        val result = YoloMaskAssembler.extractPrototypeMasks(buffer, protoShape)

        assertEquals(size, result.size)
        assertEquals(0f, result[0], 0.001f)
        assertEquals((size - 1).toFloat(), result[size - 1], 0.001f)
    }

    @Test
    fun testDeinterleavePrototypes() {
        val h = 2
        val w = 2
        val c = 2
        // Data: [P0_C0, P0_C1, P1_C0, P1_C1, P2_C0, P2_C1, P3_C0, P3_C1]
        // Let's say: [1, 10, 2, 20, 3, 30, 4, 40]
        val data = floatArrayOf(1f, 10f, 2f, 20f, 3f, 30f, 4f, 40f)
        val protoShape = intArrayOf(1, h, w, c)

        val mats = YoloMaskAssembler.deinterleavePrototypes(data, protoShape)

        assertEquals(2, mats.size)

        // Channel 0 Mat should be [[1, 2], [3, 4]]
        assertEquals(1.0, mats[0].get(0, 0)[0], 0.001)
        assertEquals(2.0, mats[0].get(0, 1)[0], 0.001)
        assertEquals(3.0, mats[0].get(1, 0)[0], 0.001)
        assertEquals(4.0, mats[0].get(1, 1)[0], 0.001)

        // Channel 1 Mat should be [[10, 20], [30, 40]]
        assertEquals(10.0, mats[1].get(0, 0)[0], 0.001)
        assertEquals(20.0, mats[1].get(0, 1)[0], 0.001)
        assertEquals(30.0, mats[1].get(1, 0)[0], 0.001)
        assertEquals(40.0, mats[1].get(1, 1)[0], 0.001)

        mats.forEach { it.release() }
    }

    @Test
    fun testApplySigmoid() {
        val mat = Mat(1, 1, CvType.CV_32FC1)
        mat.put(0, 0, floatArrayOf(0f)) // sigmoid(0) = 0.5
        YoloMaskAssembler.applySigmoid(mat)
        assertEquals(0.5, mat.get(0, 0)[0], 0.001)

        mat.put(0, 0, floatArrayOf(100f)) // sigmoid(100) approx 1.0
        YoloMaskAssembler.applySigmoid(mat)
        assertEquals(1.0, mat.get(0, 0)[0], 0.001)

        mat.put(0, 0, floatArrayOf(-100f)) // sigmoid(-100) approx 0.0
        YoloMaskAssembler.applySigmoid(mat)
        assertEquals(0.0, mat.get(0, 0)[0], 0.001)

        mat.release()
    }

    @Test
    fun testAssembleMaskFromPrototypes() {
        // Simple case: 1 prototype (all 1s), coeff 1.0, box (0,0,1,1)
        val h = 10
        val w = 10
        val protoMat = Mat.ones(h, w, CvType.CV_32FC1)
        val protos = listOf(protoMat)
        val coeffs = floatArrayOf(2.0f) // logit 2.0 -> sigmoid(2.0) = 0.88 -> threshold 0.4 -> 1.0

        val mask = YoloMaskAssembler.assembleMaskFromPrototypes(
            coeffs, protos, 0f, 0f, 1f, 1f, 1f, 10, 10
        )

        assertNotNull(mask)
        assertFalse(mask.empty())
        // Resulting mask should be CV_8UC1 and all 255 (due to OPENCV_BYTE_SCALE=255.0 and thresholding)
        assertEquals(CvType.CV_8UC1, mask.type())
        assertEquals(255.0, mask.get(0, 0)[0], 0.001)

        mask.release()
        protoMat.release()
    }
}
