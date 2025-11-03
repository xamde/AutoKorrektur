package de.konradvoelkel.android.autokorrektur.ml

import android.content.Context
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import io.mockk.Runs
import io.mockk.every
import io.mockk.just
import io.mockk.mockkObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.mockito.Mockito
import java.nio.ByteBuffer
import java.nio.ByteOrder

class YoloInferenceParseDetectionsTest {

    private lateinit var appContext: Context
    private lateinit var yolo: YoloInferenceTFLite

    @Before
    fun setUp() {
        // Mock AppLogger to avoid android.util.Log calls during JVM tests
        mockkObject(AppLogger)
        every { AppLogger.debug(any(), any()) } just Runs
        every { AppLogger.info(any(), any()) } just Runs
        every { AppLogger.warn(any(), any()) } just Runs
        every { AppLogger.error(any(), any()) } just Runs

        appContext = Mockito.mock(Context::class.java)
        yolo = YoloInferenceTFLite(appContext)
    }

    @Test
    fun parseDetections_returnsExpectedDetections_forSmallSyntheticBuffer() {
        // Given a tiny synthetic output with feature-major layout [features, proposals]
        val numProposals = 2
        val numClasses = 80
        val numBBoxCoords = 4
        val numMaskCoeffs = 32
        val featuresPerProposal = numBBoxCoords + numClasses + numMaskCoeffs // 116

        val floats = FloatArray(featuresPerProposal * numProposals) { 0f }

        // Proposal 0 (index i=0): a car (classId 2) with high logit so sigmoid > threshold
        val i0 = 0
        val cx0 = 0.5f
        val cy0 = 0.5f
        val w0 = 0.2f
        val h0 = 0.1f
        // Place bbox at features x proposals: value at index = featureIndex * P + i
        @Suppress("KotlinConstantConditions")
        floats[0 * numProposals + i0] = cx0
        floats[1 * numProposals + i0] = cy0
        floats[2 * numProposals + i0] = w0
        floats[3 * numProposals + i0] = h0
        // Class logits: set class 2 high (e.g., 3.0 -> sigmoid ~0.95), others low
        for (c in 0 until numClasses) {
            val featureIndex = numBBoxCoords + c
            val logit = if (c == 2) 3.0f else -5.0f
            floats[featureIndex * numProposals + i0] = logit
        }
        // Mask coeffs (not used in this test's expectations)
        for (j in 0 until numMaskCoeffs) {
            val featureIndex = numBBoxCoords + numClasses + j
            floats[featureIndex * numProposals + i0] = 0.0f
        }

        // Proposal 1 (index i=1): set low scores so it gets filtered out
        val i1 = 1
        @Suppress("KotlinConstantConditions")
        floats[0 * numProposals + i1] = 0.1f
        floats[1 * numProposals + i1] = 0.1f
        floats[2 * numProposals + i1] = 0.05f
        floats[3 * numProposals + i1] = 0.05f
        for (c in 0 until numClasses) {
            val featureIndex = numBBoxCoords + c
            floats[featureIndex * numProposals + i1] = -5.0f
        }
        for (j in 0 until numMaskCoeffs) {
            val featureIndex = numBBoxCoords + numClasses + j
            floats[featureIndex * numProposals + i1] = 0.0f
        }

        val buffer = ByteBuffer.allocateDirect(floats.size * 4).order(ByteOrder.nativeOrder())
        buffer.asFloatBuffer().put(floats)

        // When invoking the private parseDetections via reflection
        val method = YoloInferenceTFLite::class.java.getDeclaredMethod(
            "parseDetections",
            ByteBuffer::class.java,
            Float::class.javaPrimitiveType,
            Int::class.javaPrimitiveType,
            Int::class.javaPrimitiveType,
            Int::class.javaPrimitiveType
        )
        method.isAccessible = true

        @Suppress("UNCHECKED_CAST")
        val result = method.invoke(
            yolo,
            buffer,
            0.6f, // threshold
            numProposals,
            featuresPerProposal,
            numClasses
        ) as List<Detection>

        // Then: only proposal 0 should remain as a vehicle class (car=2)
        assertEquals(1, result.size)
        val det = result[0]
        assertEquals(cx0 - w0 / 2f, det.x, 1e-6f)
        assertEquals(cy0 - h0 / 2f, det.y, 1e-6f)
        assertEquals(w0, det.width, 1e-6f)
        assertEquals(h0, det.height, 1e-6f)
        assertEquals(2, det.classId)
        assertTrue("confidence should be > 0.6", det.confidence > 0.6f)
    }
}
