package de.konradvoelkel.android.autokorrektur.ml.post

import de.konradvoelkel.android.autokorrektur.ml.model.Detection
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class YoloPostprocessorTest {

    @Test
    fun calculateIoU_nonOverlapping_returnsZero() {
        val a = Detection(0f, 0f, 0.1f, 0.1f, 0.9f, 2, FloatArray(32))
        val b = Detection(0.5f, 0.5f, 0.1f, 0.1f, 0.8f, 2, FloatArray(32))
        val iou = YoloPostprocessor.calculateIoU(a, b)
        assertEquals(0f, iou, 1e-6f)
    }

    @Test
    fun calculateIoU_identicalBoxes_returnsOne() {
        val a = Detection(0.1f, 0.2f, 0.3f, 0.4f, 0.9f, 2, FloatArray(32))
        val b = Detection(0.1f, 0.2f, 0.3f, 0.4f, 0.8f, 2, FloatArray(32))
        val iou = YoloPostprocessor.calculateIoU(a, b)
        assertEquals(1f, iou, 1e-6f)
    }

    @Test
    fun applyNMS_suppressesHighOverlap_sameClass() {
        val a = Detection(0.1f, 0.1f, 0.4f, 0.4f, 0.95f, 2, FloatArray(32))
        val b = Detection(0.12f, 0.12f, 0.4f, 0.4f, 0.90f, 2, FloatArray(32))
        val c = Detection(0.6f, 0.6f, 0.2f, 0.2f, 0.70f, 2, FloatArray(32))

        val out = YoloPostprocessor.applyNMS(listOf(a, b, c), iouThreshold = 0.5f, topAmountPerClass = 100, numClasses = 80)
        // a and b overlap heavily -> keep only the highest confidence (a), plus c
        assertEquals(2, out.size)
        assertTrue(out.contains(a))
        assertTrue(out.contains(c))
    }

    @Test
    fun parseDetections_smallSyntheticBuffer_producesExpectedDetection() {
        val numProposals = 2
        val numClasses = 3
        val numMask = 32
        val featuresPerProposal = 4 + numClasses + numMask
        val totalFloats = featuresPerProposal * numProposals
        val floats = FloatArray(totalFloats) { 0f }

        // Feature-major layout: [cx, cy, w, h, class0..classN-1, maskCoeff0..31] each as [feature][i]
        fun idx(feature: Int, proposal: Int) = feature * numProposals + proposal

        // Proposal 0: centered box, moderate size, class 1 high score
        floats[idx(0, 0)] = 0.5f // cx
        floats[idx(1, 0)] = 0.5f // cy
        floats[idx(2, 0)] = 0.4f // w
        floats[idx(3, 0)] = 0.3f // h
        // class logits: after sigmoid we want class 1 to be high (> 0.6)
        floats[idx(4 + 0, 0)] = 0.0f     // ~0.5
        floats[idx(4 + 1, 0)] = 1.0f     // ~0.731 -> should pass threshold 0.6
        floats[idx(4 + 2, 0)] = -1.0f    // ~0.269
        // mask coeffs
        for (j in 0 until numMask) {
            floats[idx(4 + numClasses + j, 0)] = j / 31f
        }

        // Proposal 1: low confidence -> filtered out
        floats[idx(0, 1)] = 0.2f
        floats[idx(1, 1)] = 0.2f
        floats[idx(2, 1)] = 0.1f
        floats[idx(3, 1)] = 0.1f
        floats[idx(4 + 0, 1)] = -10f // ~0
        floats[idx(4 + 1, 1)] = -10f // ~0
        floats[idx(4 + 2, 1)] = -10f // ~0
        for (j in 0 until numMask) {
            floats[idx(4 + numClasses + j, 1)] = 0f
        }

        val buffer = ByteBuffer.allocateDirect(totalFloats * 4).order(ByteOrder.nativeOrder())
        buffer.asFloatBuffer().put(floats)
        buffer.rewind()

        val detections = YoloPostprocessor.parseDetections(
            buffer = buffer,
            numProposals = numProposals,
            featuresPerProposal = featuresPerProposal,
            numClasses = numClasses,
            scoreThreshold = 0.6f,
            allowedClassIndices = intArrayOf(1) // Only allow class 1
        )

        assertEquals(1, detections.size)
        val d = detections[0]
        assertEquals(0.5f - 0.2f, d.x)
        assertEquals(0.5f - 0.15f, d.y)
        assertEquals(0.4f, d.width)
        assertEquals(0.3f, d.height)
        assertEquals(1, d.classId)
        assertEquals(32, d.maskCoefficients.size)
    }

    @Test
    fun postprocess_configWrapper_appliesThresholdsAndNms() {
        // Two overlapping high-confidence of same class -> NMS keeps one
        val a = Detection(0.1f, 0.1f, 0.4f, 0.4f, 0.95f, 2, FloatArray(32))
        val b = Detection(0.12f, 0.12f, 0.4f, 0.4f, 0.90f, 2, FloatArray(32))
        val detections = YoloPostprocessor.applyNMS(listOf(a, b), iouThreshold = 0.5f, topAmountPerClass = 1, numClasses = 80)
        assertEquals(1, detections.size)
    }
}