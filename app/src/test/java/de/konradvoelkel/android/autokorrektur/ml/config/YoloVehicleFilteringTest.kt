package de.konradvoelkel.android.autokorrektur.ml.config

import de.konradvoelkel.android.autokorrektur.ml.post.YoloPostprocessor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class YoloVehicleFilteringTest {

    @Test
    fun defaultYoloConfig_vehicleClassIndices_matchExpectedCocoVehicleLabels() {
        val config = YoloConfig()
        val indices = config.vehicleClassIndices.toList()
        
        // COCO labels: 2=car, 3=motorcycle, 5=bus, 7=truck
        assertEquals(listOf(2, 3, 5, 7), indices)
        assertEquals("car", config.labels[2])
        assertEquals("motorcycle", config.labels[3])
        assertEquals("bus", config.labels[5])
        assertEquals("truck", config.labels[7])
    }

    @Test
    fun parseDetections_filtersNonVehicleObjects_retainsOnlyVehicles() {
        val config = YoloConfig()
        val numProposals = 5
        val numClasses = 80
        val numBBoxCoords = 4
        val numMaskCoeffs = 32
        val featuresPerProposal = numBBoxCoords + numClasses + numMaskCoeffs

        val floats = FloatArray(featuresPerProposal * numProposals)

        // Function to set up proposal
        fun setupProposal(propIdx: Int, classId: Int, score: Float) {
            floats[0 * numProposals + propIdx] = 0.5f // cx
            floats[1 * numProposals + propIdx] = 0.5f // cy
            floats[2 * numProposals + propIdx] = 0.2f // w
            floats[3 * numProposals + propIdx] = 0.2f // h
            for (c in 0 until numClasses) {
                val featureIndex = numBBoxCoords + c
                // logit corresponding to score via inverse sigmoid: log(s / (1 - s))
                val logit = if (c == classId) kotlin.math.ln(score / (1f - score)) else -10f
                floats[featureIndex * numProposals + propIdx] = logit
            }
        }

        // Proposal 0: Person (class 0) -> should be filtered out
        setupProposal(0, classId = 0, score = 0.95f)
        // Proposal 1: Car (class 2) -> SHOULD BE KEPT
        setupProposal(1, classId = 2, score = 0.90f)
        // Proposal 2: Dog (class 16) -> should be filtered out
        setupProposal(2, classId = 16, score = 0.85f)
        // Proposal 3: Bus (class 5) -> SHOULD BE KEPT
        setupProposal(3, classId = 5, score = 0.88f)
        // Proposal 4: Truck (class 7) -> SHOULD BE KEPT
        setupProposal(4, classId = 7, score = 0.92f)

        val buffer = ByteBuffer.allocateDirect(floats.size * 4).order(ByteOrder.nativeOrder())
        buffer.asFloatBuffer().put(floats)
        buffer.rewind()

        val detections = YoloPostprocessor.parseDetections(
            buffer = buffer,
            numProposals = numProposals,
            featuresPerProposal = featuresPerProposal,
            numClasses = numClasses,
            scoreThreshold = config.scoreThreshold,
            allowedClassIndices = config.vehicleClassIndices
        )

        assertEquals(3, detections.size)
        val detectedClasses = detections.map { it.classId }.toSet()
        assertEquals(setOf(2, 5, 7), detectedClasses)
        assertTrue(detectedClasses.contains(2)) // car
        assertTrue(detectedClasses.contains(5)) // bus
        assertTrue(detectedClasses.contains(7)) // truck
    }
}
