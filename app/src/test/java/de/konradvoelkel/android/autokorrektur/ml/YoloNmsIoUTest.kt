package de.konradvoelkel.android.autokorrektur.ml

import de.konradvoelkel.android.autokorrektur.ml.model.Detection
import de.konradvoelkel.android.autokorrektur.ml.post.YoloPostprocessor
import org.junit.Assert.assertEquals
import org.junit.Test

class YoloNmsIoUTest {

    @Test
    fun applyNms_keepsHigherConfidenceWhenIoUAboveThreshold() {
        val detHigh = Detection(
            x = 0.4f, y = 0.4f, width = 0.3f, height = 0.3f,
            confidence = 0.90f, classId = 2, maskCoefficients = FloatArray(32)
        )
        val detLow = Detection(
            x = 0.42f, y = 0.42f, width = 0.3f, height = 0.3f,
            confidence = 0.80f, classId = 2, maskCoefficients = FloatArray(32)
        )

        val detections = listOf(detHigh, detLow)
        val kept = YoloPostprocessor.applyNMS(
            detections = detections,
            iouThreshold = 0.5f,
            topAmountPerClass = 100,
            numClasses = 80
        )

        assertEquals(1, kept.size)
        assertEquals(detHigh, kept.first())
    }

    @Test
    fun calculateIoU_matchesExpectedForSimpleBoxes() {
        val detA = Detection(0.0f, 0.0f, 1.0f, 1.0f, 0.9f, 2, FloatArray(32))
        val detB = Detection(0.5f, 0.5f, 1.0f, 1.0f, 0.8f, 2, FloatArray(32))
        val iou = YoloPostprocessor.calculateIoU(detA, detB)
        // Overlap is a 0.5x0.5 square = 0.25; union = 1 + 1 - 0.25 = 1.75; IoU = 0.25/1.75 ≈ 0.142857
        assertEquals(0.25f / 1.75f, iou, 1e-5f)
    }
}
