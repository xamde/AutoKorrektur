package de.konradvoelkel.android.autokorrektur.ml.post

import de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig
import de.konradvoelkel.android.autokorrektur.ml.model.Detection
import java.nio.ByteBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Pure Kotlin post-processing utilities for YOLO segmentation models.
 *
 * These functions mirror the behavior of the legacy implementations inside
 * YoloInferenceTFLite and are intentionally conservative to preserve behavior.
 */
object YoloPostprocessor {

    /**
     * Parse detections from a TFLite output buffer with feature-major layout:
     * shape [1, featuresPerProposal, numProposals].
     *
     * features layout per proposal: [cx, cy, w, h, class0..classN-1, maskCoeffs(32)]
     */
    fun parseDetections(
        buffer: ByteBuffer,
        numProposals: Int,
        @Suppress("UNUSED_PARAMETER") featuresPerProposal: Int,
        numClasses: Int,
        scoreThreshold: Float = 0.6f,
        allowedClassIndices: IntArray = intArrayOf(2, 3, 5, 7)
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        buffer.rewind()

        val numBBoxCoords = 4
        val numMaskCoeffs = 32

        // Read all floats once
        val floatArray = FloatArray(buffer.capacity() / 4)
        buffer.asFloatBuffer().get(floatArray)

        for (i in 0 until numProposals) {
            val cx = floatArray[i]
            val cy = floatArray[1 * numProposals + i]
            val w = floatArray[2 * numProposals + i]
            val h = floatArray[3 * numProposals + i]

            // Find best class (after sigmoid)
            var maxProb = 0f
            var bestClass = -1
            for (classId in 0 until numClasses) {
                val raw = floatArray[(numBBoxCoords + classId) * numProposals + i]
                val prob = (1f / (1f + exp(-raw.toDouble()))).toFloat()
                if (prob > maxProb) {
                    maxProb = prob
                    bestClass = classId
                }
            }

            // Mask coeffs
            val coeffs = FloatArray(numMaskCoeffs)
            for (j in 0 until numMaskCoeffs) {
                coeffs[j] = floatArray[(numBBoxCoords + numClasses + j) * numProposals + i]
            }

            if (maxProb > scoreThreshold && allowedClassIndices.contains(bestClass)) {
                val xMin = (cx - (w / 2f))
                val yMin = (cy - (h / 2f))
                detections.add(
                    Detection(
                        x = xMin,
                        y = yMin,
                        width = w,
                        height = h,
                        confidence = maxProb,
                        classId = bestClass,
                        maskCoefficients = coeffs
                    )
                )
            }
        }
        return detections
    }

    /**
     * Non-maximum suppression by IoU threshold, then per-class top-K filtering.
     */
    fun applyNMS(
        detections: List<Detection>,
        iouThreshold: Float = 0.9f,
        topAmountPerClass: Int = 100,
        numClasses: Int = 80
    ): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        val sorted = detections.sortedByDescending { it.confidence }
        val keep = mutableListOf<Int>()
        val suppressed = BooleanArray(sorted.size)

        for (i in sorted.indices) {
            if (suppressed[i]) continue
            keep.add(i)
            val a = sorted[i]
            for (j in i + 1 until sorted.size) {
                if (suppressed[j]) continue
                val b = sorted[j]
                val iou = calculateIoU(a, b)
                if (iou > iouThreshold) suppressed[j] = true
            }
        }

        val result = mutableListOf<Detection>()
        val classCounts = IntArray(numClasses)
        for (idx in keep) {
            val d = sorted[idx]
            if (d.classId in 0 until numClasses && classCounts[d.classId] < topAmountPerClass) {
                result.add(d)
                classCounts[d.classId]++
            }
        }
        return result
    }

    /**
     * Intersection over Union for boxes represented as (x,y,width,height) in normalized 0..1.
     */
    fun calculateIoU(a: Detection, b: Detection): Float {
        val xA = max(a.x, b.x)
        val yA = max(a.y, b.y)
        val xB = min(a.x + a.width, b.x + b.width)
        val yB = min(a.y + a.height, b.y + b.height)

        val interW = max(0f, xB - xA)
        val interH = max(0f, yB - yA)
        val interArea = interW * interH
        if (interArea <= 0f) return 0f

        val areaA = a.width * a.height
        val areaB = b.width * b.height
        val union = areaA + areaB - interArea
        return if (union > 0.000001f) interArea / union else 0f
    }

    /** Convenience wrapper using a config object. */
    @Suppress("unused")
    fun postprocess(
        detectionsBuffer: ByteBuffer,
        numProposals: Int,
        featuresPerProposal: Int,
        numClasses: Int,
        config: YoloConfig = YoloConfig()
    ): List<Detection> {
        val parsed = parseDetections(
            buffer = detectionsBuffer,
            numProposals = numProposals,
            featuresPerProposal = featuresPerProposal,
            numClasses = numClasses,
            scoreThreshold = config.scoreThreshold,
            allowedClassIndices = config.vehicleClassIndices
        )
        return applyNMS(
            detections = parsed,
            iouThreshold = config.iouThreshold,
            topAmountPerClass = config.topAmountPerClass,
            numClasses = config.labels.size
        )
    }
}