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

    private const val YOLO_INPUT_SIZE = 640f
    private const val NUM_BBOX_COORDS = 4
    private const val NUM_MASK_COEFFS = 32
    private const val BYTES_PER_FLOAT = 4
    private const val DEFAULT_SCORE_THRESHOLD = 0.6f
    private const val MIN_UNION_EPSILON = 0.000001f

    private val DEFAULT_ALLOWED_CLASSES = intArrayOf(2, 3, 5, 7)

    private data class NormalizedBox(
        val x: Float,
        val y: Float,
        val w: Float,
        val h: Float
    )

    /**
     * Parse detections from a TFLite output buffer with feature-major layout:
     * shape [1, featuresPerProposal, numProposals].
     *
     * features layout per proposal: [cx, cy, w, h, class0..classN-1, maskCoeffs(32)]
     */
    @Suppress("LongParameterList")
    fun parseDetections(
        buffer: ByteBuffer,
        numProposals: Int,
        @Suppress("UNUSED_PARAMETER") featuresPerProposal: Int,
        numClasses: Int,
        scoreThreshold: Float = DEFAULT_SCORE_THRESHOLD,
        allowedClassIndices: IntArray = DEFAULT_ALLOWED_CLASSES
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        buffer.rewind()

        // Normalize allowed classes: keep only valid indices; if none valid -> allow all
        val validAllowed: Set<Int> = allowedClassIndices.filter { it in 0 until numClasses }.toSet()
        val allowAll = validAllowed.isEmpty()

        // Read all floats once
        val floatArray = FloatArray(buffer.capacity() / BYTES_PER_FLOAT)
        buffer.asFloatBuffer().get(floatArray)

        for (i in 0 until numProposals) {
            val cx = floatArray[i]
            val cy = floatArray[1 * numProposals + i]
            val w = floatArray[2 * numProposals + i]
            val h = floatArray[3 * numProposals + i]

            // Degenerate boxes are discarded early
            if (w <= 0f || h <= 0f) continue

            val (maxProb, bestClass) = findBestClass(floatArray, i, numProposals, numClasses)

            val coeffs = FloatArray(NUM_MASK_COEFFS)
            for (j in 0 until NUM_MASK_COEFFS) {
                coeffs[j] = floatArray[(NUM_BBOX_COORDS + numClasses + j) * numProposals + i]
            }

            val classAllowed = allowAll || validAllowed.contains(bestClass)
            if (maxProb > scoreThreshold && classAllowed) {
                val box = normalizeBox(cx, cy, w, h)
                detections.add(
                    Detection(
                        x = box.x.coerceIn(0f, 1f),
                        y = box.y.coerceIn(0f, 1f),
                        width = box.w.coerceIn(0f, 1f),
                        height = box.h.coerceIn(0f, 1f),
                        confidence = maxProb,
                        classId = bestClass,
                        maskCoefficients = coeffs
                    )
                )
            }
        }
        return detections
    }

    private fun findBestClass(
        floatArray: FloatArray,
        proposalIdx: Int,
        numProposals: Int,
        numClasses: Int
    ): Pair<Float, Int> {
        var maxProb = 0f
        var bestClass = -1
        for (classId in 0 until numClasses) {
            val raw = floatArray[(NUM_BBOX_COORDS + classId) * numProposals + proposalIdx]
            val prob = (1f / (1f + exp(-raw.toDouble()))).toFloat()
            if (prob > maxProb) {
                maxProb = prob
                bestClass = classId
            }
        }
        return Pair(maxProb, bestClass)
    }

    private fun normalizeBox(cx: Float, cy: Float, w: Float, h: Float): NormalizedBox {
        val halfW = w / 2f
        val halfH = h / 2f
        val normX = if (cx > 1f) (cx - halfW) / YOLO_INPUT_SIZE else (cx - halfW)
        val normY = if (cy > 1f) (cy - halfH) / YOLO_INPUT_SIZE else (cy - halfH)
        val normW = if (w > 1f) w / YOLO_INPUT_SIZE else w
        val normH = if (h > 1f) h / YOLO_INPUT_SIZE else h
        return NormalizedBox(normX, normY, normW, normH)
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
        return if (union > MIN_UNION_EPSILON) interArea / union else 0f
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