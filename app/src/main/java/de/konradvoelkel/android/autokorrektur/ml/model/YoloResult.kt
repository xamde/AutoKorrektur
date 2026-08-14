package de.konradvoelkel.android.autokorrektur.ml.model

import org.opencv.core.Mat

/**
 * Result of a YOLO inference pass.
 * - [mask]: subtractive overlay mask (CV_8UC1) aligned to the requested output (post crop/resize).
 * - [detections]: kept detections after score filtering and NMS.
 */
data class YoloResult(
    val mask: Mat,
    val detections: List<Detection>,
    val warnings: List<String> = emptyList()
) : AutoCloseable {
    fun release() {
        try {
            mask.release()
        } catch (_: Exception) {}
    }

    override fun close() {
        release()
    }
}
