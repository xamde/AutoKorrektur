package de.konradvoelkel.android.autokorrektur.ml.preprocess

import org.opencv.core.Mat

/**
 * Contract for preparing an RGB Mat for the YOLO model.
 * Implementations should not hold native resources; all Mats returned are owned by the caller.
 */
interface Preprocessor {
    /**
     * Prepares [rgbMat] for the model with the given input dimensions.
     * Returns a [PreprocessResult] that includes:
     * - [PreprocessResult.forEngine]: 8UC3 RGB Mat sized to [targetW]x[targetH]
     * - [PreprocessResult.forBitmap]: 8UC3 RGB Mat sized to [targetW]x[targetH] (for UI display)
     * - [PreprocessResult.xRatio]/[PreprocessResult.yRatio]: scale factors used during letterboxing
     */
    fun prepare(rgbMat: Mat, targetW: Int, targetH: Int): PreprocessResult
}

/** Result of a preprocessing step. */
data class PreprocessResult(
    val forEngine: Mat,  // CV_8UC3
    val forBitmap: Mat,  // CV_8UC3 (same size as forEngine)
    val xRatio: Float,
    val yRatio: Float
) : AutoCloseable {
    fun release() {
        try {
            if (forEngine !== forBitmap) {
                forEngine.release()
                forBitmap.release()
            } else {
                forEngine.release()
            }
        } catch (_: Exception) {}
    }

    override fun close() {
        release()
    }
}
