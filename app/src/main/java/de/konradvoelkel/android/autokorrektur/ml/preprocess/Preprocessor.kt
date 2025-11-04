@file:Suppress("KDocUnresolvedReference")

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
     * - [forEngine]: 8UC3 RGB Mat sized to [targetW]x[targetH]
     * - [forBitmap]: 8UC3 RGB Mat sized to [targetW]x[targetH] (for UI display)
     * - [xRatio]/[yRatio]: scale factors used during letterboxing
     */
    fun prepare(rgbMat: Mat, targetW: Int, targetH: Int): PreprocessResult
}

/** Result of a preprocessing step. */
data class PreprocessResult(
    val forEngine: Mat,  // CV_8UC3
    val forBitmap: Mat,  // CV_8UC3 (same size as forEngine)
    val xRatio: Float,
    val yRatio: Float
)
