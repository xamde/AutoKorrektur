package de.konradvoelkel.android.autokorrektur.ml.api

import org.opencv.core.Mat

/** Public facade for YOLO inference used by the app. */
interface YoloService {
    fun initialize(modelName: String = "yolo11s", useFP16: Boolean = false)

    /**
     * Runs inference and returns an overlay mask Mat (CV_8UC1) aligned to the model input size.
     * The returned Mat contains 255 for background and lower values for masked areas (subtractive mask),
     * matching the legacy pipeline behavior.
     */
    fun infer(
        transformedMat: Mat,
        xRatio: Float,
        yRatio: Float,
        upscaleFactor: Float = 1.0f,
        originalWidth: Int? = null,
        originalHeight: Int? = null
    ): Mat

    fun close()
}