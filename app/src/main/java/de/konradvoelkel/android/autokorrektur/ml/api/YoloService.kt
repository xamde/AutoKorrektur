package de.konradvoelkel.android.autokorrektur.ml.api

import de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig
import de.konradvoelkel.android.autokorrektur.ml.model.YoloResult
import org.opencv.core.Mat

/** Public facade for YOLO inference used by the app. */
interface YoloService {
    fun initialize(modelName: String = "yolo11s", useFP16: Boolean = false, config: YoloConfig = YoloConfig())

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

    /**
     * Runs inference and returns both the subtractive mask and the kept detections.
     * Optional per-call [overrideConfig] can tweak thresholds/classes; when null the initialized config is used.
     */
    fun inferDetailed(
        transformedMat: Mat,
        xRatio: Float,
        yRatio: Float,
        upscaleFactor: Float = 1.0f,
        originalWidth: Int? = null,
        originalHeight: Int? = null,
        overrideConfig: YoloConfig? = null
    ): YoloResult

    fun close()
}