package de.konradvoelkel.android.autokorrektur.ml.engine

import de.konradvoelkel.android.autokorrektur.ml.errors.InferenceException
import de.konradvoelkel.android.autokorrektur.ml.errors.ModelLoadException
import de.konradvoelkel.android.autokorrektur.ml.model.RawOutputs
import de.konradvoelkel.android.autokorrektur.ml.model.Shapes
import org.opencv.core.Mat

/**
 * Interface for YOLO engines (e.g., TFLite).
 */
interface YoloEngine {
    val isInitialized: Boolean
    val isClosed: Boolean

    @Throws(ModelLoadException::class)
    suspend fun initialize(modelName: String = "yolo11s", useFP16: Boolean = false)

    @Throws(InferenceException::class)
    fun run(rgbMat: Mat): RawOutputs

    fun shapes(): Shapes

    fun close()
}
