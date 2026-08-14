package de.konradvoelkel.android.autokorrektur.ml.factory

import android.content.Context
import de.konradvoelkel.android.autokorrektur.ml.InpaintingEngine
import de.konradvoelkel.android.autokorrektur.ml.LamaInference
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference

/**
 * Factory for creating configured [InpaintingEngine] instances.
 */
object InpaintingEngineFactory {

    /**
     * Creates an on-device [InpaintingEngine] corresponding to the requested [modelType].
     *
     * @param context Application context.
     * @param modelType Target model architecture (MI-GAN or LaMa).
     * @return Configured InpaintingEngine instance.
     */
    fun createEngine(
        context: Context,
        modelType: InpaintingModelType = InpaintingModelType.MIGAN
    ): InpaintingEngine {
        return when (modelType) {
            InpaintingModelType.LAMA -> LamaInference(context)
            else -> MiGanInference(context)
        }
    }
}
