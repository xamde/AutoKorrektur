package de.konradvoelkel.android.autokorrektur.ml

import android.net.Uri
import java.io.IOException

/**
 * Service for loading and preprocessing images for ML inference.
 */
interface ImagePreprocessingService {
    /**
     * Processes an input image URI for ML inference.
     * @return A [ImageProcessor.ProcessedImage] containing all required data.
     * @throws IOException if image loading fails.
     */
    @Throws(IOException::class)
    fun processInputImage(
        imageUri: Uri,
        modelWidth: Int,
        modelHeight: Int,
        downscaleMp: Float? = null
    ): ImageProcessor.ProcessedImage
}
