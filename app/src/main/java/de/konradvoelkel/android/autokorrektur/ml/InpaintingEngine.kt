package de.konradvoelkel.android.autokorrektur.ml

import org.opencv.core.Mat
import java.io.IOException

/**
 * Interface for inpainting engines (e.g., Mi-GAN).
 */
interface InpaintingEngine {
    @Throws(IOException::class)
    suspend fun initialize()

    @Throws(IOException::class)
    suspend fun inferMiGan(imageMat: Mat, maskMat: Mat): Mat

    fun close()
}
