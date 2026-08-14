package de.konradvoelkel.android.autokorrektur.ml

import org.opencv.core.Mat
import java.io.IOException

/**
 * Interface for on-device and remote inpainting engines (e.g., Mi-GAN, SDXL).
 */
interface InpaintingEngine {
    /**
     * Initializes neural network weights and model sessions.
     */
    @Throws(IOException::class)
    suspend fun initialize()

    /**
     * Executes neural inpainting on [imageMat] guided by [maskMat].
     *
     * @param imageMat 8UC3 / 8UC4 source image Mat.
     * @param maskMat 8UC1 subtractive mask Mat (0 = vehicle region to inpaint, 255 = keep background).
     * @return Freshly allocated OpenCV Mat containing the inpainted image (caller must release).
     */
    @Throws(IOException::class)
    suspend fun inpaint(imageMat: Mat, maskMat: Mat): Mat

    /**
     * Deprecated alias for [inpaint].
     */
    @Deprecated("Use inpaint() instead", ReplaceWith("inpaint(imageMat, maskMat)"))
    @Throws(IOException::class)
    suspend fun inferMiGan(imageMat: Mat, maskMat: Mat): Mat = inpaint(imageMat, maskMat)

    /**
     * Releases model sessions and associated native buffers.
     */
    fun close()
}
