package de.konradvoelkel.android.autokorrektur.ar

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar

/**
 * OpenCV Temporal Background Accumulator for Real-Time AR Car Removal.
 * Stores clean background pixels (road, sidewalk, building facade) from un-masked frame regions
 * as the user moves the camera, blending accumulated textures into vehicle mask regions in real-time.
 */
class TemporalBackgroundAccumulator : AutoCloseable {

    private var backgroundMat: Mat? = null

    /**
     * Accumulates clean background pixels from [frameMat] and blends accumulated background
     * into vehicle regions specified by [maskMat].
     *
     * Note: The returned [Mat] is a newly allocated matrix whose native memory must be released
     * by the caller via [Mat.release].
     *
     * @param frameMat Current camera frame BGRA matrix.
     * @param maskMat Binary mask matrix (255 for vehicle pixels, 0 for background).
     * @return Blended BGRA matrix with vehicles replaced by accumulated background (caller must release).
     */
    @Synchronized
    fun accumulateAndBlend(frameMat: Mat, maskMat: Mat): Mat {
        if (frameMat.empty()) return frameMat.clone()

        val width = frameMat.cols()
        val height = frameMat.rows()

        // 1. Create binary non-vehicle mask (cleanMask = 255 - maskMat)
        val cleanMask = Mat()
        try {
            Core.bitwise_not(maskMat, cleanMask)

            // 2. Initialize or resize background accumulation buffer
            var bg = backgroundMat
            if (bg == null || bg.cols() != width || bg.rows() != height || bg.type() != frameMat.type()) {
                bg?.release()
                bg = frameMat.clone()
                backgroundMat = bg
            }

            // 3. Update background buffer with new un-masked pixels
            frameMat.copyTo(bg, cleanMask)

            // 4. Create blended output frame
            val outputMat = frameMat.clone()

            // 5. Replace vehicle pixels in outputMat with accumulated background
            bg.copyTo(outputMat, maskMat)

            return outputMat
        } finally {
            cleanMask.release()
        }
    }

    /**
     * Resets the accumulated background buffer and releases native memory.
     */
    @Synchronized
    fun reset() {
        backgroundMat?.release()
        backgroundMat = null
    }

    override fun close() {
        reset()
    }

    protected fun finalize() {
        if (backgroundMat != null) {
            de.konradvoelkel.android.autokorrektur.utils.AppLogger.warn(
                "TemporalBackgroundAccumulator was not closed before garbage collection. Releasing native Mat."
            )
            reset()
        }
    }
}
