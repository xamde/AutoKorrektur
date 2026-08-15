package de.konradvoelkel.android.autokorrektur.ar

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo

/**
 * OpenCV Temporal Background Accumulator & Real-Time Inpainter for AR Car Removal.
 * Replaces detected vehicle pixels with instant on-device texture inpainting and
 * blends accumulated clean background textures in real-time.
 */
class TemporalBackgroundAccumulator : AutoCloseable {

    private var backgroundMat: Mat? = null

    /**
     * Whether clean background pixels have been accumulated into the buffer.
     */
    val hasAccumulatedBackground: Boolean
        get() = backgroundMat != null

    /**
     * Accumulates clean background pixels from [frameMat] and replaces vehicle regions
     * with real-time inpainting and accumulated background.
     *
     * @param frameMat Current camera frame RGBA matrix.
     * @param maskMat Binary mask matrix (255 for vehicle pixels, 0 for background).
     * @return Blended RGBA matrix with vehicles erased (caller must release).
     */
    @Synchronized
    fun accumulateAndBlend(frameMat: Mat, maskMat: Mat): Mat {
        if (frameMat.empty()) return frameMat.clone()

        val width = frameMat.cols()
        val height = frameMat.rows()

        val cleanMask = Mat()
        val carMask8U = Mat()

        return try {
            // 1. Ensure binary mask format (CV_8UC1)
            if (maskMat.type() != CvType.CV_8UC1) {
                maskMat.convertTo(carMask8U, CvType.CV_8UC1)
            } else {
                maskMat.copyTo(carMask8U)
            }

            Core.bitwise_not(carMask8U, cleanMask)

            val carPixelCount = Core.countNonZero(carMask8U)

            // 2. Initialize or maintain background accumulator
            var bg = backgroundMat
            if (bg == null || bg.cols() != width || bg.rows() != height || bg.type() != frameMat.type()) {
                bg?.release()
                bg = Mat.zeros(height, width, frameMat.type())
                backgroundMat = bg
            }

            // Update background buffer with unmasked clean pixels
            frameMat.copyTo(bg, cleanMask)

            val outputMat = frameMat.clone()

            // 3. If car pixels are detected, inpaint the car region instantly
            if (carPixelCount > 0) {
                val bgrFrame = Mat()
                val inpaintedBgr = Mat()
                try {
                    if (frameMat.channels() == 4) {
                        Imgproc.cvtColor(frameMat, bgrFrame, Imgproc.COLOR_RGBA2BGR)
                    } else {
                        frameMat.copyTo(bgrFrame)
                    }

                    // Fast downscaled inpainting for 30 FPS responsiveness
                    val scale = 0.5
                    val smallW = (width * scale).toInt().coerceAtLeast(1)
                    val smallH = (height * scale).toInt().coerceAtLeast(1)
                    val smallBgr = Mat()
                    val smallMask = Mat()
                    val smallInpainted = Mat()

                    Imgproc.resize(bgrFrame, smallBgr, Size(smallW.toDouble(), smallH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
                    Imgproc.resize(carMask8U, smallMask, Size(smallW.toDouble(), smallH.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)

                    Photo.inpaint(smallBgr, smallMask, smallInpainted, 3.0, Photo.INPAINT_TELEA)

                    Imgproc.resize(smallInpainted, inpaintedBgr, Size(width.toDouble(), height.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)

                    smallBgr.release()
                    smallMask.release()
                    smallInpainted.release()

                    val inpaintedRgba = Mat()
                    if (frameMat.channels() == 4) {
                        Imgproc.cvtColor(inpaintedBgr, inpaintedRgba, Imgproc.COLOR_BGR2RGBA)
                    } else {
                        inpaintedBgr.copyTo(inpaintedRgba)
                    }

                    // Apply inpainting directly onto the vehicle region in outputMat
                    inpaintedRgba.copyTo(outputMat, carMask8U)
                    inpaintedRgba.release()
                } finally {
                    bgrFrame.release()
                    inpaintedBgr.release()
                }
            }

            outputMat
        } finally {
            cleanMask.release()
            carMask8U.release()
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
