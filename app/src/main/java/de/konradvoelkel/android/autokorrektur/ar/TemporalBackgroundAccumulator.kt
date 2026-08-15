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
 * Generates transparent overlay patches (alpha=0 for background, alpha=255 for inpainted car region)
 * to allow the native CameraX preview to run at buttery-smooth 30-60 FPS passthrough.
 */
class TemporalBackgroundAccumulator : AutoCloseable {

    private var backgroundMat: Mat? = null

    /**
     * Whether clean background pixels have been accumulated into the buffer.
     */
    val hasAccumulatedBackground: Boolean
        get() = backgroundMat != null

    /**
     * Replaces detected vehicle pixels with instant on-device texture inpainting,
     * returning a transparent RGBA Mat (alpha=255 over car, alpha=0 elsewhere).
     *
     * @param frameMat Current camera frame RGBA matrix.
     * @param maskMat Binary mask matrix (255 for vehicle pixels, 0 for background).
     * @return Transparent RGBA matrix containing ONLY the inpainted vehicle patch (caller must release).
     */
    @Synchronized
    fun accumulateAndBlend(frameMat: Mat, maskMat: Mat): Mat {
        if (frameMat.empty()) return Mat.zeros(frameMat.rows(), frameMat.cols(), CvType.CV_8UC4)

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

            // Transparent overlay: initialize with all zeros (alpha=0 everywhere)
            val transparentOverlay = Mat.zeros(height, width, CvType.CV_8UC4)

            // If no car is detected, return completely transparent overlay immediately
            if (carPixelCount == 0) {
                return transparentOverlay
            }

            // 2. Perform fast downscaled inpainting for the car region
            val bgrFrame = Mat()
            val inpaintedBgr = Mat()
            try {
                if (frameMat.channels() == 4) {
                    Imgproc.cvtColor(frameMat, bgrFrame, Imgproc.COLOR_RGBA2BGR)
                } else {
                    frameMat.copyTo(bgrFrame)
                }

                // Fast downscaled inpainting for minimal latency
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
                Imgproc.cvtColor(inpaintedBgr, inpaintedRgba, Imgproc.COLOR_BGR2RGBA)

                // Copy inpainted pixels ONLY where carMask8U is non-zero into transparentOverlay
                inpaintedRgba.copyTo(transparentOverlay, carMask8U)
                inpaintedRgba.release()
            } finally {
                bgrFrame.release()
                inpaintedBgr.release()
            }

            transparentOverlay
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
