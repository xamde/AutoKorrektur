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
    private var backgroundValidityMat: Mat? = null
    private val alphaMovingAvg = 0.85

    /**
     * Whether clean background pixels have been accumulated into the buffer.
     */
    val hasAccumulatedBackground: Boolean
        get() = backgroundMat != null

    /**
     * Replaces detected vehicle pixels using temporal background accumulation
     * merged with fast on-device texture inpainting for unobserved regions,
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

            // Transparent overlay: initialize with all zeros (alpha=0 everywhere)
            val transparentOverlay = Mat.zeros(height, width, CvType.CV_8UC4)

            val activeBg: Mat
            val activeBgValid: Mat
            val currentBg = backgroundMat
            val currentBgValid = backgroundValidityMat
            if (currentBg == null || currentBgValid == null || currentBg.cols() != width || currentBg.rows() != height) {
                currentBg?.release()
                currentBgValid?.release()
                activeBg = Mat.zeros(height, width, CvType.CV_8UC4).also { backgroundMat = it }
                activeBgValid = Mat.zeros(height, width, CvType.CV_8UC1).also { backgroundValidityMat = it }
            } else {
                activeBg = currentBg
                activeBgValid = currentBgValid
            }

            // 2. Update persistent background plate with clean pixels from current frame
            val cleanPixelCount = Core.countNonZero(cleanMask)
            if (cleanPixelCount > 0) {
                // Copy clean pixels into persistent background plate
                val currentCleanRgba = Mat()
                frameMat.copyTo(currentCleanRgba, cleanMask)
                currentCleanRgba.copyTo(activeBg, cleanMask)
                activeBgValid.setTo(Scalar(255.0), cleanMask)
                currentCleanRgba.release()
            }

            val carPixelCount = Core.countNonZero(carMask8U)
            if (carPixelCount == 0) {
                return transparentOverlay
            }

            // 3. Synthesize vehicle removal patch
            val inpaintedPatchRgba = Mat.zeros(height, width, CvType.CV_8UC4)
            val bgrFrame = Mat()
            val inpaintedBgr = Mat()

            try {
                if (frameMat.channels() == 4) {
                    Imgproc.cvtColor(frameMat, bgrFrame, Imgproc.COLOR_RGBA2BGR)
                } else {
                    frameMat.copyTo(bgrFrame)
                }

                // Fast downscaled inpainting for remaining non-accumulated regions
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

                Imgproc.cvtColor(inpaintedBgr, inpaintedPatchRgba, Imgproc.COLOR_BGR2RGBA)

                // Blend: if we have temporally accumulated background for this region, use it
                val validCarHoles = Mat()
                Core.bitwise_and(carMask8U, activeBgValid, validCarHoles)
                val validCarPixelCount = Core.countNonZero(validCarHoles)

                if (validCarPixelCount > 0) {
                    activeBg.copyTo(inpaintedPatchRgba, validCarHoles)
                }
                validCarHoles.release()

                // Copy synthesized patch ONLY where carMask8U is non-zero into transparentOverlay
                inpaintedPatchRgba.copyTo(transparentOverlay, carMask8U)
            } finally {
                bgrFrame.release()
                inpaintedBgr.release()
                inpaintedPatchRgba.release()
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
        backgroundValidityMat?.release()
        backgroundValidityMat = null
    }

    override fun close() {
        reset()
    }
}
