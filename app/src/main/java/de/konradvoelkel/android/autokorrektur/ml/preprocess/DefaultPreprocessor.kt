package de.konradvoelkel.android.autokorrektur.ml.preprocess

import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Default implementation of [Preprocessor].
 *
 * Inputs are expected to be RGB Mats (CV_8UC3). The output Mats are also CV_8UC3 sized to the
 * model input dimensions. Letterboxing is applied by padding to a square before the final resize.
 */
class DefaultPreprocessor(
    private val stride: Int = 32
) : Preprocessor {

    override fun prepare(rgbMat: Mat, targetW: Int, targetH: Int): PreprocessResult {
        // 1) Make dimensions divisible by stride to preserve downsample alignment
        val (w, h) = ImageProcessingUtils.divStride(stride, rgbMat.cols(), rgbMat.rows())
        val resizedMat = Mat()
        Imgproc.resize(
            rgbMat,
            resizedMat,
            Size(w.toDouble(), h.toDouble()),
            0.0,
            0.0,
            Imgproc.INTER_LANCZOS4
        )

        // 2) Pad to square (letterbox) with black borders
        val pr =
            ImageProcessingUtils.computeSquarePaddingAndRatios(resizedMat.cols(), resizedMat.rows())
        val paddedMat = Mat()
        Core.copyMakeBorder(
            resizedMat,
            paddedMat,
            0,
            pr.yPad,
            0,
            pr.xPad,
            Core.BORDER_CONSTANT,
            Scalar(0.0, 0.0, 0.0)
        )
        resizedMat.release()

        // 3) Resize to final model input resolution (keep 8-bit for engine)
        val final8U = Mat()
        Imgproc.resize(
            paddedMat,
            final8U,
            Size(targetW.toDouble(), targetH.toDouble())
        )
        paddedMat.release()

        // Return two separate Mats to avoid aliasing/lifecycle coupling in callers.
        val forEngine = final8U.clone()
        val forBitmap = final8U

        return PreprocessResult(
            forEngine = forEngine,
            forBitmap = forBitmap,
            xRatio = pr.xRatio,
            yRatio = pr.yRatio
        )
    }
}
