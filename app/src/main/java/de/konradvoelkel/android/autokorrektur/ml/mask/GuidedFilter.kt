package de.konradvoelkel.android.autokorrektur.ml.mask

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * High-performance O(1) edge-preserving Guided Filter for segmentation mask boundary refinement.
 *
 * Uses the guidance image (RGB photo) to transfer high-frequency structural edges to the
 * continuous/binary segmentation mask, snapping boundaries cleanly to car contours and
 * eliminating halos.
 */
object GuidedFilter {

    private const val DEFAULT_EPSILON = 0.04 // Regularization parameter
    private const val BASE_RADIUS_SCALE = 6.0
    private const val BASE_RESOLUTION = 640.0
    private const val MIN_RADIUS = 3

    /**
     * Calculates dynamic filter radius proportional to input image dimensions.
     */
    fun calculateDynamicRadius(width: Int, height: Int): Int {
        val maxDim = kotlin.math.max(width, height)
        val scaled = (maxDim / BASE_RESOLUTION) * BASE_RADIUS_SCALE
        return scaled.toInt().coerceAtLeast(MIN_RADIUS)
    }

    /**
     * Applies guided filter refinement to a single-channel mask using a guide image.
     *
     * @param guide Guidance image (CV_8UC3, CV_8UC4, or CV_8UC1).
     * @param srcMask Input binary or grayscale mask (CV_8UC1).
     * @param radius Filter radius in pixels. If <= 0, automatically computed from dimensions.
     * @param eps Regularization parameter preventing division by zero in flat regions.
     * @return Refined mask (CV_8UC1). Caller is responsible for releasing returned Mat.
     */
    fun filter(
        guide: Mat,
        srcMask: Mat,
        radius: Int = 0,
        eps: Double = DEFAULT_EPSILON
    ): Mat {
        require(!guide.empty()) { "Guide image must not be empty" }
        require(!srcMask.empty()) { "Source mask must not be empty" }

        val r = if (radius > 0) radius else calculateDynamicRadius(guide.cols(), guide.rows())
        val ksize = Size((2 * r + 1).toDouble(), (2 * r + 1).toDouble())

        val matsToRelease = mutableListOf<Mat>()

        try {
            // 1. Prepare Guidance Image (Convert to CV_32FC1 grayscale normalized 0..1)
            val guideGray32F = Mat().also { matsToRelease.add(it) }
            val scale = if (guide.depth() == CvType.CV_32F) 1.0 else 1.0 / 255.0
            when (guide.channels()) {
                4 -> {
                    val gray = Mat().also { matsToRelease.add(it) }
                    Imgproc.cvtColor(guide, gray, Imgproc.COLOR_RGBA2GRAY)
                    gray.convertTo(guideGray32F, CvType.CV_32FC1, scale)
                }
                3 -> {
                    val gray = Mat().also { matsToRelease.add(it) }
                    Imgproc.cvtColor(guide, gray, Imgproc.COLOR_RGB2GRAY)
                    gray.convertTo(guideGray32F, CvType.CV_32FC1, scale)
                }
                else -> {
                    guide.convertTo(guideGray32F, CvType.CV_32FC1, scale)
                }
            }

            // 2. Prepare Source Mask (Convert to CV_32FC1 normalized 0..1)
            val mask32F = Mat().also { matsToRelease.add(it) }
            if (srcMask.channels() > 1) {
                val grayMask = Mat().also { matsToRelease.add(it) }
                val code = if (srcMask.channels() == 4) Imgproc.COLOR_RGBA2GRAY else Imgproc.COLOR_RGB2GRAY
                Imgproc.cvtColor(srcMask, grayMask, code)
                grayMask.convertTo(mask32F, CvType.CV_32FC1, 1.0 / 255.0)
            } else {
                srcMask.convertTo(mask32F, CvType.CV_32FC1, 1.0 / 255.0)
            }

            // 3. Step 1: Means of I and p
            val meanI = Mat().also { matsToRelease.add(it) }
            val meanP = Mat().also { matsToRelease.add(it) }
            Imgproc.boxFilter(guideGray32F, meanI, CvType.CV_32FC1, ksize)
            Imgproc.boxFilter(mask32F, meanP, CvType.CV_32FC1, ksize)

            // 4. Step 2: Autocorrelation and Cross-correlation
            val II = Mat().also { matsToRelease.add(it) }
            val Ip = Mat().also { matsToRelease.add(it) }
            Core.multiply(guideGray32F, guideGray32F, II)
            Core.multiply(guideGray32F, mask32F, Ip)

            val meanII = Mat().also { matsToRelease.add(it) }
            val meanIp = Mat().also { matsToRelease.add(it) }
            Imgproc.boxFilter(II, meanII, CvType.CV_32FC1, ksize)
            Imgproc.boxFilter(Ip, meanIp, CvType.CV_32FC1, ksize)

            // 5. Step 3: Variance of I and Covariance of (I, p)
            val meanIMeanI = Mat().also { matsToRelease.add(it) }
            Core.multiply(meanI, meanI, meanIMeanI)
            val varI = Mat().also { matsToRelease.add(it) }
            Core.subtract(meanII, meanIMeanI, varI)

            val meanIMeanP = Mat().also { matsToRelease.add(it) }
            Core.multiply(meanI, meanP, meanIMeanP)
            val covIp = Mat().also { matsToRelease.add(it) }
            Core.subtract(meanIp, meanIMeanP, covIp)

            // 6. Step 4: Linear coefficients a and b
            // a = covIp / (varI + eps)
            val varIPlusEps = Mat().also { matsToRelease.add(it) }
            Core.add(varI, Scalar(eps), varIPlusEps)
            val a = Mat().also { matsToRelease.add(it) }
            Core.divide(covIp, varIPlusEps, a)

            // b = meanP - a * meanI
            val aMeanI = Mat().also { matsToRelease.add(it) }
            Core.multiply(a, meanI, aMeanI)
            val b = Mat().also { matsToRelease.add(it) }
            Core.subtract(meanP, aMeanI, b)

            // 7. Step 5: Means of a and b
            val meanA = Mat().also { matsToRelease.add(it) }
            val meanB = Mat().also { matsToRelease.add(it) }
            Imgproc.boxFilter(a, meanA, CvType.CV_32FC1, ksize)
            Imgproc.boxFilter(b, meanB, CvType.CV_32FC1, ksize)

            // 8. Step 6: Filtered output q = meanA * I + meanB
            val meanAI = Mat().also { matsToRelease.add(it) }
            Core.multiply(meanA, guideGray32F, meanAI)
            val q = Mat().also { matsToRelease.add(it) }
            Core.add(meanAI, meanB, q)

            // 9. Threshold and convert back to CV_8UC1 (0..255)
            val result8U = Mat()
            // Binary threshold at 0.5 probability
            Imgproc.threshold(q, q, 0.5, 1.0, Imgproc.THRESH_BINARY)
            q.convertTo(result8U, CvType.CV_8U, 255.0)

            return result8U
        } finally {
            matsToRelease.forEach { it.release() }
        }
    }
}
