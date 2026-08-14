package de.konradvoelkel.android.autokorrektur.utils

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.log10

/**
 * Utility functions for calculating image quality metrics (PSNR and SSIM).
 */
@Suppress("MagicNumber", "MaxLineLength", "LongMethod")
object ImageQualityMetrics {

    private const val MAX_8BIT_VAL = 255.0
    private const val DEFAULT_PSNR_PERFECT = 100.0
    private const val SSIM_C1 = 6.5025  // (0.01 * 255)^2
    private const val SSIM_C2 = 58.5225 // (0.03 * 255)^2
    private const val GAUSSIAN_SIGMA = 1.5
    private const val KERNEL_SIZE_11 = 11.0

    /**
     * Data class holding benchmark evaluation metrics for an image pair.
     */
    data class MetricsResult(
        val psnrDb: Double,
        val ssim: Double
    )

    /**
     * Calculates PSNR (Peak Signal-to-Noise Ratio) in dB between two OpenCV Mats of identical dimensions.
     */
    fun calculatePsnr(mat1: Mat, mat2: Mat): Double {
        require(mat1.rows() == mat2.rows() && mat1.cols() == mat2.cols()) {
            "Image dimensions must match for PSNR calculation"
        }
        val diff = Mat()
        Core.absdiff(mat1, mat2, diff)

        val diff32f = Mat()
        diff.convertTo(diff32f, CvType.CV_32F)
        diff.release()

        val squared = diff32f.mul(diff32f)
        diff32f.release()

        val scalarS = Core.sumElems(squared)
        squared.release()

        val totalChannels = mat1.channels()
        val totalElements = mat1.rows().toDouble() * mat1.cols().toDouble() * totalChannels.toDouble()

        var sse = scalarS.`val`[0]
        if (totalChannels > 1) sse += scalarS.`val`[1]
        if (totalChannels > 2) sse += scalarS.`val`[2]

        val mse = sse / totalElements
        if (mse <= 1e-10) return DEFAULT_PSNR_PERFECT

        return 10.0 * log10((MAX_8BIT_VAL * MAX_8BIT_VAL) / mse)
    }

    /**
     * Calculates SSIM (Structural Similarity Index Measure) between two OpenCV Mats of identical dimensions.
     * Value ranges from -1.0 to +1.0 (typically 0.0 to 1.0 for image similarity).
     */
    fun calculateSsim(mat1: Mat, mat2: Mat): Double {
        require(mat1.rows() == mat2.rows() && mat1.cols() == mat2.cols()) {
            "Image dimensions must match for SSIM calculation"
        }

        val i1 = Mat()
        val i2 = Mat()
        mat1.convertTo(i1, CvType.CV_32F)
        mat2.convertTo(i2, CvType.CV_32F)

        val i1Sq = i1.mul(i1)
        val i2Sq = i2.mul(i2)
        val i1I2 = i1.mul(i2)

        val mu1 = Mat()
        val mu2 = Mat()
        val kSize = Size(KERNEL_SIZE_11, KERNEL_SIZE_11)
        Imgproc.GaussianBlur(i1, mu1, kSize, GAUSSIAN_SIGMA)
        Imgproc.GaussianBlur(i2, mu2, kSize, GAUSSIAN_SIGMA)

        val mu1Sq = mu1.mul(mu1)
        val mu2Sq = mu2.mul(mu2)
        val mu1Mu2 = mu1.mul(mu2)

        val sigma1Sq = Mat()
        val sigma2Sq = Mat()
        val sigma12 = Mat()

        val tmp1 = Mat()
        Imgproc.GaussianBlur(i1Sq, tmp1, kSize, GAUSSIAN_SIGMA)
        Core.subtract(tmp1, mu1Sq, sigma1Sq)
        tmp1.release()

        val tmp2 = Mat()
        Imgproc.GaussianBlur(i2Sq, tmp2, kSize, GAUSSIAN_SIGMA)
        Core.subtract(tmp2, mu2Sq, sigma2Sq)
        tmp2.release()

        val tmp3 = Mat()
        Imgproc.GaussianBlur(i1I2, tmp3, kSize, GAUSSIAN_SIGMA)
        Core.subtract(tmp3, mu1Mu2, sigma12)
        tmp3.release()

        val t1 = Mat()
        Core.multiply(mu1Mu2, Scalar(2.0, 2.0, 2.0), t1)
        Core.add(t1, Scalar(SSIM_C1, SSIM_C1, SSIM_C1), t1)

        val t2 = Mat()
        Core.multiply(sigma12, Scalar(2.0, 2.0, 2.0), t2)
        Core.add(t2, Scalar(SSIM_C2, SSIM_C2, SSIM_C2), t2)

        val t3 = Mat()
        Core.add(mu1Sq, mu2Sq, t3)
        Core.add(t3, Scalar(SSIM_C1, SSIM_C1, SSIM_C1), t3)

        val t4 = Mat()
        Core.add(sigma1Sq, sigma2Sq, t4)
        Core.add(t4, Scalar(SSIM_C2, SSIM_C2, SSIM_C2), t4)

        val num = t1.mul(t2)
        val den = t3.mul(t4)
        val ssimMap = Mat()
        Core.divide(num, den, ssimMap)

        val mSsim = Core.mean(ssimMap)

        // Cleanup intermediate Mats
        i1.release(); i2.release(); i1Sq.release(); i2Sq.release(); i1I2.release()
        mu1.release(); mu2.release(); mu1Sq.release(); mu2Sq.release(); mu1Mu2.release()
        sigma1Sq.release(); sigma2Sq.release(); sigma12.release()
        t1.release(); t2.release(); t3.release(); t4.release()
        num.release(); den.release(); ssimMap.release()

        val channels = mat1.channels()
        var ssimSum = mSsim.`val`[0]
        if (channels > 1) ssimSum += mSsim.`val`[1]
        if (channels > 2) ssimSum += mSsim.`val`[2]
        return ssimSum / channels.toDouble()
    }

    /**
     * Calculates PSNR between two Bitmaps.
     */
    fun calculatePsnr(bitmap1: Bitmap, bitmap2: Bitmap): Double {
        val mat1 = Mat()
        val mat2 = Mat()
        Utils.bitmapToMat(bitmap1, mat1)
        Utils.bitmapToMat(bitmap2, mat2)
        val psnr = calculatePsnr(mat1, mat2)
        mat1.release()
        mat2.release()
        return psnr
    }

    /**
     * Calculates SSIM between two Bitmaps.
     */
    fun calculateSsim(bitmap1: Bitmap, bitmap2: Bitmap): Double {
        val mat1 = Mat()
        val mat2 = Mat()
        Utils.bitmapToMat(bitmap1, mat1)
        Utils.bitmapToMat(bitmap2, mat2)
        val ssim = calculateSsim(mat1, mat2)
        mat1.release()
        mat2.release()
        return ssim
    }

    /**
     * Calculates both PSNR and SSIM for a pair of Bitmaps.
     */
    fun calculateMetrics(bitmap1: Bitmap, bitmap2: Bitmap): MetricsResult {
        val mat1 = Mat()
        val mat2 = Mat()
        Utils.bitmapToMat(bitmap1, mat1)
        Utils.bitmapToMat(bitmap2, mat2)
        val psnr = calculatePsnr(mat1, mat2)
        val ssim = calculateSsim(mat1, mat2)
        mat1.release()
        mat2.release()
        return MetricsResult(psnrDb = psnr, ssim = ssim)
    }
}
