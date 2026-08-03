package de.konradvoelkel.android.autokorrektur.ml

import android.graphics.Bitmap
import androidx.core.graphics.createBitmap
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Handles megapixel-based scaling of OpenCV Mats and conversion to display Bitmaps.
 */
object MatScaler {
    private const val MEGAPIXEL = 1_000_000f

    /**
     * Downscales a Mat if its megapixel count exceeds [maxMegapixels].
     * Returns a new Mat if scaling occurred, or the original Mat if no scaling was needed.
     * Note: If a new Mat is returned, the caller is responsible for releasing it.
     */
    fun downscaleIfLarge(mat: Mat, maxMegapixels: Float?): Mat {
        if (maxMegapixels == null) return mat

        val currentMegapixels = (mat.rows() * mat.cols()) / MEGAPIXEL
        if (currentMegapixels <= maxMegapixels) return mat

        val scale = sqrt(maxMegapixels.toDouble() / currentMegapixels)
        val newSize = Size(
            (mat.cols() * scale).roundToInt().toDouble(),
            (mat.rows() * scale).roundToInt().toDouble()
        )

        val downscaledMat = Mat()
        Imgproc.resize(mat, downscaledMat, newSize, 0.0, 0.0, Imgproc.INTER_AREA)
        AppLogger.debug("MatScaler: Downscaled Mat to ${downscaledMat.width()}x${downscaledMat.height()}")
        return downscaledMat
    }

    /**
     * Converts a 3-channel RGB Mat into a displayable ARGB_8888 Bitmap.
     */
    fun createDisplayBitmap(rgbMat: Mat): Bitmap {
        val bgraMat = Mat()
        Imgproc.cvtColor(rgbMat, bgraMat, Imgproc.COLOR_RGB2RGBA)

        val bitmap = createBitmap(bgraMat.cols(), bgraMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(bgraMat, bitmap)

        bgraMat.release()
        return bitmap
    }
}
