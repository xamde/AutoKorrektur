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
     * Converts an OpenCV Mat (1, 3, or 4 channels) into a displayable ARGB_8888 Bitmap safely.
     */
    fun createDisplayBitmap(mat: Mat): Bitmap {
        if (mat.empty() || mat.cols() <= 0 || mat.rows() <= 0) {
            return createBitmap(1, 1, Bitmap.Config.ARGB_8888)
        }
        val rgbaMat = Mat()
        try {
            when (mat.channels()) {
                1 -> Imgproc.cvtColor(mat, rgbaMat, Imgproc.COLOR_GRAY2RGBA)
                3 -> Imgproc.cvtColor(mat, rgbaMat, Imgproc.COLOR_RGB2RGBA)
                4 -> mat.copyTo(rgbaMat)
                else -> {
                    val channels = mutableListOf<Mat>()
                    org.opencv.core.Core.split(mat, channels)
                    if (channels.size >= 3) {
                        val rgb3 = listOf(channels[0], channels[1], channels[2])
                        val tempRgb = Mat()
                        org.opencv.core.Core.merge(rgb3, tempRgb)
                        Imgproc.cvtColor(tempRgb, rgbaMat, Imgproc.COLOR_RGB2RGBA)
                        tempRgb.release()
                    } else if (channels.isNotEmpty()) {
                        Imgproc.cvtColor(channels[0], rgbaMat, Imgproc.COLOR_GRAY2RGBA)
                    }
                    channels.forEach { it.release() }
                }
            }
        } catch (e: Exception) {
            AppLogger.warn("createDisplayBitmap fallback: ${e.message}")
            try {
                if (rgbaMat.empty()) {
                    Imgproc.cvtColor(mat, rgbaMat, Imgproc.COLOR_BGR2RGBA)
                }
            } catch (_: Exception) {}
        }

        val outMat = if (!rgbaMat.empty() && rgbaMat.cols() > 0 && rgbaMat.rows() > 0) rgbaMat else mat
        val bitmap = createBitmap(outMat.cols(), outMat.rows(), Bitmap.Config.ARGB_8888)
        try {
            Utils.matToBitmap(outMat, bitmap)
        } catch (e: Exception) {
            AppLogger.error("createDisplayBitmap Utils.matToBitmap failed (outMat type=${outMat.type()}, channels=${outMat.channels()})", e)
        } finally {
            if (outMat === rgbaMat) {
                rgbaMat.release()
            }
        }
        return bitmap
    }
}
