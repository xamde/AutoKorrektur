package de.konradvoelkel.android.autokorrektur.utils

import android.graphics.Bitmap
import androidx.core.graphics.createBitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat

/**
 * Converts a floating-point OpenCV Mat (CV_32FC1) into a Bitmap for debugging.
 * It normalizes the Mat's values to the 0-255 range to make them visible.
 */
fun matToBitmapForDebug(mat: Mat): Bitmap {
    val matsToRelease = mutableListOf<Mat>()
    try {
        val normalizedMat = Mat().also { matsToRelease.add(it) }
        Core.normalize(mat, normalizedMat, 0.0, 255.0, Core.NORM_MINMAX)

        val displayMat = Mat().also { matsToRelease.add(it) }
        normalizedMat.convertTo(displayMat, CvType.CV_8U)

        val bitmap = createBitmap(displayMat.cols(), displayMat.rows())
        Utils.matToBitmap(displayMat, bitmap)
        return bitmap
    } finally {
        matsToRelease.forEach { it.release() }
    }
}