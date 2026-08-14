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
    // The input mat is CV_32FC1. Its values can be anything (e.g., -5.0 to 10.0).
    // To visualize it, we need to normalize these values to a 0-255 range.
    val normalizedMat = Mat()
    Core.normalize(mat, normalizedMat, 0.0, 255.0, Core.NORM_MINMAX)

    // Now convert the normalized float matrix to an 8-bit unsigned integer matrix.
    val displayMat = Mat()
    normalizedMat.convertTo(displayMat, CvType.CV_8U)

    // Create a Bitmap with the same dimensions as the Mat.
    val bitmap = createBitmap(displayMat.cols(), displayMat.rows())

    // Copy the Mat data to the Bitmap.
    Utils.matToBitmap(displayMat, bitmap)

    // Release intermediate Mats to free up memory
    normalizedMat.release()
    displayMat.release()

    return bitmap
}