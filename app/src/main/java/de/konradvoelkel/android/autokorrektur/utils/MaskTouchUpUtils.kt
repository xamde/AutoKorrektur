package de.konradvoelkel.android.autokorrektur.utils

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Utility functions for manual mask touch-up, dilation, and boundary refinement.
 */
object MaskTouchUpUtils {

    /**
     * Dilates a binary segmentation mask by [radiusPx] pixels to expand vehicle boundary coverage.
     */
    fun createDilatedMask(maskBitmap: Bitmap, radiusPx: Int): Bitmap {
        if (radiusPx <= 0) return maskBitmap

        val matsToRelease = mutableListOf<Mat>()
        try {
            val mat = Mat().also { matsToRelease.add(it) }
            Utils.bitmapToMat(maskBitmap, mat)

            val kernelSize = (radiusPx * 2 + 1).toDouble()
            val kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE,
                Size(kernelSize, kernelSize)
            ).also { matsToRelease.add(it) }

            val dilatedMat = Mat().also { matsToRelease.add(it) }
            Imgproc.dilate(mat, dilatedMat, kernel)

            val resultBitmap = Bitmap.createBitmap(
                maskBitmap.width,
                maskBitmap.height,
                Bitmap.Config.ARGB_8888
            )
            Utils.matToBitmap(dilatedMat, resultBitmap)
            return resultBitmap
        } finally {
            matsToRelease.forEach { it.release() }
        }
    }

    /**
     * Combines an original detection mask with user brush strokes.
     */
    fun mergeMaskWithStrokes(baseMask: Bitmap, brushStrokeBitmap: Bitmap): Bitmap {
        val merged = Bitmap.createBitmap(baseMask.width, baseMask.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(merged)
        canvas.drawBitmap(baseMask, 0f, 0f, null)
        canvas.drawBitmap(brushStrokeBitmap, 0f, 0f, null)
        return merged
    }
}
