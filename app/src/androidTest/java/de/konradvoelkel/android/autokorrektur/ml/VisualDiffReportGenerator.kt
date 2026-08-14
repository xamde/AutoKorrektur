package de.konradvoelkel.android.autokorrektur.ml

import android.graphics.Bitmap
import android.graphics.Color
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream

/**
 * Generates visual diff heatmaps and side-by-side verification composites for ML test reporting.
 *
 * Color encoding for error maps:
 *  - 🟩 Green (0, 220, 0): True Positive (Correctly segmented car bodywork)
 *  - 🟥 Red (230, 30, 30): False Positive (Over-masking non-car background)
 *  - 🟦 Blue (30, 100, 240): False Negative (Missed car bodywork)
 */
object VisualDiffReportGenerator {

    /**
     * Generates a 3-color error visualization overlay blended with the original photo.
     */
    fun generateErrorHeatmap(originalMat: Mat, predMaskMat: Mat, gtMaskMat: Mat): Mat {
        val width = originalMat.cols()
        val height = originalMat.rows()

        val rgbOrig = Mat()
        if (originalMat.channels() == 4) {
            Imgproc.cvtColor(originalMat, rgbOrig, Imgproc.COLOR_RGBA2RGB)
        } else {
            originalMat.copyTo(rgbOrig)
        }

        val overlayMat = rgbOrig.clone()

        val predBytes = ByteArray(width * height)
        val gtBytes = ByteArray(width * height)
        val overlayBytes = ByteArray(width * height * 3)

        val pred8u = Mat()
        val gt8u = Mat()
        predMaskMat.convertTo(pred8u, CvType.CV_8UC1)
        gtMaskMat.convertTo(gt8u, CvType.CV_8UC1)

        pred8u.get(0, 0, predBytes)
        gt8u.get(0, 0, gtBytes)
        rgbOrig.get(0, 0, overlayBytes)

        for (i in 0 until width * height) {
            // Pipeline pred mask: 0 is car, 255 is background
            val isPred = (predBytes[i].toInt() and 0xFF) < 128
            // GT mask in triples: 255 is car, 0 is background (or vice versa if passed converted)
            val isGt = (gtBytes[i].toInt() and 0xFF) > 128

            val idx = i * 3
            if (isPred && isGt) {
                // True Positive -> Green
                overlayBytes[idx] = 0.toByte()
                overlayBytes[idx + 1] = 220.toByte()
                overlayBytes[idx + 2] = 0.toByte()
            } else if (isPred && !isGt) {
                // False Positive -> Red (Overmasking)
                overlayBytes[idx] = 230.toByte()
                overlayBytes[idx + 1] = 30.toByte()
                overlayBytes[idx + 2] = 30.toByte()
            } else if (!isPred && isGt) {
                // False Negative -> Blue (Missed)
                overlayBytes[idx] = 30.toByte()
                overlayBytes[idx + 1] = 100.toByte()
                overlayBytes[idx + 2] = 240.toByte()
            }
        }

        val coloredMat = Mat(height, width, CvType.CV_8UC3)
        coloredMat.put(0, 0, overlayBytes)

        val blendedMat = Mat()
        Core.addWeighted(rgbOrig, 0.45, coloredMat, 0.55, 0.0, blendedMat)

        rgbOrig.release()
        overlayMat.release()
        pred8u.release()
        gt8u.release()
        coloredMat.release()

        return blendedMat
    }

    /**
     * Saves a Mat as a PNG image file in the specified destination.
     */
    fun saveMatToPng(mat: Mat, destinationFile: File) {
        val bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bmp)
        FileOutputStream(destinationFile).use { out ->
            bmp.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        bmp.recycle()
    }
}
