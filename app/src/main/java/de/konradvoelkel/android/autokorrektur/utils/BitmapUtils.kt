package de.konradvoelkel.android.autokorrektur.utils

import android.graphics.Bitmap
import org.opencv.core.Mat
import org.opencv.core.CvType
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer

object BitmapUtils {

    fun matToBitmap(mat: Mat): Bitmap? {
        if (mat.empty()) {
            return null
        }

        val bitmap: Bitmap
        val matToProcess: Mat

        when (mat.type()) {
            CvType.CV_8UC1 -> {
                bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ALPHA_8)
                matToProcess = mat
            }
            CvType.CV_8UC3 -> {
                bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
                matToProcess = Mat()
                Imgproc.cvtColor(mat, matToProcess, Imgproc.COLOR_RGB2RGBA)
            }
            CvType.CV_8UC4 -> {
                 bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
                 matToProcess = mat
            }
            else -> {
                // Fallback for other types
                matToProcess = Mat()
                mat.convertTo(matToProcess, CvType.CV_8UC4)
                bitmap = Bitmap.createBitmap(matToProcess.cols(), matToProcess.rows(), Bitmap.Config.ARGB_8888)
            }
        }

        val data = ByteArray(matToProcess.total().toInt() * matToProcess.channels())
        matToProcess.get(0, 0, data)
        bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(data))

        // Release temporary Mat if it was created
        if (matToProcess !== mat) {
            matToProcess.release()
        }

        return bitmap
    }
}
