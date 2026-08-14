package de.konradvoelkel.android.autokorrektur.ar

import androidx.camera.core.ImageProxy
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * High-performance native image converter for real-time AR Camera streams.
 * Converts CameraX [ImageProxy] buffers to OpenCV matrices, handles sensor rotation,
 * and normalizes dimensions for neural model inference.
 */
object ArFrameConverter {

    private val threadLocalNv21Buffer = ThreadLocal<ByteArray>()

    /**
     * Converts a CameraX YUV_420_888 [ImageProxy] into an OpenCV RGBA [Mat] (CV_8UC4).
     *
     * @param image CameraX ImageProxy frame.
     * @return Newly allocated RGBA Mat (caller is responsible for releasing via [Mat.release]).
     */
    fun yuvImageProxyToRgbaMat(image: ImageProxy): Mat {
        val width = image.width
        val height = image.height

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val nv21Size = width * height * 3 / 2
        var nv21 = threadLocalNv21Buffer.get()
        if (nv21 == null || nv21.size < nv21Size) {
            nv21 = ByteArray(nv21Size)
            threadLocalNv21Buffer.set(nv21)
        }

        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride

        var pos = 0
        if (yPixelStride == 1 && yRowStride == width) {
            yBuffer.rewind()
            yBuffer.get(nv21, 0, width * height)
            pos = width * height
        } else {
            val yBytes = ByteArray(yRowStride)
            for (row in 0 until height) {
                yBuffer.position(row * yRowStride)
                yBuffer.get(yBytes, 0, width)
                System.arraycopy(yBytes, 0, nv21, pos, width)
                pos += width
            }
        }

        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride
        val uvWidth = width / 2
        val uvHeight = height / 2

        val uBytes = ByteArray(uvRowStride)
        val vBytes = ByteArray(vPlane.rowStride)

        for (row in 0 until uvHeight) {
            uBuffer.position(row * uvRowStride)
            uBuffer.get(uBytes, 0, max(0, minOf(uBuffer.remaining(), uvRowStride)))
            vBuffer.position(row * vPlane.rowStride)
            vBuffer.get(vBytes, 0, max(0, minOf(vBuffer.remaining(), vPlane.rowStride)))

            for (col in 0 until uvWidth) {
                val vVal = vBytes[col * uvPixelStride]
                val uVal = uBytes[col * uvPixelStride]
                nv21[pos++] = vVal
                nv21[pos++] = uVal
            }
        }

        val yuvMat = Mat(height + height / 2, width, CvType.CV_8UC1)
        yuvMat.put(0, 0, nv21)

        val rgbaMat = Mat()
        Imgproc.cvtColor(yuvMat, rgbaMat, Imgproc.COLOR_YUV2RGBA_NV21)
        yuvMat.release()

        return rgbaMat
    }

    /**
     * Rotates an OpenCV [Mat] according to sensor [rotationDegrees] (0, 90, 180, 270).
     *
     * @param src Input matrix.
     * @param rotationDegrees Sensor orientation in degrees.
     * @return Rotated matrix (caller must release).
     */
    fun rotateMat(src: Mat, rotationDegrees: Int): Mat {
        val normalizedDegrees = ((rotationDegrees % 360) + 360) % 360
        return when (normalizedDegrees) {
            90 -> {
                val dst = Mat()
                Core.rotate(src, dst, Core.ROTATE_90_CLOCKWISE)
                dst
            }
            180 -> {
                val dst = Mat()
                Core.rotate(src, dst, Core.ROTATE_180)
                dst
            }
            270 -> {
                val dst = Mat()
                Core.rotate(src, dst, Core.ROTATE_90_COUNTERCLOCKWISE)
                dst
            }
            else -> src.clone()
        }
    }

    /**
     * Resizes and symmetrically pads an OpenCV RGB/RGBA matrix to target square dimensions (640x640)
     * for YOLO inference while preserving aspect ratio.
     *
     * @param src Input camera matrix.
     * @param targetSize Target square dimension (e.g. 640).
     * @return 640x640 matrix matching [src.type()] (caller must release).
     */
    fun scaleAndPadForYolo(src: Mat, targetSize: Int = 640): Mat {
        val w = src.cols()
        val h = src.rows()
        val maxDim = max(w, h)
        val scale = targetSize.toFloat() / maxDim.toFloat()
        val newW = max(1, (w * scale).roundToInt())
        val newH = max(1, (h * scale).roundToInt())

        val resized = Mat()
        Imgproc.resize(src, resized, Size(newW.toDouble(), newH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)

        val targetMat = Mat.zeros(targetSize, targetSize, src.type())
        val xPad = (targetSize - newW) / 2
        val yPad = (targetSize - newH) / 2

        val roi = Rect(xPad, yPad, newW, newH)
        val targetRoi = targetMat.submat(roi)
        resized.copyTo(targetRoi)
        targetRoi.release()
        resized.release()

        return targetMat
    }
}
