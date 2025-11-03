package de.konradvoelkel.android.autokorrektur.shared

import android.content.Context
import android.os.Build
import android.util.Log
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * Shared OpenCV helpers for instrumented tests only.
 * Keep this Android-only (placed under androidTest) to avoid leaking into JVM tests.
 */
object OpenCvTestUtils {
    private const val TAG = "OpenCvTestUtils"

    /** Save an RGB (8u3) Mat as PNG, converting to BGR for correct color in PNG writer. */
    fun saveDebugRgbMatAsPngBgr(matRgb: Mat, file: File) {
        val bgr = Mat()
        Imgproc.cvtColor(matRgb, bgr, Imgproc.COLOR_RGB2BGR)
        Imgcodecs.imwrite(file.absolutePath, bgr)
        bgr.release()
    }

    /** Convert a Mat loaded as BGR (e.g., via Imgcodecs) to RGB. */
    fun matLoadedFromFileBgrToRgb(matBgr: Mat): Mat {
        val rgb = Mat()
        Imgproc.cvtColor(matBgr, rgb, Imgproc.COLOR_BGR2RGB)
        return rgb
    }

    /**
     * Mean absolute difference per channel in white areas of a grayscale mask.
     * Preconditions:
     * - maskGray: CV_8UC1
     * - aRgb8u3, bRgb8u3: CV_8UC3
     * - All sizes match
     */
    fun meanAbsDiffOnMaskRgb8u3(maskGray: Mat, aRgb8u3: Mat, bRgb8u3: Mat): Double {
        require(maskGray.type() == CvType.CV_8UC1) { "maskGray must be CV_8UC1" }
        require(aRgb8u3.type() == CvType.CV_8UC3) { "aRgb8u3 must be CV_8UC3" }
        require(bRgb8u3.type() == CvType.CV_8UC3) { "bRgb8u3 must be CV_8UC3" }
        require(aRgb8u3.rows() == bRgb8u3.rows() && aRgb8u3.cols() == bRgb8u3.cols()) { "Image sizes must match" }
        require(maskGray.rows() == aRgb8u3.rows() && maskGray.cols() == aRgb8u3.cols()) { "Mask size must match image size" }

        val aData = ByteArray(aRgb8u3.rows() * aRgb8u3.cols() * aRgb8u3.channels())
        val bData = ByteArray(bRgb8u3.rows() * bRgb8u3.cols() * bRgb8u3.channels())
        val mData = ByteArray(maskGray.rows() * maskGray.cols())
        aRgb8u3.get(0, 0, aData)
        bRgb8u3.get(0, 0, bData)
        maskGray.get(0, 0, mData)

        var sumAbs: Long = 0
        var count: Long = 0
        val ch = 3
        for (i in mData.indices) {
            val mv = mData[i].toInt() and 0xFF
            if (mv >= 245) { // consider white-ish mask
                val base = i * ch
                val d0 = kotlin.math.abs((aData[base].toInt() and 0xFF) - (bData[base].toInt() and 0xFF))
                val d1 = kotlin.math.abs((aData[base + 1].toInt() and 0xFF) - (bData[base + 1].toInt() and 0xFF))
                val d2 = kotlin.math.abs((aData[base + 2].toInt() and 0xFF) - (bData[base + 2].toInt() and 0xFF))
                sumAbs += (d0 + d1 + d2)
                count += 3
            }
        }
        check(count > 0) { "Reference mask must contain white pixels" }
        return sumAbs.toDouble() / count.toDouble()
    }

    /** Exact byte-by-byte equality for Mats of same size/type. */
    fun matsAreExactlyEqual(a: Mat, b: Mat): Boolean {
        if (a.rows() != b.rows() || a.cols() != b.cols() || a.type() != b.type()) return false
        val total = a.rows() * a.cols() * a.channels()
        val ad = ByteArray(total)
        val bd = ByteArray(total)
        a.get(0, 0, ad)
        b.get(0, 0, bd)
        for (i in 0 until total) {
            if (ad[i] != bd[i]) return false
        }
        return true
    }

    /**
     * Decide whether we should write debug artifacts (PNGs etc.).
     * Default is false. We enable only if:
     *  - App is debuggable AND
     *  - An instrumentation argument or system property AUTOKORREKTUR_WRITE_DEBUG == "1".
     */
    fun shouldWriteDebugArtifacts(context: Context): Boolean {
        val debuggable = (context.applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0
        if (!debuggable) return false
        val fromArg = try {
            // Available in instrumented runs; best-effort read
            val bundle = androidx.test.platform.app.InstrumentationRegistry.getArguments()
            bundle.getString("AUTOKORREKTUR_WRITE_DEBUG")
        } catch (_: Throwable) { null }
        val sys = try { System.getProperty("AUTOKORREKTUR_WRITE_DEBUG") } catch (_: Throwable) { null }
        val env = try { System.getenv("AUTOKORREKTUR_WRITE_DEBUG") } catch (_: Throwable) { null }
        val flag = (fromArg ?: sys ?: env)
        val enabled = flag == "1"
        if (enabled) Log.d(TAG, "Debug artifact writing enabled via AUTOKORREKTUR_WRITE_DEBUG=1 (SDK=${Build.VERSION.SDK_INT})")
        return enabled
    }
}