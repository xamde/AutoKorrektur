package de.konradvoelkel.android.autokorrektur.ar

import android.graphics.Bitmap
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.util.concurrent.atomic.AtomicBoolean

/**
 * High-performance real-time coordination engine managing YOLO inference,
 * temporal background accumulation, frame-skipping, and viewfinder texture rendering.
 */
class RealtimeArPipeline(
    val yoloService: YoloService,
    val accumulator: TemporalBackgroundAccumulator = TemporalBackgroundAccumulator()
) : AutoCloseable {

    private val pipelineScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val isProcessingFrame = AtomicBoolean(false)

    private var _isInitialized = false
    val isInitialized: Boolean get() = _isInitialized

    private var _isClosed = false
    val isClosed: Boolean get() = _isClosed

    var onFrameRendered: ((Bitmap, Float) -> Unit)? = null

    private var lastFrameTimeNs = 0L
    private var smoothedFps = 30f

    /**
     * Initializes the underlying YOLO engine.
     */
    suspend fun initialize(modelName: String = "yolo11s") {
        if (_isInitialized) return
        yoloService.initialize(modelName = modelName, useFP16 = false)
        _isInitialized = true
        AppLogger.info("RealtimeArPipeline: YOLO initialized with model $modelName")
    }

    /**
     * Ingests a camera frame RGBA matrix, performs non-blocking vehicle detection,
     * blends accumulated background textures, and outputs the rendered frame.
     *
     * Drops frames gracefully if previous inference is still executing.
     */
    fun processFrame(frameRgbaMat: Mat) {
        if (!_isInitialized || _isClosed || frameRgbaMat.empty()) return

        if (!isProcessingFrame.compareAndSet(false, true)) {
            // Frame dropped cleanly to preserve camera preview framerate
            return
        }

        val frameCopy = frameRgbaMat.clone()

        pipelineScope.launch {
            try {
                val origW = frameCopy.cols()
                val origH = frameCopy.rows()

                // 1. Scale and pad to 640x640 for YOLO
                val yoloRgba = ArFrameConverter.scaleAndPadForYolo(frameCopy, 640)
                val yoloRgb = Mat()
                Imgproc.cvtColor(yoloRgba, yoloRgb, Imgproc.COLOR_RGBA2RGB)
                yoloRgba.release()

                // 2. Run YOLO segmentation
                val maxDim = kotlin.math.max(origW, origH).toFloat()
                val xRatio = maxDim / origW.toFloat()
                val yRatio = maxDim / origH.toFloat()
                val subtractiveMask = yoloService.infer(
                    transformedMat = yoloRgb,
                    xRatio = xRatio,
                    yRatio = yRatio,
                    originalWidth = origW,
                    originalHeight = origH
                )
                yoloRgb.release()

                // 3. Invert subtractive mask (0=car, 255=bg) -> (255=car, 0=bg) for accumulator
                val carMaskMat = Mat()
                Core.bitwise_not(subtractiveMask, carMaskMat)
                subtractiveMask.release()

                // 4. Accumulate clean background and blend
                val blendedMat = accumulator.accumulateAndBlend(frameCopy, carMaskMat)
                carMaskMat.release()

                // 5. Convert blended Mat to Bitmap
                val outputBitmap = Bitmap.createBitmap(origW, origH, Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(blendedMat, outputBitmap)
                blendedMat.release()

                // 6. Calculate rolling FPS
                val nowNs = System.nanoTime()
                if (lastFrameTimeNs > 0L) {
                    val deltaSec = (nowNs - lastFrameTimeNs) / 1_000_000_000.0f
                    if (deltaSec > 0f) {
                        val currentFps = 1.0f / deltaSec
                        smoothedFps = 0.85f * smoothedFps + 0.15f * currentFps
                    }
                }
                lastFrameTimeNs = nowNs

                // 7. Dispatch to listener
                onFrameRendered?.invoke(outputBitmap, smoothedFps)
            } catch (e: Exception) {
                AppLogger.error("RealtimeArPipeline: Frame processing failed", e)
            } finally {
                frameCopy.release()
                isProcessingFrame.set(false)
            }
        }
    }

    /**
     * Clears the accumulated background buffer.
     */
    fun reset() {
        accumulator.reset()
        AppLogger.info("RealtimeArPipeline: Reset accumulated background buffer")
    }

    override fun close() {
        if (_isClosed) return
        _isClosed = true
        pipelineScope.cancel()
        accumulator.close()
        yoloService.close()
        AppLogger.info("RealtimeArPipeline: Released all resources")
    }
}
