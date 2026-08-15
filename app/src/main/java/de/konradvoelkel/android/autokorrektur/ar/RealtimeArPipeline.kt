package de.konradvoelkel.android.autokorrektur.ar

import android.graphics.Bitmap
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.ml.preprocess.DefaultPreprocessor
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
    private val preprocessor = DefaultPreprocessor()

    private var _isInitialized = false
    val isInitialized: Boolean get() = _isInitialized

    private var _isClosed = false
    val isClosed: Boolean get() = _isClosed

    var onFrameRendered: ((Bitmap, Float) -> Unit)? = null

    private var lastFrameTimeNs = 0L
    private var smoothedFps = 30f

    private var reusableOutputBitmap: Bitmap? = null

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

                // 1. Convert RGBA to RGB for neural preprocessor
                val frameRgb = Mat()
                Imgproc.cvtColor(frameCopy, frameRgb, Imgproc.COLOR_RGBA2RGB)

                // 2. Preprocess with DefaultPreprocessor (stride=32, letterbox alignment)
                val prep = preprocessor.prepare(frameRgb, 640, 640)
                frameRgb.release()

                val transformedFloatMat = Mat()
                prep.forEngine.convertTo(transformedFloatMat, CvType.CV_32F, 1.0 / 255.0)
                prep.forEngine.release()
                prep.forBitmap.release()

                // 3. Run YOLO segmentation with exact geometric ratios
                val subtractiveMask = yoloService.infer(
                    transformedMat = transformedFloatMat,
                    xRatio = prep.xRatio,
                    yRatio = prep.yRatio,
                    originalWidth = origW,
                    originalHeight = origH
                )
                transformedFloatMat.release()

                // 4. Invert subtractive mask (0=car, 255=bg) -> (255=car, 0=bg) for accumulator
                val carMaskMat = Mat()
                Core.bitwise_not(subtractiveMask, carMaskMat)
                subtractiveMask.release()

                // 5. Accumulate clean background and blend
                val blendedMat = accumulator.accumulateAndBlend(frameCopy, carMaskMat)
                carMaskMat.release()

                // 6. Convert blended Mat to Bitmap
                var outputBitmap = reusableOutputBitmap
                if (outputBitmap == null || outputBitmap.width != origW || outputBitmap.height != origH) {
                    outputBitmap?.recycle()
                    outputBitmap = Bitmap.createBitmap(origW, origH, Bitmap.Config.ARGB_8888)
                    reusableOutputBitmap = outputBitmap
                }
                Utils.matToBitmap(blendedMat, outputBitmap)
                blendedMat.release()

                // 7. Calculate rolling FPS
                val nowNs = System.nanoTime()
                if (lastFrameTimeNs > 0L) {
                    val deltaSec = (nowNs - lastFrameTimeNs) / 1_000_000_000.0f
                    if (deltaSec > 0f) {
                        val currentFps = 1.0f / deltaSec
                        smoothedFps = 0.85f * smoothedFps + 0.15f * currentFps
                    }
                }
                lastFrameTimeNs = nowNs

                // 8. Dispatch to listener
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
