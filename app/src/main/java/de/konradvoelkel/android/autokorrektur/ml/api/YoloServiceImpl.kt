package de.konradvoelkel.android.autokorrektur.ml.api

import android.content.Context
import de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloEngine
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine
import de.konradvoelkel.android.autokorrektur.ml.mask.GuidedFilter
import de.konradvoelkel.android.autokorrektur.ml.mask.YoloMaskAssembler
import de.konradvoelkel.android.autokorrektur.ml.model.Detection
import de.konradvoelkel.android.autokorrektur.ml.model.YoloResult
import de.konradvoelkel.android.autokorrektur.ml.post.YoloPostprocessor
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Default implementation composing the engine + postprocessor + mask assembler.
 * This mirrors the legacy YoloInferenceTFLite behavior but with separated concerns.
 */
class YoloServiceImpl(
    private val engine: YoloEngine
) : YoloService {

    @Volatile
    private var currentConfig: YoloConfig = YoloConfig()

    private val mutex = Mutex()

    override val isInitialized: Boolean
        get() = engine.isInitialized

    override suspend fun initialize(modelName: String, config: YoloConfig) {
        mutex.withLock {
            currentConfig = config
            engine.initialize(modelName)
        }
        AppLogger.debug("YoloServiceImpl initialized with $modelName")
    }

    override suspend fun infer(
        transformedMat: Mat,
        xRatio: Float,
        yRatio: Float,
        upscaleFactor: Float,
        originalWidth: Int?,
        originalHeight: Int?
    ): Mat {
        return inferDetailed(
            transformedMat = transformedMat,
            xRatio = xRatio,
            yRatio = yRatio,
            upscaleFactor = upscaleFactor,
            originalWidth = originalWidth,
            originalHeight = originalHeight,
            overrideConfig = null
        ).mask
    }

    override suspend fun inferDetailed(
        transformedMat: Mat,
        xRatio: Float,
        yRatio: Float,
        upscaleFactor: Float,
        originalWidth: Int?,
        originalHeight: Int?,
        overrideConfig: YoloConfig?
    ): YoloResult = withContext(Dispatchers.Default) {
        if (!engine.isInitialized) {
            AppLogger.info("YoloServiceImpl inferDetailed called on uninitialized engine; auto-initializing default model...")
            initialize()
        }

        val config = overrideConfig ?: currentConfig
        val matsToRelease = mutableListOf<Mat>()
        val warnings = mutableListOf<String>()

        try {
            // 1) Ensure input format
            val inputMat = ensureCorrectInputFormat(transformedMat, matsToRelease)

            // 2) Run model
            val raw = engine.run(inputMat)
            val shapes = raw.shapes

            // 3) Parse results
            val kept = parseDetectionsAndApplyNMS(raw, config)
            if (kept.isEmpty()) {
                warnings.add("No vehicles detected in the image.")
            }

            // 4) Assemble mask
            val protoShape = shapes.protoShape
            val prototypes = YoloMaskAssembler.extractPrototypeMasks(raw.prototypes, protoShape)
            val deinterleaved = try {
                YoloMaskAssembler.deinterleavePrototypes(prototypes, protoShape)
            } catch (e: Exception) {
                AppLogger.warn("Deinterleave prototypes failed: ${e.message}")
                warnings.add("Prototype extraction degraded: ${e.message}")
                null
            }

            try {
                val overlay = assembleFinalMask(
                    kept,
                    deinterleaved,
                    prototypes,
                    protoShape,
                    shapes,
                    upscaleFactor,
                    warnings,
                    matsToRelease
                )

                // Edge-preserving Guided Filter refinement using input RGB as guidance
                val refinedOverlay = if (kept.isNotEmpty()) {
                    try {
                        val filtered = GuidedFilter.filter(guide = transformedMat, srcMask = overlay)
                        matsToRelease.add(filtered)
                        filtered
                    } catch (e: Exception) {
                        AppLogger.warn("GuidedFilter fallback: ${e.message}")
                        overlay
                    }
                } else {
                    overlay
                }

                // 5) Final cropping and resizing
                val resultMat = postProcessResultMask(
                    refinedOverlay,
                    shapes,
                    xRatio,
                    yRatio,
                    originalWidth,
                    originalHeight
                )

                YoloResult(mask = resultMat, detections = kept, warnings = warnings)
            } finally {
                deinterleaved?.forEach { it.release() }
            }
        } finally {
            matsToRelease.forEach { it.release() }
        }
    }

    private fun ensureCorrectInputFormat(
        transformedMat: Mat,
        matsToRelease: MutableList<Mat>
    ): Mat {
        var current = transformedMat
        if (current.depth() != CvType.CV_8U) {
            val tmp = Mat().also { matsToRelease.add(it) }
            val scale = if (current.depth() == CvType.CV_32F) 255.0 else 1.0
            current.convertTo(tmp, CvType.CV_8U, scale)
            current = tmp
        }
        if (current.channels() != 3) {
            val tmp = Mat().also { matsToRelease.add(it) }
            when (current.channels()) {
                4 -> Imgproc.cvtColor(current, tmp, Imgproc.COLOR_RGBA2RGB)
                1 -> Imgproc.cvtColor(current, tmp, Imgproc.COLOR_GRAY2RGB)
                else -> {
                    val channels = mutableListOf<Mat>()
                    Core.split(current, channels)
                    if (channels.size >= 3) {
                        val rgb3 = listOf(channels[0], channels[1], channels[2])
                        Core.merge(rgb3, tmp)
                    } else if (channels.isNotEmpty()) {
                        Imgproc.cvtColor(channels[0], tmp, Imgproc.COLOR_GRAY2RGB)
                    }
                    channels.forEach { it.release() }
                }
            }
            current = tmp
        }
        return current
    }

    private fun parseDetectionsAndApplyNMS(
        raw: de.konradvoelkel.android.autokorrektur.ml.model.RawOutputs,
        config: YoloConfig
    ): List<Detection> {
        val shapes = raw.shapes
        val proposals = shapes.detShape.getOrNull(2) ?: 0
        val features = shapes.detShape.getOrNull(1) ?: 0
        val numClasses = config.labels.size

        val parsed = YoloPostprocessor.parseDetections(
            buffer = raw.detections,
            numProposals = proposals,
            featuresPerProposal = features,
            numClasses = numClasses,
            scoreThreshold = config.scoreThreshold,
            allowedClassIndices = config.vehicleClassIndices
        )
        return YoloPostprocessor.applyNMS(
            detections = parsed,
            iouThreshold = config.iouThreshold,
            topAmountPerClass = config.topAmountPerClass,
            numClasses = numClasses
        )
    }

    private fun assembleFinalMask(
        kept: List<Detection>,
        deinterleaved: List<Mat>?,
        prototypes: FloatArray,
        protoShape: IntArray,
        shapes: de.konradvoelkel.android.autokorrektur.ml.model.Shapes,
        upscaleFactor: Float,
        warnings: MutableList<String>,
        matsToRelease: MutableList<Mat>
    ): Mat {
        val tightUpscaleFactor = upscaleFactor.coerceIn(1.0f, 1.05f)
        val overlay =
            Mat(shapes.inputH, shapes.inputW, CvType.CV_8UC1).also { matsToRelease.add(it) }
        overlay.setTo(Scalar(255.0))

        for (det in kept) {
            try {
                if (deinterleaved != null) {
                    YoloMaskAssembler.createDetectionMask(
                        det,
                        overlay,
                        tightUpscaleFactor,
                        deinterleaved,
                        shapes.inputW,
                        shapes.inputH
                    )
                } else {
                    YoloMaskAssembler.createDetectionMask(
                        det,
                        overlay,
                        tightUpscaleFactor,
                        prototypes,
                        shapes.inputW,
                        shapes.inputH,
                        protoShape
                    )
                }
            } catch (e: Exception) {
                val msg = "Mask assembly failed for detection at [${det.x}, ${det.y}]: ${e.message}"
                AppLogger.warn(msg)
                warnings.add(msg)
            }
        }

        // Automatically expand vehicle holes downwards into contact shadows and ground reflections
        if (kept.isNotEmpty()) {
            expandShadowsAndGroundReflections(overlay)
        }

        return overlay
    }

    /**
     * Intelligently expands vehicle holes downwards onto the ground plane / asphalt
     * to eliminate contact shadows and puddle reflections automatically.
     */
    private fun expandShadowsAndGroundReflections(subtractiveOverlay: Mat) {
        if (subtractiveOverlay.empty()) return

        // Use a purely vertical kernel to expand holes downward without lateral spread.
        // Eroding the subtractive mask (where car=0, background=255) directly expands
        // the hole region downward, eliminating the need for two bitwise_not operations.
        val kHeight = (subtractiveOverlay.rows() * 0.025).toInt().coerceIn(7, 25)
        val kernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT,
            Size(1.0, kHeight.toDouble()),
            org.opencv.core.Point(0.0, 1.0) // anchor near top → expansion projects downward
        )

        Imgproc.erode(subtractiveOverlay, subtractiveOverlay, kernel)
        kernel.release()
    }

    private fun postProcessResultMask(
        overlay: Mat,
        shapes: de.konradvoelkel.android.autokorrektur.ml.model.Shapes,
        xRatio: Float,
        yRatio: Float,
        originalWidth: Int?,
        originalHeight: Int?
    ): Mat {
        if (overlay.empty() || overlay.cols() <= 0 || overlay.rows() <= 0) {
            val targetW = originalWidth ?: shapes.inputW
            val targetH = originalHeight ?: shapes.inputH
            val emptyMask = Mat(targetH, targetW, CvType.CV_8UC1)
            emptyMask.setTo(Scalar(255.0))
            return emptyMask
        }
        var resultMat = overlay.clone()
        try {
            val contentW =
                kotlin.math.max(1, kotlin.math.min(resultMat.cols(), (shapes.inputW / xRatio).toInt()))
            val contentH =
                kotlin.math.max(1, kotlin.math.min(resultMat.rows(), (shapes.inputH / yRatio).toInt()))

            if (contentW < resultMat.cols() || contentH < resultMat.rows()) {
                val safeRect = Rect(0, 0, contentW.coerceAtMost(resultMat.cols()), contentH.coerceAtMost(resultMat.rows()))
                val sub = resultMat.submat(safeRect)
                val cropped = sub.clone()
                sub.release()
                resultMat.release()
                resultMat = cropped
            }

            if (originalWidth != null && originalHeight != null && originalWidth > 0 && originalHeight > 0) {
                val resized = Mat()
                Imgproc.resize(
                    resultMat,
                    resized,
                    Size(originalWidth.toDouble(), originalHeight.toDouble()),
                    0.0,
                    0.0,
                    Imgproc.INTER_NEAREST
                )
                resultMat.release()
                resultMat = resized
            }
            return resultMat
        } catch (e: Exception) {
            AppLogger.warn("postProcessResultMask error, returning fallback: ${e.message}")
            val targetW = originalWidth ?: shapes.inputW
            val targetH = originalHeight ?: shapes.inputH
            val fallback = Mat(targetH, targetW, CvType.CV_8UC1)
            fallback.setTo(Scalar(255.0))
            resultMat.release()
            return fallback
        }
    }

    override fun close() {
        engine.close()
    }
}