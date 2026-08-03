package de.konradvoelkel.android.autokorrektur.ml.api

import android.content.Context
import de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloEngine
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine
import de.konradvoelkel.android.autokorrektur.ml.mask.YoloMaskAssembler
import de.konradvoelkel.android.autokorrektur.ml.model.Detection
import de.konradvoelkel.android.autokorrektur.ml.model.YoloResult
import de.konradvoelkel.android.autokorrektur.ml.post.YoloPostprocessor
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
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

    override suspend fun initialize(modelName: String, useFP16: Boolean, config: YoloConfig) {
        mutex.withLock {
            currentConfig = config
            engine.initialize(modelName, useFP16)
        }
        AppLogger.debug("YoloServiceImpl initialized with ${modelName}, fp16=$useFP16")
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

                // 5) Final cropping and resizing
                val resultMat = postProcessResultMask(
                    overlay,
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
        return if (transformedMat.type() != CvType.CV_8UC3) {
            val tmp = Mat().also { matsToRelease.add(it) }
            val scale = if (transformedMat.type() == CvType.CV_32FC3) 255.0 else 1.0
            transformedMat.convertTo(tmp, CvType.CV_8UC3, scale)
            tmp
        } else {
            transformedMat
        }
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
        return overlay
    }

    private fun postProcessResultMask(
        overlay: Mat,
        shapes: de.konradvoelkel.android.autokorrektur.ml.model.Shapes,
        xRatio: Float,
        yRatio: Float,
        originalWidth: Int?,
        originalHeight: Int?
    ): Mat {
        var resultMat = overlay.clone()
        try {
            val contentW =
                kotlin.math.max(1, kotlin.math.min(shapes.inputW, (shapes.inputW / xRatio).toInt()))
            val contentH =
                kotlin.math.max(1, kotlin.math.min(shapes.inputH, (shapes.inputH / yRatio).toInt()))

            if (contentW != shapes.inputW || contentH != shapes.inputH) {
                val cropped = Mat(resultMat, Rect(0, 0, contentW, contentH)).clone()
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
            resultMat.release()
            throw e
        }
    }

    override fun close() {
        engine.close()
    }
}