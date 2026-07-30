package de.konradvoelkel.android.autokorrektur.ml.api

import android.content.Context
import de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine
import de.konradvoelkel.android.autokorrektur.ml.mask.YoloMaskAssembler
import de.konradvoelkel.android.autokorrektur.ml.model.Detection
import de.konradvoelkel.android.autokorrektur.ml.model.YoloResult
import de.konradvoelkel.android.autokorrektur.ml.post.YoloPostprocessor
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
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
    private val context: Context
) : YoloService {

    private val engine by lazy { YoloTFLiteEngine(context) }
    private var currentConfig: YoloConfig = YoloConfig()

    override fun initialize(modelName: String, useFP16: Boolean, config: YoloConfig) {
        currentConfig = config
        engine.initialize(modelName, useFP16)
        AppLogger.debug("YoloServiceImpl initialized with ${modelName}, fp16=$useFP16")
    }

    override fun infer(
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

    override fun inferDetailed(
        transformedMat: Mat,
        xRatio: Float,
        yRatio: Float,
        upscaleFactor: Float,
        originalWidth: Int?,
        originalHeight: Int?,
        overrideConfig: YoloConfig?
    ): YoloResult {
        require(engine.isInitialized) { "YoloService used before initialize()" }

        val effectiveConfig = overrideConfig ?: currentConfig

        // 1) Ensure input type is CV_8UC3 RGB as expected by engine
        var inputMat = transformedMat
        try {
            if (transformedMat.type() != CvType.CV_8UC3) {
                val tmp = Mat()
                // If this is a normalized float Mat (CV_32FC3), scale back to 0..255 and cast to 8U.
                // For other types, a direct convert will be attempted.
                val scale = if (transformedMat.type() == CvType.CV_32FC3) 255.0 else 1.0
                transformedMat.convertTo(tmp, CvType.CV_8UC3, scale)
                inputMat = tmp
            }
        } catch (e: Exception) {
            AppLogger.warn("Failed to convert input Mat to CV_8UC3: ${e.message}. Proceeding with original Mat.")
            inputMat = transformedMat
        }

        // 2) Run model
        val raw = engine.run(inputMat)
        val shapes = raw.shapes

        // 3) Parse detections from raw detection buffer
        val features = shapes.detShape.getOrNull(1)
            ?: error("Unexpected detections shape: ${shapes.detShape.joinToString()}")
        val proposals = shapes.detShape.getOrNull(2)
            ?: error("Unexpected detections shape: ${shapes.detShape.joinToString()}")
        val numClasses = effectiveConfig.labels.size

        val parsed: List<Detection> = YoloPostprocessor.parseDetections(
            buffer = raw.detections,
            numProposals = proposals,
            featuresPerProposal = features,
            numClasses = numClasses,
            scoreThreshold = effectiveConfig.scoreThreshold,
            allowedClassIndices = effectiveConfig.vehicleClassIndices
        )
        val kept = YoloPostprocessor.applyNMS(
            detections = parsed,
            iouThreshold = effectiveConfig.iouThreshold,
            topAmountPerClass = effectiveConfig.topAmountPerClass,
            numClasses = numClasses
        )

        // 4) Extract prototype masks and de-interleave once
        val protoShape = shapes.protoShape
        val prototypes = YoloMaskAssembler.extractPrototypeMasks(raw.prototypes, protoShape)
        val deinterleaved = try {
            YoloMaskAssembler.deinterleavePrototypes(prototypes, protoShape)
        } catch (e: Exception) {
            AppLogger.warn("Deinterleave prototypes failed, falling back to per-detection path: ${e.message}")
            null
        }

        // Clamp upscale factor to prevent mask bleeding into building facades
        val tightUpscaleFactor = upscaleFactor.coerceIn(1.0f, 1.05f)

        // 5) Prepare an overlay (white) CV_8UC1, subtract per-detection masks
        val overlay = Mat(shapes.inputH, shapes.inputW, CvType.CV_8UC1)
        overlay.setTo(Scalar(255.0))
        if (deinterleaved != null) {
            for (det in kept) {
                try {
                    YoloMaskAssembler.createDetectionMask(
                        detection = det,
                        overlayGray = overlay,
                        upscaleFactor = tightUpscaleFactor,
                        deinterleavedPrototypes = deinterleaved,
                        inputWidth = shapes.inputW,
                        inputHeight = shapes.inputH
                    )
                } catch (e: Exception) {
                    AppLogger.warn("Mask assembly failed for one detection: ${e.message}")
                }
            }
            // Release temporary prototype mats once
            deinterleaved.forEach { it.release() }
        } else {
            for (det in kept) {
                try {
                    YoloMaskAssembler.createDetectionMask(
                        detection = det,
                        overlayGray = overlay,
                        upscaleFactor = tightUpscaleFactor,
                        prototypeMasksData = prototypes,
                        inputWidth = shapes.inputW,
                        inputHeight = shapes.inputH,
                        protoShape = protoShape
                    )
                } catch (e: Exception) {
                    AppLogger.warn("Mask assembly failed for one detection: ${e.message}")
                }
            }
        }

        // Optional: remove letterbox padding and/or resize back to original
        var maskMat = overlay
        try {
            val hasOriginal =
                (originalWidth != null && originalHeight != null && originalWidth > 0 && originalHeight > 0)
            val contentW =
                kotlin.math.max(1, kotlin.math.min(shapes.inputW, (shapes.inputW / xRatio).toInt()))
            val contentH =
                kotlin.math.max(1, kotlin.math.min(shapes.inputH, (shapes.inputH / yRatio).toInt()))

            if (contentW != shapes.inputW || contentH != shapes.inputH) {
                val roi = Rect(0, 0, contentW, contentH)
                val cropped = Mat(maskMat, roi).clone()
                if (maskMat !== cropped) maskMat.release()
                maskMat = cropped
            }

            if (hasOriginal) {
                val resized = Mat()
                Imgproc.resize(
                    maskMat,
                    resized,
                    Size(originalWidth.toDouble(), originalHeight.toDouble()),
                    0.0,
                    0.0,
                    Imgproc.INTER_NEAREST
                )
                if (maskMat !== resized) maskMat.release()
                maskMat = resized
            }
        } catch (e: Exception) {
            AppLogger.warn("YoloServiceImpl post-crop/resize failed: ${e.message}")
        }

        return YoloResult(mask = maskMat, detections = kept)
    }

    override fun close() {
        engine.close()
    }
}