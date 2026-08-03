package de.konradvoelkel.android.autokorrektur.ml.mask

import de.konradvoelkel.android.autokorrektur.ml.model.Detection
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * OpenCV-bound utilities for YOLO segmentation mask assembly.
 *
 * This is a near-verbatim extraction from the legacy YoloInferenceTFLite class,
 * adapted to be independent from Interpreter state. All required shapes and
 * dimensions are passed as parameters.
 */
object YoloMaskAssembler {

    private const val MORPH_KERNEL_SIZE_PX = 3.0
    private const val OPENCV_BYTE_SCALE = 255.0

    /**
     * Extracts prototype masks (float32) from a raw ByteBuffer according to the
     * provided prototype tensor shape [1, H, W, C]. The buffer must contain
     * H*W*C floats.
     */
    fun extractPrototypeMasks(buffer: java.nio.ByteBuffer, protoShape: IntArray): FloatArray {
        require(protoShape.size == 4) { "Prototype tensor shape must be [1, H, W, C]" }
        val prototypeHeight = protoShape[1]
        val prototypeWidth = protoShape[2]
        val numPrototypesChannels = protoShape[3]

        val prototypeMaskSize = numPrototypesChannels * prototypeHeight * prototypeWidth
        val expectedBufferSize = prototypeMaskSize * 4 // 4 bytes per float

        if (buffer.capacity() < expectedBufferSize) {
            throw IllegalStateException("Prototype masks buffer too small: ${buffer.capacity()} bytes, expected at least $expectedBufferSize bytes. Model output shape mismatch for prototypes?")
        }
        val prototypeMasks = FloatArray(prototypeMaskSize)
        buffer.asFloatBuffer().get(prototypeMasks)
        AppLogger.debug("Successfully extracted prototype masks: ${prototypeMasks.size} values")
        return prototypeMasks
    }

    /** De-interleave prototype data into per-channel Mats (once per inference). */
    fun deinterleavePrototypes(prototypeMasksData: FloatArray, protoShape: IntArray): List<Mat> {
        require(protoShape.size == 4) { "Prototype tensor shape must be [1, H, W, C]" }
        val prototypeHeight = protoShape[1]
        val prototypeWidth = protoShape[2]
        val numPrototypesChannels = protoShape[3]
        require(prototypeMasksData.size == numPrototypesChannels * prototypeHeight * prototypeWidth) {
            "Prototype data size mismatch: expected ${numPrototypesChannels * prototypeHeight * prototypeWidth}, got ${prototypeMasksData.size}"
        }
        val prototypeMats =
            List(numPrototypesChannels) { Mat(prototypeHeight, prototypeWidth, CvType.CV_32FC1) }
        for (y in 0 until prototypeHeight) {
            for (x in 0 until prototypeWidth) {
                val base =
                    (y * prototypeWidth * numPrototypesChannels) + (x * numPrototypesChannels)
                for (c in 0 until numPrototypesChannels) {
                    val value = prototypeMasksData[base + c]
                    prototypeMats[c].put(y, x, value.toDouble())
                }
            }
        }
        return prototypeMats
    }

    /**
     * Apply a sigmoid element-wise on a CV_32F Mat in-place.
     */
    fun applySigmoid(mat: Mat) {
        // 1 / (1 + exp(-x)) implemented through OpenCV primitives
        // Using: y = -mat; exp(y, y); y += 1; divide(1, y, mat)
        val temp = Mat()
        Core.multiply(mat, Scalar(-1.0), temp)
        Core.exp(temp, temp)
        Core.add(temp, Scalar(1.0), temp)
        Core.divide(1.0, temp, mat)
        temp.release()
    }

    /** Overload that consumes deinterleaved prototypes. */
    fun assembleMaskFromPrototypes(
        maskCoefficients: FloatArray,
        prototypeMats: List<Mat>,
        boxX: Float,
        boxY: Float,
        boxW: Float,
        boxH: Float, // Normalized (0-1)
        upscaleFactor: Float,
        inputWidth: Int,
        inputHeight: Int
    ): Mat {
        if (prototypeMats.isEmpty()) return Mat()
        val prototypeHeight = prototypeMats[0].rows()
        val prototypeWidth = prototypeMats[0].cols()
        val numPrototypesChannels = prototypeMats.size

        if (maskCoefficients.size != numPrototypesChannels) {
            AppLogger.debug("assembleMask: Mask coeffs size mismatch. Expected $numPrototypesChannels, got ${maskCoefficients.size}")
            return Mat()
        }

        // Crop area in prototype grid corresponding to the detection bbox
        val cropX = (boxX * prototypeWidth).toInt().coerceIn(0, prototypeWidth - 1)
        val cropY = (boxY * prototypeHeight).toInt().coerceIn(0, prototypeHeight - 1)
        val cropW =
            (boxW * prototypeWidth).toInt().coerceAtLeast(1).coerceAtMost(prototypeWidth - cropX)
        val cropH =
            (boxH * prototypeHeight).toInt().coerceAtLeast(1).coerceAtMost(prototypeHeight - cropY)
        val cropRect = Rect(cropX, cropY, cropW, cropH)
        AppLogger.debug("Cropping prototypes: x=$cropX, y=$cropY, w=$cropW, h=$cropH (from ${prototypeWidth}x${prototypeHeight})")

        val combinedProtoMask = Mat.zeros(cropH, cropW, CvType.CV_32FC1)
        val weighted = Mat()
        var nonZeroCoeffs = 0
        for (i in 0 until numPrototypesChannels) {
            val coeff = maskCoefficients[i]
            if (coeff == 0f) continue
            nonZeroCoeffs++
            val cropped = Mat(prototypeMats[i], cropRect)
            Core.multiply(cropped, Scalar(coeff.toDouble()), weighted)
            Core.add(combinedProtoMask, weighted, combinedProtoMask)
            cropped.release()
        }
        weighted.release()
        AppLogger.debug("Used $nonZeroCoeffs non-zero coefficients out of $numPrototypesChannels")

        // 1. Upscale continuous linear logits to high resolution FIRST (prevents blocky binary staircase edges)
        val targetWidth = (boxW * inputWidth * upscaleFactor).toInt().coerceAtLeast(1)
        val targetHeight = (boxH * inputHeight * upscaleFactor).toInt().coerceAtLeast(1)
        val resizedMask = Mat()
        Imgproc.resize(
            combinedProtoMask,
            resizedMask,
            Size(targetWidth.toDouble(), targetHeight.toDouble()),
            0.0,
            0.0,
            Imgproc.INTER_CUBIC
        )
        combinedProtoMask.release()

        // 2. Apply Sigmoid on high-resolution continuous probabilities
        applySigmoid(resizedMask)

        // 3. High-resolution thresholding
        Imgproc.threshold(resizedMask, resizedMask, 0.4, 1.0, Imgproc.THRESH_BINARY)

        // 4. Morphological closing to fill glare holes and smooth vehicle contour
        val kernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            Size(MORPH_KERNEL_SIZE_PX, MORPH_KERNEL_SIZE_PX)
        )
        Imgproc.morphologyEx(resizedMask, resizedMask, Imgproc.MORPH_CLOSE, kernel)
        kernel.release()

        // Convert to 8-bit for overlay
        resizedMask.convertTo(resizedMask, CvType.CV_8UC1, OPENCV_BYTE_SCALE)
        return resizedMask
    }

    /**
     * Assembles a segmentation mask for a single detection from prototype masks and coefficients.
     * Returns an 8-bit single channel mask (CV_8UC1) sized to the detection box scaled by upscaleFactor.
     *
     * boxX, boxY, boxW, boxH: normalized coordinates (0..1) on input grid with size inputWidth x inputHeight.
     */
    fun assembleMaskFromPrototypes(
        maskCoefficients: FloatArray,
        prototypeMasksData: FloatArray, // Flat array: C * H * W, interleaved per pixel
        boxX: Float,
        boxY: Float,
        boxW: Float,
        boxH: Float, // Normalized (0-1)
        upscaleFactor: Float,
        inputWidth: Int,
        inputHeight: Int,
        protoShape: IntArray
    ): Mat {
        val mats = deinterleavePrototypes(prototypeMasksData, protoShape)
        val result = assembleMaskFromPrototypes(
            maskCoefficients,
            mats,
            boxX, boxY, boxW, boxH,
            upscaleFactor,
            inputWidth,
            inputHeight
        )
        // Release temporary mats created here
        mats.forEach { it.release() }
        return result
    }

    /**
     * Creates and applies the detection mask to the overlay (mutates overlayGray in-place).
     * overlayGray must be CV_8UC1, sized to inputWidth x inputHeight.
     */
    fun createDetectionMask(
        detection: Detection,
        overlayGray: Mat,
        upscaleFactor: Float,
        prototypeMasksData: FloatArray,
        inputWidth: Int,
        inputHeight: Int,
        protoShape: IntArray
    ) {
        val boxX = detection.x
        val boxY = detection.y
        val boxW = detection.width
        val boxH = detection.height

        val maskMat = assembleMaskFromPrototypes(
            detection.maskCoefficients,
            prototypeMasksData,
            boxX, boxY, boxW, boxH,
            upscaleFactor,
            inputWidth,
            inputHeight,
            protoShape
        )

        if (maskMat.empty()) {
            AppLogger.debug("assembleMaskFromPrototypes returned empty mask. Skipping application.")
            return
        }

        val upscaledMaskWidth = maskMat.cols().toDouble()
        val upscaledMaskHeight = maskMat.rows().toDouble()

        val xModel = (boxX * inputWidth).toInt()
        val yModel = (boxY * inputHeight).toInt()
        val wModel = (boxW * inputWidth).toInt()
        val hModel = (boxH * inputHeight).toInt()

        val targetX = xModel + (wModel / 2.0) - (upscaledMaskWidth / 2.0)
        val targetY = yModel + (hModel / 2.0) - (upscaledMaskHeight / 2.0)

        val roiRect = Rect(
            kotlin.math.max(0, targetX.toInt()),
            kotlin.math.max(0, targetY.toInt()),
            kotlin.math.min(
                upscaledMaskWidth.toInt(),
                inputWidth - kotlin.math.max(0, targetX.toInt())
            ),
            kotlin.math.min(
                upscaledMaskHeight.toInt(),
                inputHeight - kotlin.math.max(0, targetY.toInt())
            )
        )

        val maskRoiRect = Rect(
            0, 0,
            kotlin.math.min(upscaledMaskWidth.toInt(), roiRect.width),
            kotlin.math.min(upscaledMaskHeight.toInt(), roiRect.height)
        )

        if (roiRect.width > 0 && roiRect.height > 0) {
            val dstRoi = Mat(overlayGray, roiRect)
            val srcMaskRoi = Mat(maskMat, maskRoiRect)
            Core.subtract(dstRoi, srcMaskRoi, dstRoi)
            dstRoi.release()
            srcMaskRoi.release()
        } else {
            AppLogger.debug("Warning: ROI for mask placement is invalid or out of bounds. Skipping mask application for this detection.")
        }
        maskMat.release()
    }

    /** Overload to consume deinterleaved prototypes without redoing the work per detection. */
    fun createDetectionMask(
        detection: Detection,
        overlayGray: Mat,
        upscaleFactor: Float,
        deinterleavedPrototypes: List<Mat>,
        inputWidth: Int,
        inputHeight: Int
    ) {
        val boxX = detection.x
        val boxY = detection.y
        val boxW = detection.width
        val boxH = detection.height

        val maskMat = assembleMaskFromPrototypes(
            detection.maskCoefficients,
            deinterleavedPrototypes,
            boxX, boxY, boxW, boxH,
            upscaleFactor,
            inputWidth,
            inputHeight
        )

        if (maskMat.empty()) {
            AppLogger.debug("assembleMaskFromPrototypes returned empty mask. Skipping application.")
            return
        }

        val upscaledMaskWidth = maskMat.cols().toDouble()
        val upscaledMaskHeight = maskMat.rows().toDouble()

        val xModel = (boxX * inputWidth).toInt()
        val yModel = (boxY * inputHeight).toInt()
        val wModel = (boxW * inputWidth).toInt()
        val hModel = (boxH * inputHeight).toInt()

        val targetX = xModel + (wModel / 2.0) - (upscaledMaskWidth / 2.0)
        val targetY = yModel + (hModel / 2.0) - (upscaledMaskHeight / 2.0)

        val roiRect = Rect(
            kotlin.math.max(0, targetX.toInt()),
            kotlin.math.max(0, targetY.toInt()),
            kotlin.math.min(
                upscaledMaskWidth.toInt(),
                inputWidth - kotlin.math.max(0, targetX.toInt())
            ),
            kotlin.math.min(
                upscaledMaskHeight.toInt(),
                inputHeight - kotlin.math.max(0, targetY.toInt())
            )
        )

        val maskRoiRect = Rect(
            0, 0,
            kotlin.math.min(upscaledMaskWidth.toInt(), roiRect.width),
            kotlin.math.min(upscaledMaskHeight.toInt(), roiRect.height)
        )

        if (roiRect.width > 0 && roiRect.height > 0) {
            val dstRoi = Mat(overlayGray, roiRect)
            val srcMaskRoi = Mat(maskMat, maskRoiRect)
            Core.subtract(dstRoi, srcMaskRoi, dstRoi)
            dstRoi.release()
            srcMaskRoi.release()
        } else {
            AppLogger.debug("Warning: ROI for mask placement is invalid or out of bounds. Skipping mask application for this detection.")
        }
        maskMat.release()
    }
}