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
        val pixelsPerChannel = prototypeHeight * prototypeWidth
        require(prototypeMasksData.size == numPrototypesChannels * pixelsPerChannel) {
            "Prototype data size mismatch: expected ${numPrototypesChannels * pixelsPerChannel}, got ${prototypeMasksData.size}"
        }

        // B8 & RF-36: Reuse single pre-allocated channel buffer to eliminate 31 intermediate GC allocations per frame
        val channelBuffer = FloatArray(pixelsPerChannel)
        val resultList = ArrayList<Mat>(numPrototypesChannels)
        for (c in 0 until numPrototypesChannels) {
            for (i in 0 until pixelsPerChannel) {
                channelBuffer[i] = prototypeMasksData[i * numPrototypesChannels + c]
            }
            val mat = Mat(prototypeHeight, prototypeWidth, CvType.CV_32FC1)
            mat.put(0, 0, channelBuffer)
            resultList.add(mat)
        }
        return resultList
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
        val numPrototypesChannels = prototypeMats.size

        if (maskCoefficients.size != numPrototypesChannels) {
            AppLogger.debug("assembleMask: Mask coeffs size mismatch. Expected $numPrototypesChannels, got ${maskCoefficients.size}")
            return Mat()
        }

        val combinedProtoMask =
            cropAndWeightPrototypes(maskCoefficients, prototypeMats, boxX, boxY, boxW, boxH)

        val targetWidth = (boxW * inputWidth * upscaleFactor).toInt().coerceAtLeast(1)
        val targetHeight = (boxH * inputHeight * upscaleFactor).toInt().coerceAtLeast(1)

        return postProcessMask(combinedProtoMask, targetWidth, targetHeight)
    }

    private fun cropAndWeightPrototypes(
        maskCoefficients: FloatArray,
        prototypeMats: List<Mat>,
        boxX: Float,
        boxY: Float,
        boxW: Float,
        boxH: Float
    ): Mat {
        val prototypeHeight = prototypeMats[0].rows()
        val prototypeWidth = prototypeMats[0].cols()
        val numPrototypesChannels = prototypeMats.size

        // D1.1: Use pure math helper
        val cp =
            YoloMaskMath.calculateCropRect(boxX, boxY, boxW, boxH, prototypeWidth, prototypeHeight)
        val cropRect = Rect(cp.x, cp.y, cp.width, cp.height)

        val combinedProtoMask = Mat.zeros(cropRect.height, cropRect.width, CvType.CV_32FC1)
        val weighted = Mat()
        for (i in 0 until numPrototypesChannels) {
            val coeff = maskCoefficients[i]
            if (coeff == 0f) continue
            val cropped = Mat(prototypeMats[i], cropRect)
            Core.multiply(cropped, Scalar(coeff.toDouble()), weighted)
            Core.add(combinedProtoMask, weighted, combinedProtoMask)
            cropped.release()
        }
        weighted.release()
        return combinedProtoMask
    }

    private fun postProcessMask(combinedProtoMask: Mat, targetWidth: Int, targetHeight: Int): Mat {
        // 1. Upscale continuous linear logits to high resolution FIRST
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
        val closeKernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            Size(MORPH_KERNEL_SIZE_PX, MORPH_KERNEL_SIZE_PX)
        )
        Imgproc.morphologyEx(resizedMask, resizedMask, Imgproc.MORPH_CLOSE, closeKernel)
        closeKernel.release()

        // 5. Slight morphological dilation to cover drop shadows & edge anti-aliasing
        val dilateKernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            Size(5.0, 5.0)
        )
        Imgproc.dilate(resizedMask, resizedMask, dilateKernel)
        dilateKernel.release()

        // Convert to 8-bit for overlay
        resizedMask.convertTo(resizedMask, CvType.CV_8U, OPENCV_BYTE_SCALE)
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
        val maskMat = assembleMaskFromPrototypes(
            detection.maskCoefficients,
            prototypeMasksData,
            detection.x, detection.y, detection.width, detection.height,
            upscaleFactor,
            inputWidth,
            inputHeight,
            protoShape
        )

        if (!maskMat.empty()) {
            applyMaskToOverlay(detection, maskMat, overlayGray, inputWidth, inputHeight)
        } else {
            AppLogger.debug("assembleMaskFromPrototypes returned empty mask. Skipping application.")
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
        val maskMat = assembleMaskFromPrototypes(
            detection.maskCoefficients,
            deinterleavedPrototypes,
            detection.x, detection.y, detection.width, detection.height,
            upscaleFactor,
            inputWidth,
            inputHeight
        )

        if (!maskMat.empty()) {
            applyMaskToOverlay(detection, maskMat, overlayGray, inputWidth, inputHeight)
        } else {
            AppLogger.debug("assembleMaskFromPrototypes returned empty mask. Skipping application.")
        }
        maskMat.release()
    }

    private fun applyMaskToOverlay(
        detection: Detection,
        maskMat: Mat,
        overlayGray: Mat,
        inputWidth: Int,
        inputHeight: Int
    ) {
        val upscaledMaskWidth = maskMat.cols()
        val upscaledMaskHeight = maskMat.rows()

        // D1.1: Use pure math helper
        val p = YoloMaskMath.calculatePlacement(
            detection.x, detection.y, detection.width, detection.height,
            upscaledMaskWidth, upscaledMaskHeight, inputWidth, inputHeight
        )

        if (p.dst.width > 0 && p.dst.height > 0) {
            val roiRect = Rect(p.dst.x, p.dst.y, p.dst.width, p.dst.height)
            val maskRoiRect = Rect(p.src.x, p.src.y, p.src.width, p.src.height)
            
            val dstRoi = Mat(overlayGray, roiRect)
            val srcMaskRoi = Mat(maskMat, maskRoiRect)
            Core.subtract(dstRoi, srcMaskRoi, dstRoi)
            dstRoi.release()
            srcMaskRoi.release()
        } else {
            AppLogger.debug("Warning: ROI for mask placement is invalid or out of bounds.")
        }
    }
}
