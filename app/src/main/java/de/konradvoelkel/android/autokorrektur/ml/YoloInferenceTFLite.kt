@file:Suppress("unused", "UNUSED_PARAMETER", "SameParameterValue")
package de.konradvoelkel.android.autokorrektur.ml

import android.annotation.SuppressLint
import android.content.Context
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.model.Detection
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.io.IOException
import java.nio.ByteBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * TensorFlow Lite implementation of YOLO model inference for car segmentation.
 * Uses TensorFlow Lite Interpreter directly.
 */
@Deprecated("Use YoloService/YoloServiceImpl instead. This class is now a thin adapter and will be removed in a future release.")
@SuppressLint("DefaultLocale")
class YoloInferenceTFLite(private val context: Context) {

    // Determine debuggability using the official ApplicationInfo flag.
    // Avoid reflection on BuildConfig.DEBUG; it can be unreliable across variants.

    // Thin adapter over the new modular service
    private val service: YoloService by lazy { YoloServiceImpl(context) }

    private var interpreter: Interpreter? = null
    private var isInitialized = false

    // Model input/output dimensions (derived from model during initialization)
    private var inputWidth = 640 // Default, will be updated from model
    private var inputHeight = 640 // Default, will be updated from model
    private var inputChannels = 3  // Default, will be updated from model

    // Cached shapes and reusable buffers to reduce per-inference allocations
    private var detectionTensorShape: IntArray? = null
    private var prototypeTensorShape: IntArray? = null
    private var inputBuffer: ByteBuffer? = null
    private val outputBuffers = mutableMapOf<Int, ByteBuffer>()

    // Labels from COCO dataset
    private val labels = arrayOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    )

    // Configuration parameters - aligned with JS reference
    private val scoreThreshold =
        0.6f // Slightly raised to reduce false positives while retaining recall for vehicle classes
    private val iouThreshold = 0.9f   // IoU threshold for Non-Maximum Suppression
    private val topAmountPerClass = 100 // top amount of Instances per class

    // Vehicle class indices (car=2, motorcycle=3, bus=5, truck=7) - COCO indices
    private val vehicleClassIndices = intArrayOf(2, 3, 5, 7)

    // Standard number of mask coefficients for YOLO segmentation models
    //private val numMaskCoefficients = 32

    @Throws(IOException::class)
    fun initialize(modelName: String = "yolo11s", useFP16: Boolean = false) {
        if (isInitialized) {
            AppLogger.debug("TFLite YoloInference already initialized")
            return
        }
        // Delegate to modular service
        try {
            service.initialize(modelName, useFP16)
            isInitialized = true
            AppLogger.debug("YoloInferenceTFLite (adapter) initialized via YoloServiceImpl")
        } catch (e: Exception) {
            throw IOException("Failed to initialize YoloService: ${e.message}", e)
        }
    }

    @Throws(IOException::class)
    fun inferYolo(
        transformedMat: Mat, // Compatible types; service handles conversion
        xRatio: Float,
        yRatio: Float,
        upscaleFactor: Float = 1.0f,
        @Suppress("UNUSED_PARAMETER") downshiftFactor: Float = 0.0f, // Kept for API compatibility
        originalWidth: Int? = null,
        originalHeight: Int? = null
    ): Mat {
        if (!isInitialized) {
            AppLogger.debug("Adapter not initialized. Initializing service now...")
            initialize()
        }
        return try {
            service.infer(
                transformedMat = transformedMat,
                xRatio = xRatio,
                yRatio = yRatio,
                upscaleFactor = upscaleFactor,
                originalWidth = originalWidth,
                originalHeight = originalHeight
            )
        } catch (e: Exception) {
            AppLogger.warn("YoloInferenceTFLite adapter inference failed: ${e.message}")
            // Fall back to an empty white mask sized to the provided transformedMat
            val w = transformedMat.cols().coerceAtLeast(1)
            val h = transformedMat.rows().coerceAtLeast(1)
            val overlay = Mat.ones(h, w, CvType.CV_8UC1)
            overlay.setTo(Scalar(255.0))
            overlay
        }
    }
    







    private fun calculateIoU(det1: Detection, det2: Detection): Float {
        // Normalized coordinates (0-1 range)
        val xA = max(det1.x, det2.x)
        val yA = max(det1.y, det2.y)
        val xB = min(det1.x + det1.width, det2.x + det2.width)
        val yB = min(det1.y + det1.height, det2.y + det2.height)

        // Intersection area
        val intersectionWidth = max(0f, xB - xA)
        val intersectionHeight = max(0f, yB - yA)
        val intersectionArea = intersectionWidth * intersectionHeight
        if (intersectionArea <= 0f) return 0f

        // Individual areas
        val area1 = det1.width * det1.height
        val area2 = det2.width * det2.height

        // Union area
        val unionArea = area1 + area2 - intersectionArea

        return if (unionArea > 0.000001f) intersectionArea / unionArea else 0f // Avoid division by zero or tiny union
    }

    /**
     * Creates and applies the detection mask to the overlay.
     * Aligned with JS reference's decodeMask and mask overlay logic.
     */
    private fun createDetectionMask(
        detection: Detection,
        overlayGray: Mat, // Full size overlay (e.g. 640x640, CV_8UC1) to draw upon
        upscaleFactor: Float,
        prototypeMasksData: FloatArray // Flat array of all prototype masks
    ) {
        // Normalize bounding box coordinates to model input dimensions (0-1 range)
        val boxX = detection.x
        val boxY = detection.y
        val boxW = detection.width
        val boxH = detection.height

        AppLogger.debug("=== CREATING DETECTION MASK ===")
        AppLogger.debug("Detection bbox (normalized): x=$boxX, y=$boxY, w=$boxW, h=$boxH")
        AppLogger.debug("Detection class=${detection.classId}, confidence=${detection.confidence}")
        AppLogger.debug("Upscale factor: $upscaleFactor")
        AppLogger.debug("Prototype masks data size: ${prototypeMasksData.size}")
        AppLogger.debug("Mask coefficients size: ${detection.maskCoefficients.size}")
        AppLogger.debug(
            "Mask coefficients: [${
                detection.maskCoefficients.take(5).joinToString(", ")
            }${if (detection.maskCoefficients.size > 5) "..." else ""}]"
        )

        try {
            // This returns a CV_8UC1 mask of size (boxW*upscaleFactor) x (boxH*upscaleFactor)
            val mask_mat = assembleMaskFromPrototypes(
                detection.maskCoefficients,
                prototypeMasksData,
                boxX, boxY, boxW, boxH, // Pass normalized box coords and dimensions
                upscaleFactor
            )

            if (!mask_mat.empty()) {
                // Calculate position for overlaying the mask on the full overlayGray
                val upscaledMaskWidth = mask_mat.cols().toDouble()
                val upscaledMaskHeight = mask_mat.rows().toDouble()

                // Convert normalized box to pixel coordinates on the model input size
                val x_model_px = (boxX * inputWidth).toInt()
                val y_model_px = (boxY * inputHeight).toInt()
                val w_model_px = (boxW * inputWidth).toInt()
                val h_model_px = (boxH * inputHeight).toInt()

                // Center the upscaled mask on the original box center (in pixel coords)
                val targetX_px = x_model_px + (w_model_px / 2.0) - (upscaledMaskWidth / 2.0)
                val targetY_px = y_model_px + (h_model_px / 2.0) - (upscaledMaskHeight / 2.0)

                // Create a ROI (Region of Interest) on the overlay_gray where the mask will be placed
                val roiRect = Rect(
                    max(0, targetX_px.toInt()),
                    max(0, targetY_px.toInt()),
                    min(upscaledMaskWidth.toInt(), inputWidth - max(0, targetX_px.toInt())),
                    min(upscaledMaskHeight.toInt(), inputHeight - max(0, targetY_px.toInt()))
                )

                // Ensure mask_mat is not larger than the ROI
                val maskRoiRect = Rect(
                    0, 0,
                    min(upscaledMaskWidth.toInt(), roiRect.width),
                    min(upscaledMaskHeight.toInt(), roiRect.height)
                )

                if (roiRect.width > 0 && roiRect.height > 0) {
                    val dstRoi = Mat(overlayGray, roiRect)
                    val srcMaskRoi = Mat(mask_mat, maskRoiRect)

                    // Subtract mask from overlay so masked area will be black
                    // Ensure both Mats have the same type and size for subtraction
                    Core.subtract(dstRoi, srcMaskRoi, dstRoi)

                    dstRoi.release() // Release the ROI Mat
                    srcMaskRoi.release() // Release the ROI Mat
                } else {
                    AppLogger.debug("Warning: ROI for mask placement is invalid or out of bounds. Skipping mask application for this detection.")
                }
                mask_mat.release() // Release the mask Mat
            } else {
                AppLogger.debug("assembleMaskFromPrototypes returned empty mask. Skipping application.")
            }

        } catch (e: Exception) {
            AppLogger.debug("CRITICAL ERROR in mask assembly/application for one detection: ${e.message}")
            e.printStackTrace()
            // Do not rethrow here, to allow other detections to be processed if one fails
        }
    }

    /**
     * Assembles a segmentation mask for a single detection from prototype masks and coefficients.
     * Aligned with JS decodeMask function.
     *
     * @param boxX, boxY, boxW, boxH: Normalized (0-1) bounding box of the detection (x_min, y_min, width, height).
     * @return A Mat (CV_8UC1) representing the decoded and resized mask for the specific object.
     */
    private fun assembleMaskFromPrototypes(
        maskCoefficients: FloatArray,
        prototypeMasksData: FloatArray, // Flat array: num_prototypes_channels * proto_height * proto_width
        boxX: Float,
        boxY: Float,
        boxW: Float,
        boxH: Float, // Normalized (0-1) bounding box
        upscaleFactor: Float
    ): Mat {
        AppLogger.debug("=== ASSEMBLING MASK FROM PROTOTYPES ===")
        val protoTensorShape = interpreter!!.getOutputTensor(1).shape() // e.g. [1, 160, 160, 32]
        val prototypeHeight = protoTensorShape[1]       // 160
        val prototypeWidth = protoTensorShape[2]        // 160
        val numPrototypesChannels = protoTensorShape[3] // 32

        AppLogger.debug("Prototype tensor shape: [${protoTensorShape.joinToString(", ")}]")
        AppLogger.debug("Prototype dims: $numPrototypesChannels channels, ${prototypeHeight}x${prototypeWidth}")
        AppLogger.debug("Expected prototype data size: ${numPrototypesChannels * prototypeHeight * prototypeWidth}")
        AppLogger.debug("Actual prototype data size: ${prototypeMasksData.size}")

        if (maskCoefficients.size != numPrototypesChannels) {
            AppLogger.debug("assembleMask: Mask coeffs size mismatch. Expected $numPrototypesChannels, got ${maskCoefficients.size}")
            return Mat() // Return empty Mat on error
        }
        if (prototypeMasksData.size != numPrototypesChannels * prototypeHeight * prototypeWidth) {
            AppLogger.debug("assembleMask: Prototype data size mismatch. Expected ${numPrototypesChannels * prototypeHeight * prototypeWidth}, got ${prototypeMasksData.size}")
            return Mat()
        }
        // 1. De-interleave the flat data and combine the prototypes
        AppLogger.debug("Step 1: De-interleaving and combining $numPrototypesChannels prototype masks")

        // Create a list to hold the 32 correctly-structured prototype Mats
        val prototypeMats = List(numPrototypesChannels) {
            Mat(prototypeHeight, prototypeWidth, CvType.CV_32FC1)
        }

        // === DE-INTERLEAVING LOGIC ===
        // Iterate over each pixel's location (y, x)
        for (y in 0 until prototypeHeight) {
            for (x in 0 until prototypeWidth) {
                // For each pixel, iterate through all the channels (prototypes)
                for (channel in 0 until numPrototypesChannels) {
                    // Calculate the index in the flat, interleaved source array
                    val sourceIndex =
                        (y * prototypeWidth * numPrototypesChannels) + (x * numPrototypesChannels) + channel

                    // Get the value and put it in the correct Mat at the correct (y, x) location
                    val value = prototypeMasksData[sourceIndex]
                    prototypeMats[channel].put(y, x, value.toDouble())
                }
            }
        }
        AppLogger.debug("Successfully de-interleaved data into ${prototypeMats.size} Mats.")


        // === CROP PROTOTYPE MASKS TO BOUNDING BOX ===
        // Convert normalized box coordinates to prototype mask coordinates
        val cropX = (boxX * prototypeWidth).toInt().coerceIn(0, prototypeWidth - 1)
        val cropY = (boxY * prototypeHeight).toInt().coerceIn(0, prototypeHeight - 1)
        val cropW =
            (boxW * prototypeWidth).toInt().coerceAtLeast(1).coerceAtMost(prototypeWidth - cropX)
        val cropH =
            (boxH * prototypeHeight).toInt().coerceAtLeast(1).coerceAtMost(prototypeHeight - cropY)

        AppLogger.debug("Cropping prototypes: x=$cropX, y=$cropY, w=$cropW, h=$cropH (from ${prototypeWidth}x${prototypeHeight})")

        val cropRect = Rect(cropX, cropY, cropW, cropH)

        // === COMBINATION LOGIC (Corrected with cropping) ===
        val combinedProtoMask = Mat.zeros(cropH, cropW, CvType.CV_32FC1)
        val weightedProtoMat = Mat() // Reusable Mat for the multiplication result
        var nonZeroCoeffs = 0

        for (i in 0 until numPrototypesChannels) {
            val coeff = maskCoefficients[i]
            if (coeff == 0f) continue
            nonZeroCoeffs++

            // Get the correctly structured prototype Mat and crop it to bounding box
            val singleProtoMat = prototypeMats[i]
            //if (isDebugBuild) {
            //    matToBitmapForDebug(singleProtoMat)
            //}
            val croppedProtoMat = Mat(singleProtoMat, cropRect)
            //if (isDebugBuild) {
            //    matToBitmapForDebug(croppedProtoMat)
            //}


            // Step 1: Multiply the cropped prototype by its coefficient using Core.multiply
            Core.multiply(croppedProtoMat, Scalar(coeff.toDouble()), weightedProtoMat)

            // Step 2: Add the weighted result to the combined mask
            Core.add(combinedProtoMask, weightedProtoMat, combinedProtoMask)
            //if (isDebugBuild) {
            //    matToBitmapForDebug(combinedProtoMask)
            //}
            croppedProtoMat.release()
        }

        // Clean up memory
        weightedProtoMat.release()
        prototypeMats.forEach { it.release() }

        AppLogger.debug("Used $nonZeroCoeffs non-zero coefficients out of $numPrototypesChannels")

        // === DEBUG VIEW 1: After combining all prototypes ===
        //var debugBitmap1: Bitmap? = null
        //var debugBitmap2: Bitmap? = null
        //var debugBitmap3: Bitmap? = null
        //var debugBitmap4: Bitmap? = null
        //var debugBitmap5: Bitmap? = null

        //if (isDebugBuild) {
        //    debugBitmap1 = matToBitmapForDebug(combinedProtoMask)
        //}
        // Check combined mask statistics
        val minMaxResult = Core.minMaxLoc(combinedProtoMask)
        AppLogger.debug("Combined mask before sigmoid: min=${minMaxResult.minVal}, max=${minMaxResult.maxVal}")

        // 2. Apply sigmoid to the combined 160x160 mask (values become 0-1)
        AppLogger.debug("Step 2: Applying sigmoid")
        applySigmoid(combinedProtoMask)
        // if (isDebugBuild) {
        //    debugBitmap2 = matToBitmapForDebug(combinedProtoMask)
        // }

        val minMaxAfterSigmoid = Core.minMaxLoc(combinedProtoMask)
        AppLogger.debug("Combined mask after sigmoid: min=${minMaxAfterSigmoid.minVal}, max=${minMaxAfterSigmoid.maxVal}")

        // 3. Threshold the mask (e.g., 0.4) to get binary values
        AppLogger.debug("Step 3: Applying threshold (0.4)")
        Imgproc.threshold(combinedProtoMask, combinedProtoMask, 0.4, 1.0, Imgproc.THRESH_BINARY)


        // if (isDebugBuild) {
        //     debugBitmap3 = matToBitmapForDebug(combinedProtoMask)
        // }

        val minMaxAfterThreshold = Core.minMaxLoc(combinedProtoMask)
        AppLogger.debug("Combined mask after threshold: min=${minMaxAfterThreshold.minVal}, max=${minMaxAfterThreshold.maxVal}")

        // 4. Resize the mask to the bounding box dimensions (scaled by upscaleFactor)
        val dWidth = boxW * inputWidth * upscaleFactor
        val dWidthInt = dWidth.toInt()
        val targetWidth = dWidthInt.coerceAtLeast(1)
        val targetHeight = (boxH * inputHeight * upscaleFactor).toInt().coerceAtLeast(1)
        AppLogger.debug("Step 4: Resizing mask from ${cropW}x${cropH} to ${targetWidth}x${targetHeight}")
        if (targetWidth == 1 && targetHeight == 1) {
            AppLogger.debug("Ahrg in Step 4: target size is 1x1 pixel.")
        }
        val resizedMask = Mat()
        Imgproc.resize(
            combinedProtoMask,
            resizedMask,
            Size(targetWidth.toDouble(), targetHeight.toDouble()),
            0.0,
            0.0,
            Imgproc.INTER_LINEAR
        )
        // if (isDebugBuild) {
        //     debugBitmap4 = matToBitmapForDebug(combinedProtoMask)
        // }

        // 5. Convert to 8UC1 (grayscale) for overlay
        AppLogger.debug("Step 5: Converting to CV_8UC1")
        resizedMask.convertTo(resizedMask, CvType.CV_8UC1, 255.0)
        // if (isDebugBuild) {
        //     debugBitmap5 = matToBitmapForDebug(resizedMask)
        // }

        val finalMinMax = Core.minMaxLoc(resizedMask)
        AppLogger.debug(
            "Final mask: size=${resizedMask.cols()}x${resizedMask.rows()}, type=${
                CvType.typeToString(
                    resizedMask.type()
                )
            }, min=${finalMinMax.minVal}, max=${finalMinMax.maxVal}"
        )

        combinedProtoMask.release()

        return resizedMask
    }

    /**
     * Applies sigmoid function element-wise to a CV_32FC1 Mat.
     */
    private fun applySigmoid(mat: Mat) { // Input/Output Mat is CV_32FC1
        if (mat.type() != CvType.CV_32FC1) {
            AppLogger.warn(
                "applySigmoid: Mat type is not CV_32FC1. Type: ${
                    CvType.typeToString(
                        mat.type()
                    )
                }"
            )
            return
        }
        val data = FloatArray((mat.total() * mat.channels()).toInt())
        mat.get(0, 0, data)
        for (i in data.indices) {
            data[i] = (1.0f / (1.0f + exp(-data[i].toDouble()))).toFloat()
        }
        mat.put(0, 0, data)
    }


    fun close() {
        try {
            service.close()
        } catch (_: Exception) {
        } finally {
            isInitialized = false
        }
        AppLogger.debug("YoloInferenceTFLite adapter closed.")
    }
}
