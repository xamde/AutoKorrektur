package de.konradvoelkel.android.autokorrektur.ml

import android.content.Context
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import de.konradvoelkel.android.autokorrektur.utils.matToBitmapForDebug
import org.opencv.BuildConfig
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
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * TensorFlow Lite implementation of YOLO model inference for car segmentation.
 * Uses TensorFlow Lite Interpreter directly.
 */
class YoloInferenceTFLite(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var isInitialized = false

    // Model input/output dimensions (derived from model during initialization)
    private var inputWidth = 640 // Default, will be updated from model
    private var inputHeight = 640 // Default, will be updated from model
    private var inputChannels = 3  // Default, will be updated from model

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
        0.5f // Lowered from 0.5f to allow more detections - confidence threshold after sigmoid for filtering detections
    private val iouThreshold = 0.9f   // IoU threshold for Non-Maximum Suppression
    private val topAmountPerClass = 100 // top amount of Instances per class

    // Vehicle class indices (car, motorcycle, truck) - COCO indices
    private val vehicleClassIndices = intArrayOf(2, 3, 7) // car, motorcycle, truck; bus=5,

    // Standard number of mask coefficients for YOLO segmentation models
    private val numMaskCoefficients = 32 // is this ever used?

    @Throws(IOException::class)
    fun initialize(modelName: String = "yolo11s", useFP16: Boolean = false) {
        if (isInitialized) {
            AppLogger.debug("TFLite YoloInference already initialized")
            return
        }

        val modelFile = if (useFP16) {
            "model/${modelName}-seg_saved_model/${modelName}-seg_float16.tflite"
        } else {
            "model/${modelName}-seg_saved_model/${modelName}-seg_float32.tflite"
        }

        AppLogger.debug("TFLite YoloInference.initialize() - Loading model: $modelFile")

        try {
            val assetFileDescriptor = context.assets.openFd(modelFile)
            val inputStream = assetFileDescriptor.createInputStream()
            val modelBytes = inputStream.readBytes()
            inputStream.close()
            assetFileDescriptor.close()

            AppLogger.debug("TFLite model loaded: ${modelBytes.size} bytes")

            val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size)
            modelBuffer.order(ByteOrder.nativeOrder())
            modelBuffer.put(modelBytes)
            modelBuffer.rewind()

            val options = Interpreter.Options()
            options.setNumThreads(
                Runtime.getRuntime().availableProcessors()
            ) // Use available processors
            interpreter = Interpreter(modelBuffer, options)

            val inputTensor = interpreter!!.getInputTensor(0)
            val inputShape = inputTensor.shape()
            if (inputShape.size == 4) { // Expected [1, H, W, C]
                inputHeight = inputShape[1]
                inputWidth = inputShape[2]
                inputChannels = inputShape[3]
            } else {
                throw IOException("Unexpected model input shape: ${inputShape.joinToString(",")}")
            }

            AppLogger.debug("Model input shape: [${inputShape.joinToString(", ")}]")
            AppLogger.debug("Input dimensions: ${inputWidth}x${inputHeight}x${inputChannels}")
            AppLogger.debug("Detection threshold: $scoreThreshold")

            isInitialized = true
            AppLogger.debug("TFLite YoloInference initialized successfully")

        } catch (e: Exception) {
            AppLogger.error("========== TFLITE INITIALIZATION FAILED ==========")
            AppLogger.error("Exception type: ${e.javaClass.simpleName}")
            AppLogger.error("Exception message: ${e.message}", e)
            AppLogger.error("================================================")
            throw IOException("Failed to initialize TFLite YOLO model: ${e.message}", e)
        }
    }

    @Throws(IOException::class)
    fun inferYolo(
        transformedMat: Mat, // Expected: CV_32FC3, RGB, Normalized, this.inputWidth x this.inputHeight
        @Suppress("UNUSED_PARAMETER") xRatio: Float, // Letterbox ratio, potentially useful for drawing results on original image
        @Suppress("UNUSED_PARAMETER") yRatio: Float, // Letterbox ratio
        upscaleFactor: Float = 1.0f, // Default to 1.0f as in JS reference
        @Suppress("UNUSED_PARAMETER") downshiftFactor: Float = 0.0f // Re-added for compatibility, but ignored
    ): Mat {
        if (!isInitialized || interpreter == null) {
            AppLogger.debug("Interpreter not initialized. Initializing now...")
            initialize() // This might throw IOException
        }

        // Output mask will be the same size as model input, representing the letterboxed image content
        val overlayGray = Mat.ones(this.inputHeight, this.inputWidth, CvType.CV_8UC1)
        overlayGray.setTo(Scalar(255.0)) // White background

        try {
            // Prepare input buffer
            val inputBufferCapacity =
                4 * this.inputWidth * this.inputHeight * this.inputChannels // 4 bytes per float
            val inputBuffer = ByteBuffer.allocateDirect(inputBufferCapacity)
            inputBuffer.order(ByteOrder.nativeOrder())
            matToByteBuffer(transformedMat, inputBuffer) // Populate inputBuffer from transformedMat

            // Prepare output buffers map
            val outputMap = mutableMapOf<Int, Any>()
            val numOutputs = interpreter!!.outputTensorCount
            AppLogger.debug("Model has $numOutputs outputs")

            for (i in 0 until numOutputs) {
                val outputTensor = interpreter!!.getOutputTensor(i)
                val outputShape = outputTensor.shape()
                AppLogger.debug("Output $i shape: [${outputShape.joinToString(", ")}]")
                // Calculate size needed for the buffer based on tensor shape (product of dimensions) * bytes_per_element
                val outputSize = outputShape.fold(1L) { acc, dim -> acc * dim }
                    .toInt() // Use Long for intermediate to avoid overflow
                val outputBuffer =
                    ByteBuffer.allocateDirect(4 * outputSize) // Assuming Float32 output
                outputBuffer.order(ByteOrder.nativeOrder())
                outputMap[i] = outputBuffer
            }

            // Run inference
            interpreter!!.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

            // Process outputs to create segmentation mask on overlayGray
            processOutputsToMask(
                outputMap,
                overlayGray,
                upscaleFactor
            )

            return overlayGray

        } catch (e: Exception) {
            AppLogger.debug("TFLite inference failed: ${e.message}")
            e.printStackTrace()
            // Return current state of overlayGray even on error
            return overlayGray
        }
    }

    /**
     * Converts an OpenCV Mat to ByteBuffer for TensorFlow Lite input.
     * Assumes 'mat' is already:
     * - Correctly sized (this.inputWidth x this.inputHeight)
     * - CV_32FC3 type
     * - RGB channel order
     * - Normalized pixel values [0, 1]
     * These assumptions rely on ImageProcessor.kt providing the correct Mat.
     */
    private fun matToByteBuffer(mat: Mat, buffer: ByteBuffer) {
        buffer.rewind() // Ensure buffer is ready for writing

        // --- Precondition Checks (Optional but good for debugging) ---
        if (mat.width() != this.inputWidth || mat.height() != this.inputHeight) {
            AppLogger.debug(
                "WARNING: matToByteBuffer input Mat dimensions (${mat.width()}x${mat.height()}) " +
                        "do not match model input dimensions (${this.inputWidth}x${this.inputHeight}). Resizing."
            )
            val tempResizedMat = Mat()
            Imgproc.resize(
                mat,
                tempResizedMat,
                Size(this.inputWidth.toDouble(), this.inputHeight.toDouble())
            )
            val data = FloatArray(this.inputWidth * this.inputHeight * this.inputChannels)
            tempResizedMat.get(0, 0, data)
            for (value in data) {
                buffer.putFloat(value)
            }
            tempResizedMat.release()
            return
        }
        if (mat.type() != CvType.CV_32FC3) {
            AppLogger.warn(
                "CRITICAL WARNING: matToByteBuffer input Mat type is not CV_32FC3. Type: ${
                    CvType.typeToString(
                        mat.type()
                    )
                }"
            )
            val floatMat = Mat()
            val conversionFactor =
                if (CvType.depth(mat.type()) == CvType.CV_8U) 1.0 / 255.0 else 1.0
            mat.convertTo(floatMat, CvType.CV_32FC3, conversionFactor)
            val data = FloatArray(this.inputWidth * this.inputHeight * this.inputChannels)
            floatMat.get(0, 0, data)
            for (value in data) {
                buffer.putFloat(value)
            }
            floatMat.release()
            return
        }
        if (mat.channels() != this.inputChannels) {
            AppLogger.warn(
                "CRITICAL WARNING: matToByteBuffer input Mat channels (${mat.channels()}) " +
                        "do not match model input channels (${this.inputChannels}). Expected RGB."
            )
            throw IllegalArgumentException("Input Mat channels mismatch model expectation (expected ${this.inputChannels} for RGB).")
        }
        // --- End Precondition Checks ---

        // Direct copy if preconditions met
        val data = FloatArray(this.inputWidth * this.inputHeight * this.inputChannels)
        mat.get(0, 0, data) // Assumes mat is CV_32FC3, RGB, Normalized

        for (value in data) {
            buffer.putFloat(value)
        }
    }


    private fun processOutputsToMask(
        outputs: Map<Int, Any>,
        overlayGray: Mat, // Output mask (e.g., 640x640, CV_8UC1)
        upscaleFactor: Float
    ) {
        try {
            AppLogger.debug("Processing model outputs for mask creation")
            val detectionsBuffer = outputs[0] as? ByteBuffer
                ?: throw IllegalStateException("Model output 0 (detections) not found or not a ByteBuffer.")
            val prototypeMasksBuffer = outputs[1] as? ByteBuffer
                ?: throw IllegalStateException("Model output 1 (prototypes) not found or not a ByteBuffer.")

            detectionsBuffer.rewind()
            prototypeMasksBuffer.rewind()

            // Example for detectionsBuffer
            val floatBuffer = detectionsBuffer.asFloatBuffer()
            val sampleData = FloatArray(20)
            floatBuffer.get(sampleData)
            AppLogger.debug("[DEBUG_DUMP] Detections Buffer (first 20 floats): ${sampleData.joinToString()}")
            detectionsBuffer.rewind() // Rewind again for the actual processing

            AppLogger.debug("Detections buffer capacity: ${detectionsBuffer.capacity()} bytes")
            AppLogger.debug("Prototype masks buffer capacity: ${prototypeMasksBuffer.capacity()} bytes")

            // Get detection tensor shape to correctly determine numProposals and featuresPerProposal
            val detectionTensor = interpreter!!.getOutputTensor(0)
            val detectionTensorShape =
                detectionTensor.shape() // [1, 116, 8400] = 1 F N
            val numProposals: Int
            val featuresPerProposal: Int

            // Determine actual shape based on common YOLOv8/v11 TFLite outputs
            if (detectionTensorShape.size == 3) {
                // Assuming [1, features_per_proposal, num_proposals] (column-major)
                featuresPerProposal = detectionTensorShape[1]
                numProposals = detectionTensorShape[2]
            } else {
                throw IllegalStateException("Unexpected detection tensor shape: ${detectionTensorShape.joinToString()}")
            }

            AppLogger.debug("Detection tensor shape: ${detectionTensorShape.joinToString()}, NumProposals: $numProposals, FeaturesPerProposal: $featuresPerProposal")

            val detections = parseDetections(
                detectionsBuffer,
                scoreThreshold, // Use class-level scoreThreshold
                numProposals,
                featuresPerProposal,
                labels.size // Pass actual number of classes
            )
            AppLogger.debug("Detections after score threshold: ${detections.size}")

            val detection = detections.get(0)
            AppLogger.debug("Detection[0].maskCoefficients: ${detection.maskCoefficients}")
            AppLogger.debug("Detection[0].classId: ${detection.classId}")
            AppLogger.debug("Detection[0].confidence: ${detection.confidence}")
            AppLogger.debug("Detection[0].x: ${detection.x}")
            AppLogger.debug("Detection[0].y: ${detection.y}")
            AppLogger.debug("Detection[0].width: ${detection.width}")
            AppLogger.debug("Detection[0].height: ${detection.height}")
            AppLogger.debug("Detection[0].classId: ${detection.classId}")
            AppLogger.debug("Detection[0].confidence: ${detection.confidence}")

            val filteredDetections = applyNMS(detections, iouThreshold)
            AppLogger.debug("Filtered detections after NMS: ${filteredDetections.size}")

            if (filteredDetections.isNotEmpty()) {
                val prototypeMasks = extractPrototypeMasks(prototypeMasksBuffer)
                AppLogger.debug("Creating masks for ${filteredDetections.size} vehicle detections")

                // Check overlayGray state before processing detections
                val beforeMinMax = Core.minMaxLoc(overlayGray)
                AppLogger.debug("OverlayGray before mask creation: min=${beforeMinMax.minVal}, max=${beforeMinMax.maxVal}")

                for (detection in filteredDetections) {
                    AppLogger.debug("Processing detection: class=${if (detection.classId < labels.size && detection.classId >= 0) labels[detection.classId] else "Unknown"} (${detection.classId}), confidence=${detection.confidence}")
                    createDetectionMask(
                        detection,
                        overlayGray,
                        upscaleFactor,
                        prototypeMasks // Pass the flat array
                    )
                }

                // Check overlayGray state after processing detections
                val afterMinMax = Core.minMaxLoc(overlayGray)
                AppLogger.debug("OverlayGray after mask creation: min=${afterMinMax.minVal}, max=${afterMinMax.maxVal}")

                // Count black pixels in final result
                val blackMask = Mat()
                Core.inRange(overlayGray, Scalar(0.0), Scalar(10.0), blackMask)
                val blackPixels = Core.countNonZero(blackMask)
                val totalPixels = overlayGray.rows() * overlayGray.cols()
                val blackRatio = blackPixels.toDouble() / totalPixels.toDouble()
                AppLogger.debug(
                    "Final result: Black pixels: $blackPixels / $totalPixels (${
                        String.format(
                            "%.4f",
                            blackRatio * 100
                        )
                    }%)"
                )
                blackMask.release()

            } else {
                AppLogger.debug("No detections after NMS to process for mask creation.")
            }
            AppLogger.debug("Completed mask creation for all detections")

        } catch (e: Exception) {
            AppLogger.debug("CRITICAL: Error processing model outputs: ${e.message}")
            e.printStackTrace()
            throw e // Rethrow to signal failure to the caller
        }
    }

    private fun extractPrototypeMasks(buffer: ByteBuffer): FloatArray {
        val prototypeTensor = interpreter!!.getOutputTensor(1)
        val prototypeTensorShape = prototypeTensor.shape() // e.g., [1, 160, 160, 32]
        AppLogger.debug("Prototype tensor shape: ${prototypeTensorShape.joinToString()}")

        // Correctly extract dimensions: [batch, height, width, channels]
        val prototypeHeight = prototypeTensorShape[1]       // 160
        val prototypeWidth = prototypeTensorShape[2]        // 160
        val numPrototypesChannels = prototypeTensorShape[3] // 32

        val prototypeMaskSize = numPrototypesChannels * prototypeHeight * prototypeWidth
        val expectedBufferSize = prototypeMaskSize * 4 // 4 bytes per float

        AppLogger.debug("Extracted prototype dims: $numPrototypesChannels channels, ${prototypeHeight}x${prototypeWidth}. Expected total floats: $prototypeMaskSize")

        if (buffer.capacity() < expectedBufferSize) {
            throw IllegalStateException("Prototype masks buffer too small: ${buffer.capacity()} bytes, expected at least $expectedBufferSize bytes. Model output shape mismatch for prototypes?")
        }

        val prototypeMasks = FloatArray(prototypeMaskSize)
        buffer.asFloatBuffer().get(prototypeMasks) // More efficient bulk read
        AppLogger.debug("Successfully extracted prototype masks: ${prototypeMasks.size} values")
        return prototypeMasks
    }

    /**
     * Parses detection results from TensorFlow Lite output buffer.
     * Aligned with JS reference's column-major data access logic.
     *
     * @param buffer The ByteBuffer containing the flattened detection output.
     * @param currentScoreThreshold The confidence threshold for filtering.
     * @param numProposals The total number of detection proposals (e.g., 8400).
     * @param featuresPerProposal The number of features per proposal (e.g., 116).
     * @param numClasses The total number of classes (e.g., 80).
     * @returns A list of parsed Detection objects.
     */
    private fun parseDetections(
        buffer: ByteBuffer,
        currentScoreThreshold: Float,
        numProposals: Int,
        featuresPerProposal: Int,
        numClasses: Int // Pass actual number of classes
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        buffer.rewind() // Ensure buffer is at the beginning

        val numBBoxCoords = 4
        val numMaskCoeffs = 32
        AppLogger.debug("Parsing detections: $numProposals proposals, $featuresPerProposal features each. Assuming $numClasses classes, $numBBoxCoords bbox, $numMaskCoeffs mask coeffs.")

        // Read the entire buffer into a FloatArray for easier column-major access simulation
        val floatArray = FloatArray(buffer.capacity() / 4)
        buffer.asFloatBuffer().get(floatArray)

        for (i in 0 until numProposals) {
            val carConfidenceIndex = i * featuresPerProposal + 4 + 2 // 4 bbox + classId for car (2)
            (1.0f / (1.0f + exp(-floatArray[carConfidenceIndex].toDouble()))).toFloat()
            //XXX apparently carConfidence is never used?
            //AppLogger.debug("RAW DETECTION - Proposal ${i+1}/${numProposals}, Car Confidence: ${String.format("%.4f", carConfidence)}")
            // Correct row-major access: floatArray[i * featuresPerProposal + feature_index]
            val cx = floatArray[i * featuresPerProposal + 0]
            val cy = floatArray[i * featuresPerProposal + 1]
            val w = floatArray[i * featuresPerProposal + 2]
            val h = floatArray[i * featuresPerProposal + 3]

            // Read class scores and apply sigmoid
            var maxProbability = 0f
            var maxClassId = -1
            for (classId in 0 until numClasses) {
                val rawScore = floatArray[i * featuresPerProposal + (numBBoxCoords + classId)]
                val probability = (1.0f / (1.0f + exp(-rawScore.toDouble()))).toFloat()
                if (probability > maxProbability) {
                    maxProbability = probability
                    maxClassId = classId
                }
            }

            // Read mask coefficients
            val maskCoefficients = FloatArray(numMaskCoeffs)
            for (j in 0 until numMaskCoeffs) {
                maskCoefficients[j] =
                    floatArray[i * featuresPerProposal + (numBBoxCoords + numClasses + j)]
            }

            // Filter by score and class
            //if (maxProbability > 0.001f) { // Log raw detections with very low threshold
            //AppLogger.debug("Raw detection (class: ${labels.getOrNull(maxClassId)}, confidence: $maxProbability)")
            //}
            if (maxProbability > currentScoreThreshold && vehicleClassIndices.contains(maxClassId)) {
                // Convert cx,cy,w,h to x_min,y_min,width,height (normalized 0-1)
                val x_min = (cx - w / 2f).coerceIn(0f, 1f)
                val y_min = (cy - h / 2f).coerceIn(0f, 1f)
                val width = w.coerceIn(0f, 1f - x_min) // Clamp width so x_max doesn't exceed 1
                val height = h.coerceIn(0f, 1f - y_min) // Clamp height so y_max doesn't exceed 1

                if (width > 0f && height > 0f) { // Basic sanity check for bbox
                    detections.add(
                        Detection(
                            x_min,
                            y_min,
                            width,
                            height,
                            maxProbability,
                            maxClassId,
                            maskCoefficients
                        )
                    )
                }
            }
        }
        AppLogger.debug("Total vehicle detections after score threshold: ${detections.size}")
        return detections
    }


    /**
     * Performs Non-Maximum Suppression (NMS) and applies per-class top-k filtering.
     * Aligned with JS reference.
     */
    private fun applyNMS(detections: List<Detection>, currentNmsThreshold: Float): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        // Sort by confidence (descending)
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val keepIndices = mutableListOf<Int>()
        val suppressed = BooleanArray(sortedDetections.size) { false }

        for (i in sortedDetections.indices) {
            if (suppressed[i]) {
                continue
            }

            keepIndices.add(i)

            val box1 = sortedDetections[i]
            for (j in i + 1 until sortedDetections.size) {
                if (suppressed[j]) {
                    continue
                }

                val box2 = sortedDetections[j]
                val iou = calculateIoU(box1, box2)

                if (iou > currentNmsThreshold) {
                    suppressed[j] = true
                }
            }
        }

        // Apply topAmountPerClass filtering (per class)
        val finalSelectedDetections = mutableListOf<Detection>()
        val classCounts = IntArray(labels.size) { 0 } // Use labels.size for numClasses

        for (idx in keepIndices) {
            val detection = sortedDetections[idx]
            if (classCounts[detection.classId] < topAmountPerClass) {
                finalSelectedDetections.add(detection)
                classCounts[detection.classId]++
            }
        }

        return finalSelectedDetections
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
        boxH: Float, //XXX Normalized (0-1) bounding box UNUSED?
        upscaleFactor: Float
    ): Mat {
        AppLogger.debug("=== ASSEMBLING MASK FROM PROTOTYPES ===")
        val protoTensorShape = interpreter!!.getOutputTensor(1).shape() // e.g. [1, 160, 160, 32]
        val prototypeHeight = protoTensorShape[1]       // 160
        val prototypeWidth = protoTensorShape[2]        // 160
        val numPrototypesChannels = protoTensorShape[3] // 32

        AppLogger.debug("Prototype tensor shape: [${protoTensorShape.joinToString(", ")}]")
        AppLogger.debug("Prototype dims: ${numPrototypesChannels} channels, ${prototypeHeight}x${prototypeWidth}")
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
        AppLogger.debug("Step 1: De-interleaving and combining ${numPrototypesChannels} prototype masks")

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


        // === COMBINATION LOGIC (Corrected) ===
        val combinedProtoMask = Mat.zeros(prototypeHeight, prototypeWidth, CvType.CV_32FC1)
        val weightedProtoMat = Mat() // Reusable Mat for the multiplication result
        var nonZeroCoeffs = 0

        for (i in 0 until numPrototypesChannels) {
            val coeff = maskCoefficients[i]
            if (coeff == 0f) continue
            nonZeroCoeffs++

            // Get the correctly structured prototype Mat
            val singleProtoMat = prototypeMats[i]

            // Step 1: Multiply the prototype by its coefficient using Core.multiply
            Core.multiply(singleProtoMat, Scalar(coeff.toDouble()), weightedProtoMat)

            // Step 2: Add the weighted result to the combined mask
            Core.add(combinedProtoMask, weightedProtoMat, combinedProtoMask)
        }

        // Clean up memory
        weightedProtoMat.release()
        prototypeMats.forEach { it.release() }

        AppLogger.debug("Used $nonZeroCoeffs non-zero coefficients out of $numPrototypesChannels")

        // === DEBUG VIEW 1: After combining all prototypes ===
        if (BuildConfig.DEBUG) {

            // This bitmap should now look like a coherent, meaningful mask shape.
            matToBitmapForDebug(combinedProtoMask)
            // during testing, debugBitmap1 at least contains useful information, doesn't seem too wrong.
        }
        // Check combined mask statistics
        val minMaxResult = Core.minMaxLoc(combinedProtoMask)
        AppLogger.debug("Combined mask before sigmoid: min=${minMaxResult.minVal}, max=${minMaxResult.maxVal}")

        // 2. Apply sigmoid to the combined 160x160 mask (values become 0-1)
        AppLogger.debug("Step 2: Applying sigmoid")
        applySigmoid(combinedProtoMask)
        if (BuildConfig.DEBUG) {
            matToBitmapForDebug(combinedProtoMask)
            // during testing, already debugBitmap2 is suspicious, as there is so little black pixels left ...
        }

        val minMaxAfterSigmoid = Core.minMaxLoc(combinedProtoMask)
        AppLogger.debug("Combined mask after sigmoid: min=${minMaxAfterSigmoid.minVal}, max=${minMaxAfterSigmoid.maxVal}")

        // 3. Threshold the mask (e.g., 0.4) to get binary values
        AppLogger.debug("Step 3: Applying threshold (0.4)")
        Imgproc.threshold(combinedProtoMask, combinedProtoMask, 0.4, 1.0, Imgproc.THRESH_BINARY)


        if (BuildConfig.DEBUG) {
            matToBitmapForDebug(combinedProtoMask)
        }

        val minMaxAfterThreshold = Core.minMaxLoc(combinedProtoMask)
        AppLogger.debug("Combined mask after threshold: min=${minMaxAfterThreshold.minVal}, max=${minMaxAfterThreshold.maxVal}")

        // 4. Resize the mask to the bounding box dimensions (scaled by upscaleFactor)
        val targetWidth = (boxW * inputWidth * upscaleFactor).toInt().coerceAtLeast(1)
        val targetHeight = (boxH * inputHeight * upscaleFactor).toInt().coerceAtLeast(1)
        AppLogger.debug("Step 4: Resizing mask from ${prototypeWidth}x${prototypeHeight} to ${targetWidth}x${targetHeight}")
        if ((targetWidth == 1) and (targetHeight == 1)) {
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
        if (BuildConfig.DEBUG) {
            matToBitmapForDebug(combinedProtoMask)
        }

        // 5. Convert to 8UC1 (grayscale) for overlay
        AppLogger.debug("Step 5: Converting to CV_8UC1")
        resizedMask.convertTo(resizedMask, CvType.CV_8UC1, 255.0)
        if (BuildConfig.DEBUG) {
            matToBitmapForDebug(resizedMask)
        }

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

    /**
     * Handle overflow boxes based on maxSize (normalized 0-1 range).
     * Box format: [x_min, y_min, width, height]
     * @param box box in [x_min, y_min, width, height] format
     * @param maxSize Max value for normalized coordinates, typically 1.0f.
     * @returns non overflow boxes
     */
    private fun overflowBoxes(box: FloatArray, maxSize: Float): FloatArray {
        val x_min = box[0].coerceIn(0f, maxSize)
        val y_min = box[1].coerceIn(0f, maxSize)
        val width = (box[2] + x_min).coerceAtMost(maxSize) - x_min
        val height = (box[3] + y_min).coerceAtMost(maxSize) - y_min
        return floatArrayOf(x_min, y_min, width, height)
    }

    /**
     * Shifts the mask content downwards by a factor of its height.
     * The top part created by the shift is filled with white.
     * This function is re-added for compatibility with CarDetectionDebugTest.kt,
     * but its logic is not part of the core segmentation flow in inferYolo.
     * @param originalMask The CV_8UC1 mask to shift.
     * @param downshiftFactor Percentage of height to shift down (0.0 to 1.0).
     * @return A new Mat with the shifted content. The originalMat is NOT released by this function.
     */
    private fun shiftDown(originalMask: Mat, downshiftFactor: Float): Mat {
        if (downshiftFactor <= 0.0f) return originalMask.clone() // No shift or invalid factor

        val height = originalMask.rows()
        val width = originalMask.cols()
        val shiftPixels = (height * downshiftFactor.coerceIn(0f, 1f)).toInt()

        if (shiftPixels == 0) return originalMask.clone()
        if (shiftPixels >= height) { // If shift is entire height or more, return all white
            val allWhite = Mat(height, width, originalMask.type(), Scalar(255.0))
            return allWhite
        }

        val shiftedMask =
            Mat(height, width, originalMask.type(), Scalar(255.0)) // Initialize with white

        // Define region of interest (ROI) for the part of the original mask to keep
        val sourceRoi = Rect(0, 0, width, height - shiftPixels)
        // Define where this ROI will be placed in the new mask
        val targetRoi = Rect(0, shiftPixels, width, height - shiftPixels)

        originalMask.submat(sourceRoi).copyTo(shiftedMask.submat(targetRoi))

        return shiftedMask
    }

    fun close() {
        interpreter?.close()
        isInitialized = false
        AppLogger.debug("TFLite YoloInference closed.")
    }
}

/**
 * Data class to hold detection results.
 * Coordinates are normalized (0-1 range) and represent [x_min, y_min, width, height].
 */
data class Detection(
    val x: Float, // x_min
    val y: Float, // y_min
    val width: Float,
    val height: Float,
    val confidence: Float,
    val classId: Int,
    val maskCoefficients: FloatArray // Typically 32 coefficients for YOLOv8-seg
) {
    //XXX UNUSED Convenience getters for x1, y1, x2, y2 (normalized 0-1 range)
    val x1: Float get() = x
    val y1: Float get() = y
    val x2: Float get() = x + width
    val y2: Float get() = y + height

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Detection

        if (x != other.x) return false
        if (y != other.y) return false
        if (width != other.width) return false
        if (height != other.height) return false
        if (confidence != other.confidence) return false
        if (classId != other.classId) return false
        if (!maskCoefficients.contentEquals(other.maskCoefficients)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = x.hashCode()
        result = 31 * result + y.hashCode()
        result = 31 * result + width.hashCode()
        result = 31 * result + height.hashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + classId
        result = 31 * result + maskCoefficients.contentHashCode()
        return result
    }
}
