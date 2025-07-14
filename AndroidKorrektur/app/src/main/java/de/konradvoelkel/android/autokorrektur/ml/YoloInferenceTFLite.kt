package de.konradvoelkel.android.autokorrektur.ml

import android.content.Context
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

    // Configuration parameters
    private val scoreThreshold = 0.2f // Confidence threshold after sigmoid for filtering detections
    private val nmsThreshold = 0.4f   // IoU threshold for Non-Maximum Suppression

    // Vehicle class indices (car, motorcycle, bus, truck) - COCO indices
    private val vehicleClassIndices = intArrayOf(2, 3, 5, 7) // car, motorcycle, bus, truck


    @Throws(IOException::class)
    fun initialize(modelName: String = "yolo11s", useFP16: Boolean = true) {
        if (isInitialized) {
            println("[DEBUG_LOG] TFLite YoloInference already initialized")
            return
        }

        val modelFile = if (useFP16) {
            "model/${modelName}-seg_saved_model/${modelName}-seg_float16.tflite"
        } else {
            "model/${modelName}-seg_saved_model/${modelName}-seg_float32.tflite"
        }

        println("[DEBUG_LOG] TFLite YoloInference.initialize() - Loading model: $modelFile")

        try {
            val assetFileDescriptor = context.assets.openFd(modelFile)
            val inputStream = assetFileDescriptor.createInputStream()
            val modelBytes = inputStream.readBytes()
            inputStream.close()
            assetFileDescriptor.close()

            println("[DEBUG_LOG] TFLite model loaded: ${modelBytes.size} bytes")

            val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size)
            modelBuffer.order(ByteOrder.nativeOrder())
            modelBuffer.put(modelBytes)
            modelBuffer.rewind()

            val options = Interpreter.Options()
            options.setNumThreads(Runtime.getRuntime().availableProcessors()) // Use available processors
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

            println("[DEBUG_LOG] Model input shape: [${inputShape.joinToString(", ")}]")
            println("[DEBUG_LOG] Input dimensions: ${inputWidth}x${inputHeight}x${inputChannels}")

            isInitialized = true
            println("[DEBUG_LOG] TFLite YoloInference initialized successfully")

        } catch (e: Exception) {
            println("[DEBUG_LOG] ========== TFLITE INITIALIZATION FAILED ==========")
            println("[DEBUG_LOG] Exception type: ${e.javaClass.simpleName}")
            println("[DEBUG_LOG] Exception message: ${e.message}")
            e.printStackTrace()
            println("[DEBUG_LOG] ================================================")
            throw IOException("Failed to initialize TFLite YOLO model: ${e.message}", e)
        }
    }

    @Throws(IOException::class)
    fun inferYolo(
        transformedMat: Mat, // Expected: CV_32FC3, RGB, Normalized, this.inputWidth x this.inputHeight
        @Suppress("UNUSED_PARAMETER") xRatio: Float, // Letterbox ratio, potentially useful for drawing results on original image
        @Suppress("UNUSED_PARAMETER") yRatio: Float, // Letterbox ratio
        upscaleFactor: Float = 1.2f,
        downshiftFactor: Float = 0.0f
    ): Mat {
        if (!isInitialized || interpreter == null) {
            println("[DEBUG_LOG] Interpreter not initialized. Initializing now...")
            initialize() // This might throw IOException
        }

        // Output mask will be the same size as model input, representing the letterboxed image content
        val overlayGray = Mat.ones(this.inputHeight, this.inputWidth, CvType.CV_8UC1)
        overlayGray.setTo(Scalar(255.0)) // White background

        try {
            // Prepare input buffer
            val inputBufferCapacity = 4 * this.inputWidth * this.inputHeight * this.inputChannels // 4 bytes per float
            val inputBuffer = ByteBuffer.allocateDirect(inputBufferCapacity)
            inputBuffer.order(ByteOrder.nativeOrder())
            matToByteBuffer(transformedMat, inputBuffer) // Populate inputBuffer from transformedMat

            // Prepare output buffers map
            val outputMap = mutableMapOf<Int, Any>()
            val numOutputs = interpreter!!.outputTensorCount
            println("[DEBUG_LOG] Model has $numOutputs outputs")

            for (i in 0 until numOutputs) {
                val outputTensor = interpreter!!.getOutputTensor(i)
                val outputShape = outputTensor.shape()
                println("[DEBUG_LOG] Output $i shape: [${outputShape.joinToString(", ")}]")
                // Calculate size needed for the buffer based on tensor shape (product of dimensions) * bytes_per_element
                val outputSize = outputShape.fold(1L) { acc, dim -> acc * dim }.toInt() // Use Long for intermediate to avoid overflow
                val outputBuffer = ByteBuffer.allocateDirect(4 * outputSize) // Assuming Float32 output
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

            // Apply downshift if specified
            if (downshiftFactor > 0.0f) {
                val shiftedMask = shiftDown(overlayGray, downshiftFactor)
                overlayGray.release() // Release original overlayGray as we're returning the shifted one
                return shiftedMask
            }
            return overlayGray

        } catch (e: Exception) {
            println("[DEBUG_LOG] TFLite inference failed: ${e.message}")
            e.printStackTrace()
            // Even in error, if downshift is specified, apply to whatever state overlayGray is in
            if (downshiftFactor > 0.0f) {
                val shiftedMask = shiftDown(overlayGray, downshiftFactor)
                overlayGray.release()
                return shiftedMask
            }
            return overlayGray // Or rethrow: throw RuntimeException("Inference failed", e)
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
            println("[DEBUG_LOG] WARNING: matToByteBuffer input Mat dimensions (${mat.width()}x${mat.height()}) " +
                    "do not match model input dimensions (${this.inputWidth}x${this.inputHeight}). Resizing.")
            val tempResizedMat = Mat()
            Imgproc.resize(mat, tempResizedMat, Size(this.inputWidth.toDouble(), this.inputHeight.toDouble()))
            val data = FloatArray(this.inputWidth * this.inputHeight * this.inputChannels)
            tempResizedMat.get(0, 0, data)
            for (value in data) {
                buffer.putFloat(value)
            }
            tempResizedMat.release()
            return
        }
        if (mat.type() != CvType.CV_32FC3) {
            println("[DEBUG_LOG] CRITICAL WARNING: matToByteBuffer input Mat type is not CV_32FC3. Type: ${CvType.typeToString(mat.type())}")
            val floatMat = Mat()
            val conversionFactor = if(CvType.depth(mat.type()) == CvType.CV_8U) 1.0/255.0 else 1.0
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
            println("[DEBUG_LOG] CRITICAL WARNING: matToByteBuffer input Mat channels (${mat.channels()}) " +
                    "do not match model input channels (${this.inputChannels}). Expected RGB.")
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
            println("[DEBUG_LOG] Processing model outputs for mask creation")
            val detectionsBuffer = outputs[0] as? ByteBuffer
                ?: throw IllegalStateException("Model output 0 (detections) not found or not a ByteBuffer.")
            val prototypeMasksBuffer = outputs[1] as? ByteBuffer
                ?: throw IllegalStateException("Model output 1 (prototypes) not found or not a ByteBuffer.")

            detectionsBuffer.rewind()
            prototypeMasksBuffer.rewind()

            println("[DEBUG_LOG] Detections buffer capacity: ${detectionsBuffer.capacity()} bytes")
            println("[DEBUG_LOG] Prototype masks buffer capacity: ${prototypeMasksBuffer.capacity()} bytes")

            // Get detection tensor shape to correctly determine numProposals and featuresPerProposal
            val detectionTensor = interpreter!!.getOutputTensor(0)
            val detectionTensorShape = detectionTensor.shape() // e.g., [1, 8400, 116]
            val featuresPerProposal = detectionTensorShape[1]
            val numProposals = detectionTensorShape[2]
            println("[DEBUG_LOG] Detection tensor shape: ${detectionTensorShape.joinToString()}, NumProposals: $numProposals, FeaturesPerProposal: $featuresPerProposal")


            val detections = parseDetections(detectionsBuffer, this.scoreThreshold, numProposals, featuresPerProposal)
            val filteredDetections = applyNMS(detections, this.nmsThreshold)
            println("[DEBUG_LOG] Filtered detections after NMS: ${filteredDetections.size}")

            if (filteredDetections.isNotEmpty()) {
                val prototypeMasks = extractPrototypeMasks(prototypeMasksBuffer)
                println("[DEBUG_LOG] Creating masks for ${filteredDetections.size} vehicle detections")
                for (detection in filteredDetections) {
                    println("[DEBUG_LOG] Processing detection: class=${if (detection.classId < labels.size && detection.classId >=0) labels[detection.classId] else "Unknown"} (${detection.classId}), confidence=${detection.confidence}")
                    createDetectionMask(
                        detection,
                        overlayGray,
                        upscaleFactor,
                        prototypeMasks // Pass the flat array
                    )
                }
            } else {
                println("[DEBUG_LOG] No detections after NMS to process for mask creation.")
            }
            println("[DEBUG_LOG] Completed mask creation for all detections")

        } catch (e: Exception) {
            println("[DEBUG_LOG] CRITICAL: Error processing model outputs: ${e.message}")
            e.printStackTrace()
            throw e // Rethrow to signal failure to the caller
        }
    }

    private fun extractPrototypeMasks(buffer: ByteBuffer): FloatArray {
        val prototypeTensor = interpreter!!.getOutputTensor(1)
        val prototypeTensorShape = prototypeTensor.shape() // e.g., [1, 32, 160, 160]
        println("[DEBUG_LOG] Prototype tensor shape: ${prototypeTensorShape.joinToString()}")

        val numPrototypes = prototypeTensorShape[3]
        val prototypeHeight = prototypeTensorShape[1]
        val prototypeWidth = prototypeTensorShape[2]

        val prototypeMaskSize = numPrototypes * prototypeHeight * prototypeWidth
        val expectedBufferSize = prototypeMaskSize * 4 // 4 bytes per float

        println("[DEBUG_LOG] Extracted prototype dims: $numPrototypes protos, ${prototypeHeight}x${prototypeWidth}. Expected total floats: $prototypeMaskSize")

        if (buffer.capacity() < expectedBufferSize) {
            throw IllegalStateException("Prototype masks buffer too small: ${buffer.capacity()} bytes, expected at least $expectedBufferSize bytes. Model output shape mismatch for prototypes?")
        }

        val prototypeMasks = FloatArray(prototypeMaskSize)
        buffer.asFloatBuffer().get(prototypeMasks) // More efficient bulk read
        println("[DEBUG_LOG] Successfully extracted prototype masks: ${prototypeMasks.size} values")
        return prototypeMasks
    }

    /**
     * Parses detection results from TensorFlow Lite output buffer.
     * CRITICAL ASSUMPTION for feature order: The code assumes features_per_proposal are ordered as
     * [bbox (4 floats: cx, cy, w, h), class_scores (num_classes floats), mask_coefficients (num_mask_coeffs floats)].
     * VERIFY THIS ORDER for your specific TFLite model.
     */
    private fun parseDetections(
        buffer: ByteBuffer,
        currentScoreThreshold: Float,
        numProposals: Int, // e.g., 8400
        featuresPerProposal: Int // e.g., 116 (4 bbox + 80 classes + 32 mask_coeffs)
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        buffer.rewind() // Ensure buffer is at the beginning

        val numClasses = 80 // Standard COCO classes
        val numBBoxCoords = 4
        // Calculate numMaskCoeffs based on featuresPerProposal
        val numMaskCoeffs = featuresPerProposal - numBBoxCoords - numClasses
        if (numMaskCoeffs <= 0) { // Should be 32 for typical YOLOv8-seg
            throw IllegalArgumentException("Calculated numMaskCoeffs ($numMaskCoeffs) is invalid or non-positive. " +
                    "Check featuresPerProposal ($featuresPerProposal), numClasses ($numClasses), and numBBoxCoords ($numBBoxCoords). " +
                    "Ensure the TFLite model output for detections is as expected.")
        }
        println("[DEBUG_LOG] Parsing detections: $numProposals proposals, $featuresPerProposal features each. Assuming $numClasses classes, $numBBoxCoords bbox, $numMaskCoeffs mask coeffs.")
        println("[DEBUG_LOG] Feature order current assumption: BBOX (4), THEN SCORES ($numClasses), THEN MASK_COEFFS ($numMaskCoeffs)")


        val floatBuffer = buffer.asFloatBuffer() // Use FloatBuffer for easier reading

        for (i in 0 until numProposals) {
            // --- READ BBOX (cx, cy, w, h) ---
            val cx = floatBuffer.get()
            val cy = floatBuffer.get()
            val w = floatBuffer.get()
            val h = floatBuffer.get()

            // --- READ CLASS SCORES & APPLY SIGMOID ---
            var maxProbability = 0f
            var maxClassId = -1
            for (classId in 0 until numClasses) {
                val rawScore = floatBuffer.get()
                val probability = (1.0f / (1.0f + exp(-rawScore.toDouble()))).toFloat() // Apply sigmoid
                if (probability > maxProbability) {
                    maxProbability = probability
                    maxClassId = classId
                }
            }

            // --- READ MASK COEFFICIENTS ---
            val maskCoefficients = FloatArray(numMaskCoeffs)
            for (j in 0 until numMaskCoeffs) {
                maskCoefficients[j] = floatBuffer.get()
            }

            // Filter by score and class
            if (maxProbability > currentScoreThreshold && vehicleClassIndices.contains(maxClassId)) {
                // Convert cx,cy,w,h to x1,y1,x2,y2 (normalized 0-1)
                val x1 = (cx - w / 2f).coerceIn(0f, 1f)
                val y1 = (cy - h / 2f).coerceIn(0f, 1f)
                val x2 = (cx + w / 2f).coerceIn(0f, 1f)
                val y2 = (cy + h / 2f).coerceIn(0f, 1f)

                if (x1 < x2 && y1 < y2) { // Basic sanity check for bbox
                    detections.add(
                        Detection(x1, y1, x2, y2, maxProbability, maxClassId, maskCoefficients)
                    )
                } else {
                    // println("[DEBUG_LOG] Invalid bbox after conversion or from model: ($x1,$y1,$x2,$y2) for class $maxClassId, prob $maxProbability")
                }
            }
        }
        println("[DEBUG_LOG] Total vehicle detections after score threshold: ${detections.size}")
        if (detections.isEmpty() && numProposals > 0 && floatBuffer.limit() > 0 && currentScoreThreshold < 0.99f) { // Added threshold check to avoid spamming if threshold is very high
            val tempScores = FloatArray(numProposals * numClasses)
            buffer.rewind()
            val tempFloatBuffer = buffer.asFloatBuffer()
            var maxRawScoreEncountered = -Float.MAX_VALUE
            var minRawScoreEncountered = Float.MAX_VALUE
            for(k in 0 until numProposals) {
                tempFloatBuffer.position(tempFloatBuffer.position() + numBBoxCoords) // Skip bbox
                for (l in 0 until numClasses) {
                    val score = tempFloatBuffer.get()
                    if (score > maxRawScoreEncountered) maxRawScoreEncountered = score
                    if (score < minRawScoreEncountered) minRawScoreEncountered = score
                }
                tempFloatBuffer.position(tempFloatBuffer.position() + numMaskCoeffs) // Skip mask coeffs
            }
            println("[DEBUG_LOG] WARNING: No detections met threshold $currentScoreThreshold. Max raw score found: $maxRawScoreEncountered (prob ~${(1.0f / (1.0f + exp(-maxRawScoreEncountered.toDouble()))).toFloat()}), Min raw score: $minRawScoreEncountered. Check model output, scoreThreshold, or feature parsing order if detections are expected.")
        }
        return detections
    }


    private fun applyNMS(detections: List<Detection>, currentNmsThreshold: Float): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        // Sort by confidence (descending)
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val keep = mutableListOf<Detection>()

        for (detection in sortedDetections) {
            var shouldKeep = true
            for (keptDetection in keep) {
                // Check if classes are the same, NMS is usually per-class, but here we group all vehicles
                // if (detection.classId == keptDetection.classId) {
                if (calculateIoU(detection, keptDetection) > currentNmsThreshold) {
                    shouldKeep = false
                    break
                }
                // }
            }
            if (shouldKeep) {
                keep.add(detection)
            }
        }
        return keep
    }

    private fun calculateIoU(det1: Detection, det2: Detection): Float {
        // Normalized coordinates (0-1 range)
        val xA = max(det1.x1, det2.x1)
        val yA = max(det1.y1, det2.y1)
        val xB = min(det1.x2, det2.x2)
        val yB = min(det1.y2, det2.y2)

        // Intersection area
        val intersectionArea = max(0f, xB - xA) * max(0f, yB - yA)
        if (intersectionArea <= 0f) return 0f

        // Individual areas
        val area1 = (det1.x2 - det1.x1) * (det1.y2 - det1.y1)
        val area2 = (det2.x2 - det2.x1) * (det2.y2 - det2.y1)

        // Union area
        val unionArea = area1 + area2 - intersectionArea

        return if (unionArea > 0.000001f) intersectionArea / unionArea else 0f // Avoid division by zero or tiny union
    }

    private fun createDetectionMask(
        detection: Detection,
        overlayGray: Mat, // Full size overlay (e.g. 640x640, CV_8UC1) to draw upon
        upscaleFactor: Float,
        prototypeMasksData: FloatArray // Flat array of all prototype masks
    ) {
        // println("[DEBUG_LOG] Creating detection mask for class=${if (detection.classId < labels.size) labels[detection.classId] else "Unknown"}, conf=${detection.confidence}")

        // Determine expected number of mask coefficients from model output tensor 0 shape
        val detectionTensorShape = interpreter!!.getOutputTensor(0).shape()
        val featuresPerProposal = detectionTensorShape[1]
        val numClasses = 80 // Assuming 80 COCO classes
        val numBBoxCoords = 4
        val numMaskCoeffsExpected = featuresPerProposal - numBBoxCoords - numClasses

        if (detection.maskCoefficients.size != numMaskCoeffsExpected) {
            println("[DEBUG_LOG] ERROR: Mask coefficients size mismatch. Expected $numMaskCoeffsExpected, Got ${detection.maskCoefficients.size}. Skipping mask for this detection.")
            return // Skip this detection if mask coefficients are wrong
        }

        // Adjust bounding box by upscaleFactor (coordinates are 0-1 normalized)
        var adjX1 = detection.x1
        var adjY1 = detection.y1
        var adjX2 = detection.x2
        var adjY2 = detection.y2

        if (upscaleFactor != 1.0f && upscaleFactor > 0f) {
            val boxWidth = detection.x2 - detection.x1
            val boxHeight = detection.y2 - detection.y1
            val newWidth = boxWidth * upscaleFactor
            val newHeight = boxHeight * upscaleFactor

            // Center the upscaled box
            adjX1 = detection.x1 - (newWidth - boxWidth) / 2f
            adjY1 = detection.y1 - (newHeight - boxHeight) / 2f
            adjX2 = adjX1 + newWidth
            adjY2 = adjY1 + newHeight
            // No coerceIn here yet, let assembleMaskFromPrototypes handle clamping of its inputs
        }

        try {
            // This returns a CV_32FC1 mask of size this.inputWidth x this.inputHeight (e.g., 640x640)
            // with the specific object's mask active (values 0-1) and rest 0.
            val segmentationMaskSegment = assembleMaskFromPrototypes(
                detection.maskCoefficients,
                prototypeMasksData,
                adjX1, adjY1, adjX2, adjY2 // Use adjusted (potentially upscaled) normalized coords
            )

            // Apply this segment to the main overlayGray
            if (!segmentationMaskSegment.empty()) {
                applySegmentationMask(segmentationMaskSegment, overlayGray)
                segmentationMaskSegment.release() // Release the intermediate mask segment
                // println("[DEBUG_LOG] Successfully applied segmentation mask for detection.")
            } else {
                println("[DEBUG_LOG] assembleMaskFromPrototypes returned empty mask. Skipping application.")
            }

        } catch (e: Exception) {
            println("[DEBUG_LOG] CRITICAL ERROR in mask assembly/application for one detection: ${e.message}")
            e.printStackTrace()
            // Do not rethrow here, to allow other detections to be processed if one fails
        }
    }

    /**
     * Assembles a segmentation mask for a single detection from prototype masks and coefficients.
     * @param normX1, normY1, normX2, normY2: Normalized (0-1) bounding box of the detection,
     *                                          potentially already upscaled. These define the
     *                                          region of interest on the model's input scale.
     * @return A Mat (CV_32FC1) of size this.inputWidth x this.inputHeight, containing the
     *         object's mask (values 0-1), with areas outside the object being 0.
     */
    private fun assembleMaskFromPrototypes(
        maskCoefficients: FloatArray,
        prototypeMasksData: FloatArray, // Flat array: num_prototypes * proto_height * proto_width
        normX1: Float, normY1: Float, normX2: Float, normY2: Float
    ): Mat {
        val protoTensorShape = interpreter!!.getOutputTensor(1).shape() // e.g. [1, 32, 160, 160]
        val numPrototypes = protoTensorShape[1]
        val prototypeHeight = protoTensorShape[2]
        val prototypeWidth = protoTensorShape[3]

        if (maskCoefficients.size != numPrototypes) {
            println("[DEBUG_LOG] assembleMask: Mask coeffs size mismatch. Expected $numPrototypes, got ${maskCoefficients.size}")
            return Mat() // Return empty Mat on error
        }
        if (prototypeMasksData.size != numPrototypes * prototypeHeight * prototypeWidth) {
            println("[DEBUG_LOG] assembleMask: Prototype data size mismatch. Expected ${numPrototypes * prototypeHeight * prototypeWidth}, got ${prototypeMasksData.size}")
            return Mat()
        }

        // 1. Combine prototype masks using coefficients into a single 160x160 mask
        val combinedProtoMask = Mat.zeros(prototypeHeight, prototypeWidth, CvType.CV_32FC1)
        val singleProtoMat = Mat(prototypeHeight, prototypeWidth, CvType.CV_32FC1) // Reusable
        val weightedProtoMat = Mat() // Reusable for multiplication result

        for (i in 0 until numPrototypes) {
            val coeff = maskCoefficients[i]
            if (coeff == 0f) continue // No contribution if coefficient is zero

            val offset = i * prototypeHeight * prototypeWidth
            // Basic bounds check for safety, though sizes are pre-validated
            if (offset + prototypeHeight * prototypeWidth > prototypeMasksData.size) {
                println("[DEBUG_LOG] ERROR: Reading prototype $i beyond prototypeMasksData bounds.")
                continue
            }
            val protoDataForOneMask = prototypeMasksData.copyOfRange(offset, offset + prototypeHeight * prototypeWidth)
            singleProtoMat.put(0, 0, protoDataForOneMask)

            Core.multiply(singleProtoMat, Scalar(coeff.toDouble()), weightedProtoMat)
            Core.add(combinedProtoMask, weightedProtoMat, combinedProtoMask)
        }
        singleProtoMat.release()
        weightedProtoMat.release()

        // 2. Apply sigmoid to the combined 160x160 mask (values become 0-1)
        applySigmoid(combinedProtoMask)

        // 3. Crop and resize this combined mask to the target bounding box on the full model input scale
        // Clamp normalized coordinates to the valid [0, 1] range before using them for calculations
        val clampedNormX1 = normX1.coerceIn(0f, 1f)
        val clampedNormY1 = normY1.coerceIn(0f, 1f)
        val clampedNormX2 = normX2.coerceIn(0f, 1f)
        val clampedNormY2 = normY2.coerceIn(0f, 1f)

        // Ensure valid box (x1 < x2, y1 < y2) after clamping
        if (clampedNormX1 >= clampedNormX2 || clampedNormY1 >= clampedNormY2) {
            println("[DEBUG_LOG] assembleMask: Invalid/collapsed normalized bbox after clamping: ($clampedNormX1,$clampedNormY1) to ($clampedNormX2,$clampedNormY2). Returning empty mask.")
            combinedProtoMask.release()
            return Mat() // Return empty Mat
        }

        // Map the clamped normalized box (relative to model input e.g. 640x640)
        // to pixel coordinates on the prototype mask (e.g. 160x160)
        val protoCropX = (clampedNormX1 * prototypeWidth).toInt()
        val protoCropY = (clampedNormY1 * prototypeHeight).toInt()
        val protoCropW = ((clampedNormX2 - clampedNormX1) * prototypeWidth).toInt().coerceAtLeast(1)
        val protoCropH = ((clampedNormY2 - clampedNormY1) * prototypeHeight).toInt().coerceAtLeast(1)

        // Ensure the crop rectangle is within the prototype mask dimensions
        val validProtoCropX = protoCropX.coerceIn(0, prototypeWidth -1)
        val validProtoCropY = protoCropY.coerceIn(0, prototypeHeight -1)
        // Adjust width/height to not exceed prototype boundaries from the starting point
        val validProtoCropW = (validProtoCropX + protoCropW).coerceAtMost(prototypeWidth) - validProtoCropX
        val validProtoCropH = (validProtoCropY + protoCropH).coerceAtMost(prototypeHeight) - validProtoCropY


        if (validProtoCropW <= 0 || validProtoCropH <= 0) {
            println("[DEBUG_LOG] assembleMask: Prototype crop dimensions are zero or negative. Skipping. W: $validProtoCropW, H: $validProtoCropH")
            combinedProtoMask.release()
            return Mat()
        }
        val cropRectOnProto = Rect(validProtoCropX, validProtoCropY, validProtoCropW, validProtoCropH)
        val croppedSubMaskFromProto = Mat(combinedProtoMask, cropRectOnProto)


        // Create a full-size mask (e.g., 640x640) initialized to zeros
        val fullSizeSegmentMask = Mat.zeros(this.inputHeight, this.inputWidth, CvType.CV_32FC1)
// Define the target ROI on the fullSizeSegmentMask where the croppedSubMaskFromProto will be placed and resized.
        // This ROI corresponds to the (potentially upscaled) object's bounding box on the full model input scale.
        val targetRoiX = (clampedNormX1 * this.inputWidth).toInt()
        val targetRoiY = (clampedNormY1 * this.inputHeight).toInt()
        val targetRoiW = ((clampedNormX2 - clampedNormX1) * this.inputWidth).toInt().coerceAtLeast(1)
        val targetRoiH = ((clampedNormY2 - clampedNormY1) * this.inputHeight).toInt().coerceAtLeast(1)

        // Ensure target ROI is within the bounds of fullSizeSegmentMask
        val validTargetRoiX = targetRoiX.coerceIn(0, this.inputWidth - 1)
        val validTargetRoiY = targetRoiY.coerceIn(0, this.inputHeight - 1)
        val validTargetRoiW = (validTargetRoiX + targetRoiW).coerceAtMost(this.inputWidth) - validTargetRoiX
        val validTargetRoiH = (validTargetRoiY + targetRoiH).coerceAtMost(this.inputHeight) - validTargetRoiY

        if (validTargetRoiW > 0 && validTargetRoiH > 0) {
            val targetRoi = Mat(fullSizeSegmentMask, Rect(validTargetRoiX, validTargetRoiY, validTargetRoiW, validTargetRoiH))
            Imgproc.resize(croppedSubMaskFromProto, targetRoi, targetRoi.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
            targetRoi.release()
        } else {
            println("[DEBUG_LOG] Warning: Target ROI for mask segment is invalid or out of bounds after coercion. Skipping paste for this segment.")
            println("[DEBUG_LOG] Orig Target ROI: x=$targetRoiX, y=$targetRoiY, w=$targetRoiW, h=$targetRoiH.")
            println("[DEBUG_LOG] Valid Target ROI: x=$validTargetRoiX, y=$validTargetRoiY, w=$validTargetRoiW, h=$validTargetRoiH.")
            println("[DEBUG_LOG] Full mask: ${fullSizeSegmentMask.cols()}x${fullSizeSegmentMask.rows()}")
            println("[DEBUG_LOG] From Clamped Norm Coords: x1=$clampedNormX1, y1=$clampedNormY1, x2=$clampedNormX2, y2=$clampedNormY2")
        }

        // croppedSubMaskFromProto is a submatrix (header of combinedProtoMask),
        // no need to release explicitly unless it was copied using .clone() or .copyTo().
        combinedProtoMask.release()

        return fullSizeSegmentMask // This is a CV_32FC1 mask of size this.inputWidth x this.inputHeight
    }

    /**
     * Applies the generated segmentation mask (CV_32FC1, values 0-1) to the overlay (CV_8UC1).
     * Where the segmentationMask is > 0.5 (active), overlay is set to black (0).
     * Otherwise, overlay remains unchanged (or white if initially set).
     */
    private fun applySegmentationMask(
        segmentationMaskSegment: Mat, // CV_32FC1, values 0-1, size of model input (e.g. 640x640)
        overlayGray: Mat              // CV_8UC1, size of model input (e.g. 640x640)
    ) {
        if (segmentationMaskSegment.empty() || overlayGray.empty()) {
            println("[DEBUG_LOG] applySegmentationMask: Input mask or overlay is empty. Skipping.")
            return
        }
        if (segmentationMaskSegment.size() != overlayGray.size() || segmentationMaskSegment.type() != CvType.CV_32FC1 || overlayGray.type() != CvType.CV_8UC1) {
            println("[DEBUG_LOG] applySegmentationMask: Mismatched dimensions or types. Seg: ${segmentationMaskSegment.size()} Type: ${CvType.typeToString(segmentationMaskSegment.type())}, Overlay: ${overlayGray.size()} Type: ${CvType.typeToString(overlayGray.type())}")
            return
        }

        val thresholdValue = 0.5 // Values in segmentationMaskSegment > this will be part of the mask
        val black = Scalar(0.0)    // Value to set for masked pixels in overlayGray

        // Create a binary mask (CV_8UC1) from the float segmentation mask
        val binaryMask = Mat()
        Imgproc.threshold(segmentationMaskSegment, binaryMask, thresholdValue, 255.0, Imgproc.THRESH_BINARY)
        binaryMask.convertTo(binaryMask, CvType.CV_8U) // Ensure it's CV_8UC1

        // Set pixels in overlayGray to black where binaryMask is non-zero
        overlayGray.setTo(black, binaryMask)

        binaryMask.release()
    }

    /**
     * Applies sigmoid function element-wise to a CV_32FC1 Mat.
     */
    private fun applySigmoid(mat: Mat) { // Input/Output Mat is CV_32FC1
        if (mat.type() != CvType.CV_32FC1) {
            println("[DEBUG_LOG] applySigmoid: Mat type is not CV_32FC1. Type: ${CvType.typeToString(mat.type())}")
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
     * Shifts the mask content downwards by a factor of its height.
     * The top part created by the shift is filled with white.
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

        val shiftedMask = Mat(height, width, originalMask.type(), Scalar(255.0)) // Initialize with white

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
        println("[DEBUG_LOG] TFLite YoloInference closed.")
    }
}

/**
 * Data class to hold detection results.
 * Coordinates are normalized (0-1 range).
 */
data class Detection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float,
    val classId: Int,
    val maskCoefficients: FloatArray // Typically 32 coefficients for YOLOv8-seg
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Detection

        if (x1 != other.x1) return false
        if (y1 != other.y1) return false
        if (x2 != other.x2) return false
        if (y2 != other.y2) return false
        if (confidence != other.confidence) return false
        if (classId != other.classId) return false
        if (!maskCoefficients.contentEquals(other.maskCoefficients)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = x1.hashCode()
        result = 31 * result + y1.hashCode()
        result = 31 * result + x2.hashCode()
        result = 31 * result + y2.hashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + classId
        result = 31 * result + maskCoefficients.contentHashCode()
        return result
    }
}
