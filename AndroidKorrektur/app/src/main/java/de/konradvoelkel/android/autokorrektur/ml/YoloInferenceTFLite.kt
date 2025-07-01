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
import kotlin.math.max
import kotlin.math.min

/**
 * TensorFlow Lite implementation of YOLO model inference for car segmentation.
 * Uses TensorFlow Lite Interpreter directly to avoid OpenCV DNN STRIDED_SLICE issues.
 */
class YoloInferenceTFLite(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var isInitialized = false

    // Model input/output dimensions
    private var inputWidth = 640
    private var inputHeight = 640
    private var inputChannels = 3

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
    private val scoreThreshold = 0.2f
    private val nmsThreshold = 0.4f
    private val confThreshold = 0.4f

    // Vehicle class indices (car, motorcycle, truck)
    private val vehicleClassIndices = intArrayOf(2, 3, 7)

    /**
     * Initializes the TFLite YOLO model using TensorFlow Lite Interpreter.
     *
     * @param modelName The name of the YOLO model to use (e.g., "yolo11s")
     * @param useFP16 Whether to use the float16 model (smaller, potentially faster)
     * @throws IOException If the model file cannot be loaded
     */
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
            // Load the TFLite model from assets
            val modelBytes = context.assets.open(modelFile).readBytes()
            println("[DEBUG_LOG] TFLite model loaded: ${modelBytes.size} bytes")

            // Create TensorFlow Lite Interpreter
            val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size)
            modelBuffer.order(ByteOrder.nativeOrder())
            modelBuffer.put(modelBytes)

            val options = Interpreter.Options()
            options.setNumThreads(4) // Use 4 threads for better performance
            interpreter = Interpreter(modelBuffer, options)

            // Get input and output tensor info
            val inputTensor = interpreter!!.getInputTensor(0)
            val inputShape = inputTensor.shape()
            inputHeight = inputShape[1]
            inputWidth = inputShape[2]
            inputChannels = inputShape[3]

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

    /**
     * Performs YOLO inference for car segmentation using TFLite model.
     *
     * @param transformedMat The transformed image matrix (CV_32FC3)
     * @param xRatio The x ratio of the original image
     * @param yRatio The y ratio of the original image
     * @param modelWidth The width of the model input
     * @param modelHeight The height of the model input
     * @param upscaleFactor Factor by which the segmentation mask is upscaled
     * @param scoreThreshold Confidence threshold for detections
     * @param downshiftFactor Factor by which the mask is shifted down (0.0-0.1)
     * @return A binary mask where car pixels are black (0) and background pixels are white (255)
     */
    @Throws(IOException::class)
    fun inferYolo(
        transformedMat: Mat,
        xRatio: Float,
        yRatio: Float,
        modelWidth: Int,
        modelHeight: Int,
        upscaleFactor: Float = 1.2f,
        scoreThreshold: Float = this.scoreThreshold,
        downshiftFactor: Float = 0.0f
    ): Mat {
        if (!isInitialized || interpreter == null) {
            initialize()
        }

        // Create a white background mask
        val overlayGray = Mat.ones(modelHeight, modelWidth, CvType.CV_8UC1)
        overlayGray.setTo(Scalar(255.0))

        try {
            // Prepare input buffer
            val inputBuffer = ByteBuffer.allocateDirect(4 * inputWidth * inputHeight * inputChannels)
            inputBuffer.order(ByteOrder.nativeOrder())

            // Convert OpenCV Mat to ByteBuffer
            matToByteBuffer(transformedMat, inputBuffer)

            // Prepare output buffers - YOLO models typically have multiple outputs
            val outputMap = mutableMapOf<Int, Any>()

            // Get output tensor shapes
            val numOutputs = interpreter!!.outputTensorCount
            println("[DEBUG_LOG] Model has $numOutputs outputs")

            for (i in 0 until numOutputs) {
                val outputTensor = interpreter!!.getOutputTensor(i)
                val outputShape = outputTensor.shape()
                println("[DEBUG_LOG] Output $i shape: [${outputShape.joinToString(", ")}]")

                // Allocate output buffer based on shape
                val outputSize = outputShape.fold(1) { acc, dim -> acc * dim }
                val outputBuffer = ByteBuffer.allocateDirect(4 * outputSize)
                outputBuffer.order(ByteOrder.nativeOrder())
                outputMap[i] = outputBuffer
            }

            // Run inference
            interpreter!!.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

            // Process outputs to create segmentation mask
            processOutputsToMask(outputMap, overlayGray, xRatio, yRatio, modelWidth, modelHeight, upscaleFactor, scoreThreshold)

            // Apply downshift if specified
            if (downshiftFactor > 0.0f) {
                val shiftedMask = shiftDown(overlayGray, downshiftFactor)
                overlayGray.release()
                return shiftedMask
            }

            return overlayGray

        } catch (e: Exception) {
            println("[DEBUG_LOG] TFLite inference failed: ${e.message}")
            e.printStackTrace()

            // Apply downshift even in error case if specified
            if (downshiftFactor > 0.0f) {
                val shiftedMask = shiftDown(overlayGray, downshiftFactor)
                overlayGray.release()
                return shiftedMask
            }

            return overlayGray
        }
    }

    /**
     * Converts OpenCV Mat to ByteBuffer for TensorFlow Lite input.
     */
    private fun matToByteBuffer(mat: Mat, buffer: ByteBuffer) {
        buffer.rewind()

        // Resize mat to model input size if needed
        val resizedMat = Mat()
        if (mat.width() != inputWidth || mat.height() != inputHeight) {
            Imgproc.resize(mat, resizedMat, Size(inputWidth.toDouble(), inputHeight.toDouble()))
        } else {
            mat.copyTo(resizedMat)
        }

        // Convert to RGB if needed (input is already normalized from ImageProcessor)
        val rgbMat = Mat()
        if (resizedMat.channels() == 4) {
            Imgproc.cvtColor(resizedMat, rgbMat, Imgproc.COLOR_BGRA2RGB)
        } else if (resizedMat.channels() == 3) {
            Imgproc.cvtColor(resizedMat, rgbMat, Imgproc.COLOR_BGR2RGB)
        } else {
            resizedMat.copyTo(rgbMat)
        }

        // Input is already normalized to [0, 1] by ImageProcessor, just ensure it's float
        val floatMat = Mat()
        if (rgbMat.type() != CvType.CV_32FC3) {
            rgbMat.convertTo(floatMat, CvType.CV_32FC3)
        } else {
            rgbMat.copyTo(floatMat)
        }

        // Copy data to buffer
        val data = FloatArray(inputWidth * inputHeight * inputChannels)
        floatMat.get(0, 0, data)

        for (value in data) {
            buffer.putFloat(value)
        }

        resizedMat.release()
        rgbMat.release()
        floatMat.release()
    }

    /**
     * Processes TensorFlow Lite outputs to create segmentation mask.
     * Enhanced with detailed logging to diagnose model output format.
     */
    private fun processOutputsToMask(
        outputs: Map<Int, Any>,
        overlayGray: Mat,
        xRatio: Float,
        yRatio: Float,
        modelWidth: Int,
        modelHeight: Int,
        upscaleFactor: Float,
        scoreThreshold: Float
    ) {
        try {
            println("[DEBUG_LOG] Processing model outputs for mask creation")
            println("[DEBUG_LOG] Available outputs: ${outputs.keys.joinToString(", ")}")

            // For YOLO segmentation models, we typically have:
            // Output 0: Detection boxes and scores with mask coefficients [1, 116, 8400]
            // Output 1: Prototype masks [1, 32, 160, 160] (32 prototype masks at 160x160 resolution)

            val detectionsBuffer = outputs[0] as? ByteBuffer
            val prototypeMasksBuffer = outputs[1] as? ByteBuffer

            println("[DEBUG_LOG] Detections buffer available: ${detectionsBuffer != null}")
            println("[DEBUG_LOG] Prototype masks buffer available: ${prototypeMasksBuffer != null}")

            if (detectionsBuffer != null) {
                detectionsBuffer.rewind()
                println("[DEBUG_LOG] Detections buffer size: ${detectionsBuffer.capacity()} bytes")

                // Parse detections and create masks for vehicles
                val detections = parseDetections(detectionsBuffer, scoreThreshold)

                // Apply NMS to remove overlapping detections
                val filteredDetections = applyNMS(detections, nmsThreshold)
                println("[DEBUG_LOG] Filtered detections after NMS: ${filteredDetections.size}")

                // Extract prototype masks if available
                var prototypeMasks: FloatArray? = null
                if (prototypeMasksBuffer != null) {
                    prototypeMasksBuffer.rewind()
                    println("[DEBUG_LOG] Prototype masks buffer size: ${prototypeMasksBuffer.capacity()} bytes")

                    // Prototype masks shape: [1, 32, 160, 160] = 819200 floats
                    val prototypeMaskSize = 32 * 160 * 160
                    val expectedBufferSize = prototypeMaskSize * 4 // 4 bytes per float

                    println("[DEBUG_LOG] Expected prototype mask size: $prototypeMaskSize floats ($expectedBufferSize bytes)")

                    if (prototypeMasksBuffer.capacity() >= expectedBufferSize) {
                        prototypeMasks = FloatArray(prototypeMaskSize)
                        for (i in 0 until prototypeMaskSize) {
                            prototypeMasks[i] = prototypeMasksBuffer.float
                        }
                        println("[DEBUG_LOG] Successfully extracted prototype masks: ${prototypeMasks.size} values")
                    } else {
                        println("[DEBUG_LOG] ERROR: Prototype masks buffer too small: ${prototypeMasksBuffer.capacity()} bytes, expected $expectedBufferSize bytes")
                        println("[DEBUG_LOG] This indicates the model may not be a proper segmentation model")
                    }
                } else {
                    println("[DEBUG_LOG] WARNING: No prototype masks output found")
                    println("[DEBUG_LOG] This indicates the model is likely a detection-only model, not a segmentation model")
                    println("[DEBUG_LOG] Expected output 1 to contain prototype masks [1, 32, 160, 160]")
                }

                // Create masks for vehicle detections
                println("[DEBUG_LOG] Creating masks for ${filteredDetections.size} vehicle detections")
                for (detection in filteredDetections) {
                    if (vehicleClassIndices.contains(detection.classId)) {
                        println("[DEBUG_LOG] Processing detection: class=${detection.classId}, confidence=${detection.confidence}")
                        createDetectionMask(detection, overlayGray, xRatio, yRatio, modelWidth, modelHeight, upscaleFactor, prototypeMasks)
                    }
                }
                println("[DEBUG_LOG] Completed mask creation for all detections")
            } else {
                println("[DEBUG_LOG] ERROR: No detections buffer found in output 0")
                throw IllegalStateException("Model output 0 (detections) not found or invalid format")
            }
        } catch (e: Exception) {
            println("[DEBUG_LOG] CRITICAL: Error processing model outputs: ${e.message}")
            println("[DEBUG_LOG] This may indicate:")
            println("[DEBUG_LOG] 1. Model is not a YOLOv11-seg segmentation model")
            println("[DEBUG_LOG] 2. Model output format doesn't match expected structure")
            println("[DEBUG_LOG] 3. Model file is corrupted or incompatible")
            e.printStackTrace()
            throw e
        }
    }

    /**
     * Parses detection results from TensorFlow Lite output buffer for yolo11s-seg model.
     * Output format: [1, 116, 8400] where 116 = 80 classes + 4 bbox + 32 mask coefficients
     */
    private fun parseDetections(buffer: ByteBuffer, scoreThreshold: Float): List<Detection> {
        val detections = mutableListOf<Detection>()
        buffer.rewind()

        try {
            // yolo11s-seg output format: [1, 116, 8400]
            // 116 = 4 (bbox: x, y, w, h) + 80 (class scores) + 32 (mask coefficients)
            val numDetections = 8400
            val numClasses = 80
            val numMaskCoeffs = 32
            val totalFeatures = 4 + numClasses + numMaskCoeffs // 116

            println("[DEBUG_LOG] Parsing detections: $numDetections detections, $totalFeatures features each")

            for (i in 0 until numDetections) {
                // Read bbox coordinates (x, y, w, h)
                val x = buffer.float
                val y = buffer.float
                val w = buffer.float
                val h = buffer.float

                // Read class scores
                var maxScore = 0f
                var maxClassId = -1
                for (classId in 0 until numClasses) {
                    val score = buffer.float
                    if (score > maxScore) {
                        maxScore = score
                        maxClassId = classId
                    }
                }

                // Read mask coefficients
                val maskCoefficients = FloatArray(numMaskCoeffs)
                for (j in 0 until numMaskCoeffs) {
                    maskCoefficients[j] = buffer.float
                }

                // Check if this is a vehicle class with sufficient confidence
                if (maxScore > scoreThreshold && vehicleClassIndices.contains(maxClassId)) {
                    val x1 = x - w / 2
                    val y1 = y - h / 2
                    val x2 = x + w / 2
                    val y2 = y + h / 2

                    detections.add(Detection(x1, y1, x2, y2, maxScore, maxClassId, maskCoefficients))
                    println("[DEBUG_LOG] Found vehicle detection: class=$maxClassId, score=$maxScore, bbox=($x1,$y1,$x2,$y2)")
                }
            }

            println("[DEBUG_LOG] Total vehicle detections found: ${detections.size}")
        } catch (e: Exception) {
            println("[DEBUG_LOG] Error parsing detections: ${e.message}")
            e.printStackTrace()
        }

        return detections
    }


    /**
     * Applies Non-Maximum Suppression to filter overlapping detections.
     */
    private fun applyNMS(detections: List<Detection>, nmsThreshold: Float): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        // Sort by confidence (descending)
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val keep = mutableListOf<Detection>()

        for (detection in sortedDetections) {
            var shouldKeep = true

            for (kept in keep) {
                val iou = calculateIoU(detection, kept)
                if (iou > nmsThreshold) {
                    shouldKeep = false
                    break
                }
            }

            if (shouldKeep) {
                keep.add(detection)
            }
        }

        return keep
    }

    /**
     * Calculates Intersection over Union (IoU) between two detections.
     */
    private fun calculateIoU(det1: Detection, det2: Detection): Float {
        val x1 = max(det1.x1, det2.x1)
        val y1 = max(det1.y1, det2.y1)
        val x2 = min(det1.x2, det2.x2)
        val y2 = min(det1.y2, det2.y2)

        if (x2 <= x1 || y2 <= y1) return 0f

        val intersection = (x2 - x1) * (y2 - y1)
        val area1 = (det1.x2 - det1.x1) * (det1.y2 - det1.y1)
        val area2 = (det2.x2 - det2.x1) * (det2.y2 - det2.y1)
        val union = area1 + area2 - intersection

        return if (union > 0) intersection / union else 0f
    }

    /**
     * Creates a mask for a single detection using prototype masks and mask coefficients.
     * Enhanced with robust error handling and intelligent fallback strategies.
     * Prioritizes proper segmentation but provides graceful degradation when needed.
     */
    private fun createDetectionMask(
        detection: Detection,
        overlayGray: Mat,
        xRatio: Float,
        yRatio: Float,
        modelWidth: Int,
        modelHeight: Int,
        upscaleFactor: Float,
        prototypeMasks: FloatArray?
    ) {
        println("[DEBUG_LOG] Creating detection mask for class=${detection.classId}, confidence=${detection.confidence}")
        println("[DEBUG_LOG] Detection bounds: (${detection.x1}, ${detection.y1}) to (${detection.x2}, ${detection.y2})")

        // Check if we have proper segmentation capabilities
        if (prototypeMasks == null) {
            println("[DEBUG_LOG] WARNING: No prototype masks available from model")
            println("[DEBUG_LOG] Model appears to be detection-only, not segmentation model")
            println("[DEBUG_LOG] Falling back to rectangular mask (clearly identified as fallback)")

            // Use rectangular mask as fallback with clear identification
            createRectangularMask(detection, overlayGray, xRatio, yRatio, modelWidth, modelHeight, upscaleFactor)
            return
        }

        // Validate mask coefficients
        if (detection.maskCoefficients.size != 32) {
            println("[DEBUG_LOG] WARNING: Invalid mask coefficients size: ${detection.maskCoefficients.size}, expected 32")
            println("[DEBUG_LOG] Detection data may be corrupted or model format mismatch")
            println("[DEBUG_LOG] Falling back to rectangular mask (clearly identified as fallback)")

            createRectangularMask(detection, overlayGray, xRatio, yRatio, modelWidth, modelHeight, upscaleFactor)
            return
        }

        // Attempt proper segmentation mask assembly
        try {
            println("[DEBUG_LOG] Assembling proper segmentation mask from prototypes")

            val segmentationMask = assembleMaskFromPrototypes(
                detection.maskCoefficients,
                prototypeMasks,
                detection.x1, detection.y1, detection.x2, detection.y2,
                modelWidth, modelHeight
            )

            // Apply the segmentation mask to the overlay
            applySegmentationMask(segmentationMask, overlayGray, xRatio, yRatio, upscaleFactor)
            println("[DEBUG_LOG] Successfully applied proper segmentation mask")

        } catch (e: IllegalArgumentException) {
            println("[DEBUG_LOG] ERROR: Invalid parameters for mask assembly: ${e.message}")
            println("[DEBUG_LOG] Falling back to rectangular mask due to invalid inputs (clearly identified as fallback)")
            createRectangularMask(detection, overlayGray, xRatio, yRatio, modelWidth, modelHeight, upscaleFactor)

        } catch (e: Exception) {
            println("[DEBUG_LOG] ERROR: Unexpected error in mask assembly: ${e.message}")
            println("[DEBUG_LOG] ${e.javaClass.simpleName}: ${e.message}")
            println("[DEBUG_LOG] Falling back to rectangular mask due to processing error (clearly identified as fallback)")
            e.printStackTrace()
            createRectangularMask(detection, overlayGray, xRatio, yRatio, modelWidth, modelHeight, upscaleFactor)
        }
    }

    /**
     * Assembles a segmentation mask from prototype masks and mask coefficients.
     * This implements the mask assembly logic similar to the JavaScript 3-model approach.
     * Enhanced with detailed logging and robust error handling for better debugging.
     */
    private fun assembleMaskFromPrototypes(
        maskCoefficients: FloatArray,
        prototypeMasks: FloatArray,
        x1: Float, y1: Float, x2: Float, y2: Float,
        modelWidth: Int, modelHeight: Int
    ): Mat {
        // Prototype masks are 32 masks of 160x160 each
        val prototypeSize = 160
        val numPrototypes = 32
        val expectedPrototypeArraySize = numPrototypes * prototypeSize * prototypeSize

        println("[DEBUG_LOG] Starting mask assembly from prototypes")
        println("[DEBUG_LOG] Bounding box: (${x1}, ${y1}) to (${x2}, ${y2})")
        println("[DEBUG_LOG] Model dimensions: ${modelWidth}x${modelHeight}")

        // Validate input arrays
        if (maskCoefficients.size != numPrototypes) {
            throw IllegalArgumentException("Invalid mask coefficients size: expected $numPrototypes, got ${maskCoefficients.size}")
        }

        if (prototypeMasks.size != expectedPrototypeArraySize) {
            throw IllegalArgumentException("Invalid prototype masks array size: expected $expectedPrototypeArraySize (${numPrototypes} masks of ${prototypeSize}x${prototypeSize}), got ${prototypeMasks.size}")
        }

        // Validate bounding box
        if (x1 >= x2 || y1 >= y2) {
            throw IllegalArgumentException("Invalid bounding box: (${x1}, ${y1}) to (${x2}, ${y2})")
        }

        if (x1 < 0 || y1 < 0 || x2 > modelWidth || y2 > modelHeight) {
            throw IllegalArgumentException("Bounding box out of model bounds: (${x1}, ${y1}) to (${x2}, ${y2}) for model ${modelWidth}x${modelHeight}")
        }

        // Create the final mask by combining prototype masks with coefficients
        val finalMask = Mat.zeros(prototypeSize, prototypeSize, CvType.CV_32FC1)

        try {
            println("[DEBUG_LOG] Combining ${numPrototypes} prototype masks with coefficients")

            // Linear combination of prototype masks weighted by coefficients
            for (i in 0 until numPrototypes) {
                val coefficient = maskCoefficients[i]

                // Log coefficient values for debugging (only for first few and any extreme values)
                if (i < 3 || kotlin.math.abs(coefficient) > 10.0f) {
                    println("[DEBUG_LOG] Prototype $i: coefficient = $coefficient")
                }

                try {
                    // Extract the i-th prototype mask (160x160)
                    val startIdx = i * prototypeSize * prototypeSize
                    val endIdx = startIdx + prototypeSize * prototypeSize

                    if (endIdx > prototypeMasks.size) {
                        throw IndexOutOfBoundsException("Prototype mask $i extends beyond array bounds: $endIdx > ${prototypeMasks.size}")
                    }

                    val prototypeMask = Mat(prototypeSize, prototypeSize, CvType.CV_32FC1)

                    val prototypeData = FloatArray(prototypeSize * prototypeSize)
                    for (j in 0 until prototypeSize * prototypeSize) {
                        prototypeData[j] = prototypeMasks[startIdx + j]
                    }
                    prototypeMask.put(0, 0, prototypeData)

                    // Add weighted prototype to final mask
                    val weightedMask = Mat()
                    Core.multiply(prototypeMask, Scalar(coefficient.toDouble()), weightedMask)
                    Core.add(finalMask, weightedMask, finalMask)

                    prototypeMask.release()
                    weightedMask.release()

                } catch (e: Exception) {
                    println("[DEBUG_LOG] ERROR: Failed to process prototype mask $i: ${e.message}")
                    throw IllegalStateException("Failed to process prototype mask $i", e)
                }
            }

            println("[DEBUG_LOG] Successfully combined all prototype masks")

            // Apply sigmoid activation to get probabilities
            println("[DEBUG_LOG] Applying sigmoid activation")
            try {
                applySigmoid(finalMask)
                println("[DEBUG_LOG] Sigmoid activation completed")
            } catch (e: Exception) {
                println("[DEBUG_LOG] ERROR: Sigmoid activation failed: ${e.message}")
                throw IllegalStateException("Sigmoid activation failed", e)
            }

            // Crop mask to bounding box region and resize to model size
            println("[DEBUG_LOG] Cropping and resizing mask to model dimensions")
            val croppedMask = try {
                cropAndResizeMask(finalMask, x1, y1, x2, y2, modelWidth, modelHeight)
            } catch (e: Exception) {
                println("[DEBUG_LOG] ERROR: Crop and resize failed: ${e.message}")
                throw IllegalStateException("Mask cropping and resizing failed", e)
            }

            finalMask.release()
            println("[DEBUG_LOG] Mask assembly completed successfully")

            return croppedMask

        } catch (e: IllegalArgumentException) {
            println("[DEBUG_LOG] CRITICAL: Invalid arguments for mask assembly: ${e.message}")
            finalMask.release()
            throw e
        } catch (e: IllegalStateException) {
            println("[DEBUG_LOG] CRITICAL: Mask assembly process failed: ${e.message}")
            finalMask.release()
            throw e
        } catch (e: Exception) {
            println("[DEBUG_LOG] CRITICAL: Unexpected error in mask assembly: ${e.message}")
            e.printStackTrace()
            finalMask.release()
            throw IllegalStateException("Unexpected error during mask assembly", e)
        }
    }

    /**
     * Applies sigmoid activation to a mask.
     */
    private fun applySigmoid(mask: Mat) {
        val data = FloatArray((mask.total() * mask.channels()).toInt())
        mask.get(0, 0, data)

        for (i in data.indices) {
            data[i] = (1.0f / (1.0f + kotlin.math.exp(-data[i]))).toFloat()
        }

        mask.put(0, 0, data)
    }

    /**
     * Crops mask to bounding box and resizes to model dimensions.
     */
    private fun cropAndResizeMask(
        mask: Mat, 
        x1: Float, y1: Float, x2: Float, y2: Float,
        modelWidth: Int, modelHeight: Int
    ): Mat {
        val maskSize = mask.rows() // 160

        // Convert bounding box coordinates to mask coordinates (0-160 range)
        val maskX1 = ((x1 / modelWidth) * maskSize).toInt().coerceIn(0, maskSize - 1)
        val maskY1 = ((y1 / modelHeight) * maskSize).toInt().coerceIn(0, maskSize - 1)
        val maskX2 = ((x2 / modelWidth) * maskSize).toInt().coerceIn(0, maskSize - 1)
        val maskY2 = ((y2 / modelHeight) * maskSize).toInt().coerceIn(0, maskSize - 1)

        val cropWidth = max(1, maskX2 - maskX1)
        val cropHeight = max(1, maskY2 - maskY1)

        // Crop the mask to bounding box
        val cropRect = Rect(maskX1, maskY1, cropWidth, cropHeight)
        val croppedMask = Mat(mask, cropRect)

        // Resize to model dimensions
        val resizedMask = Mat()
        Imgproc.resize(croppedMask, resizedMask, Size(modelWidth.toDouble(), modelHeight.toDouble()))

        return resizedMask
    }

    /**
     * Applies the segmentation mask to the overlay.
     */
    private fun applySegmentationMask(
        segmentationMask: Mat,
        overlayGray: Mat,
        xRatio: Float,
        yRatio: Float,
        upscaleFactor: Float
    ) {
        try {
            // Convert segmentation mask to binary (threshold at 0.5)
            val binaryMask = Mat()
            Imgproc.threshold(segmentationMask, binaryMask, 0.5, 255.0, Imgproc.THRESH_BINARY)

            // Convert to 8-bit
            val mask8bit = Mat()
            binaryMask.convertTo(mask8bit, CvType.CV_8UC1)

            // Apply mask to overlay (set masked areas to black)
            val invertedMask = Mat()
            Core.bitwise_not(mask8bit, invertedMask)
            Core.bitwise_and(overlayGray, invertedMask, overlayGray)

            binaryMask.release()
            mask8bit.release()
            invertedMask.release()

        } catch (e: Exception) {
            println("[DEBUG_LOG] Error applying segmentation mask: ${e.message}")
        }
    }

    /**
     * Creates a rectangular mask for a detection (fallback method).
     */
    private fun createRectangularMask(
        detection: Detection,
        overlayGray: Mat,
        xRatio: Float,
        yRatio: Float,
        modelWidth: Int,
        modelHeight: Int,
        upscaleFactor: Float
    ) {
        // Scale detection to original image coordinates
        val scaledX1 = (detection.x1 * xRatio * upscaleFactor).toInt()
        val scaledY1 = (detection.y1 * yRatio * upscaleFactor).toInt()
        val scaledX2 = (detection.x2 * xRatio * upscaleFactor).toInt()
        val scaledY2 = (detection.y2 * yRatio * upscaleFactor).toInt()

        // Clamp to image bounds
        val x1 = max(0, min(scaledX1, modelWidth - 1))
        val y1 = max(0, min(scaledY1, modelHeight - 1))
        val x2 = max(0, min(scaledX2, modelWidth - 1))
        val y2 = max(0, min(scaledY2, modelHeight - 1))

        if (x2 > x1 && y2 > y1) {
            // Create a rectangular mask for the detection
            val rect = Rect(x1, y1, x2 - x1, y2 - y1)
            val roi = Mat(overlayGray, rect)
            roi.setTo(Scalar(0.0))  // Set detected area to black
            roi.release()
        }
    }

    /**
     * Shifts a mask down by the specified factor and fills the top with white pixels.
     *
     * @param mask The input mask to shift
     * @param downshiftFactor Factor by which to shift down (0.0-0.1)
     * @return The shifted mask
     */
    private fun shiftDown(mask: Mat, downshiftFactor: Float): Mat {
        if (downshiftFactor <= 0.0f) {
            return mask
        }

        val height = mask.rows()
        val width = mask.cols()
        val shiftPixels = (height * downshiftFactor).toInt()

        if (shiftPixels <= 0 || shiftPixels >= height) {
            return mask
        }

        // Create a new mask with the same dimensions
        val shiftedMask = Mat.zeros(height, width, mask.type())

        // Copy the original mask shifted down
        val srcRect = Rect(0, 0, width, height - shiftPixels)
        val dstRect = Rect(0, shiftPixels, width, height - shiftPixels)

        val srcRoi = Mat(mask, srcRect)
        val dstRoi = Mat(shiftedMask, dstRect)
        srcRoi.copyTo(dstRoi)

        // Fill the top part with white (255)
        val topRect = Rect(0, 0, width, shiftPixels)
        val topRoi = Mat(shiftedMask, topRect)
        topRoi.setTo(Scalar(255.0))

        // Clean up ROI mats
        srcRoi.release()
        dstRoi.release()
        topRoi.release()

        return shiftedMask
    }

    /**
     * Closes the TensorFlow Lite interpreter and releases resources.
     */
    fun close() {
        try {
            interpreter?.close()
            interpreter = null
            isInitialized = false
            println("[DEBUG_LOG] TFLite YoloInference closed")
        } catch (e: Exception) {
            println("[DEBUG_LOG] Error closing TFLite YoloInference: ${e.message}")
        }
    }

    /**
     * Data class to represent a detection.
     */
    private data class Detection(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val confidence: Float,
        val classId: Int,
        val maskCoefficients: FloatArray
    )
}
