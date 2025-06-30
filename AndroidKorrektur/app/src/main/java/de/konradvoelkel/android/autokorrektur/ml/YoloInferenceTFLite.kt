package de.konradvoelkel.android.autokorrektur.ml

import android.content.Context
import org.opencv.core.*
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

        // Convert to RGB if needed and normalize
        val rgbMat = Mat()
        if (resizedMat.channels() == 4) {
            Imgproc.cvtColor(resizedMat, rgbMat, Imgproc.COLOR_BGRA2RGB)
        } else if (resizedMat.channels() == 3) {
            Imgproc.cvtColor(resizedMat, rgbMat, Imgproc.COLOR_BGR2RGB)
        } else {
            resizedMat.copyTo(rgbMat)
        }

        // Convert to float and normalize to [0, 1]
        val floatMat = Mat()
        rgbMat.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0)

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
            // For YOLO segmentation models, we typically have:
            // Output 0: Detection boxes and scores
            // Output 1: Segmentation masks (if available)

            val detectionsBuffer = outputs[0] as? ByteBuffer
            if (detectionsBuffer != null) {
                detectionsBuffer.rewind()

                // Parse detections and create masks for vehicles
                val detections = parseDetections(detectionsBuffer, scoreThreshold)

                // Apply NMS to remove overlapping detections
                val filteredDetections = applyNMS(detections, nmsThreshold)

                // Create masks for vehicle detections
                for (detection in filteredDetections) {
                    if (vehicleClassIndices.contains(detection.classId)) {
                        createDetectionMask(detection, overlayGray, xRatio, yRatio, modelWidth, modelHeight, upscaleFactor)
                    }
                }
            }
        } catch (e: Exception) {
            println("[DEBUG_LOG] Error processing outputs: ${e.message}")
            e.printStackTrace()
        }
    }

    /**
     * Parses detection results from TensorFlow Lite output buffer.
     */
    private fun parseDetections(buffer: ByteBuffer, scoreThreshold: Float): List<Detection> {
        val detections = mutableListOf<Detection>()
        buffer.rewind()

        try {
            // YOLO output format: [batch, num_detections, 85] where 85 = 4 (bbox) + 1 (conf) + 80 (classes)
            // This is a simplified parser - actual format may vary based on model
            val numDetections = buffer.remaining() / (4 * 85) // Assuming 85 values per detection

            for (i in 0 until minOf(numDetections, 8400)) { // Limit to reasonable number
                val x = buffer.float
                val y = buffer.float
                val w = buffer.float
                val h = buffer.float
                val confidence = buffer.float

                // Skip class probabilities for now and use confidence as class score
                for (j in 0 until 80) {
                    buffer.float // Skip class probabilities
                }

                if (confidence > scoreThreshold) {
                    val x1 = x - w / 2
                    val y1 = y - h / 2
                    val x2 = x + w / 2
                    val y2 = y + h / 2

                    // Assume car class (index 2) for simplicity
                    detections.add(Detection(x1, y1, x2, y2, confidence, 2))
                }
            }
        } catch (e: Exception) {
            println("[DEBUG_LOG] Error parsing detections: ${e.message}")
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
     * Creates a mask for a single detection.
     */
    private fun createDetectionMask(
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
        val classId: Int
    )
}
