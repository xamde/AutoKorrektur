package de.konradvoelkel.android.autokorrektur.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import kotlin.math.floor
import kotlin.math.max

/**
 * Handles YOLO model inference for car segmentation.
 * Equivalent to yoloInference.js in the web app.
 */
class YoloInference(private val context: Context) {

    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private var yoloSession: OrtSession? = null
    private var nmsSession: OrtSession? = null
    private var maskSession: OrtSession? = null

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
    private val topAmountPerClass = 100 // top amount of Instances per class
    private val intersectionOverUnionThreshold = 0.9f // iou threshold
    private val baseScoreThreshold = 0.2f // score threshold
    private val numClass = labels.size

    // Vehicle class indices (car, motorcycle, truck)
    private val vehicleClassIndices = intArrayOf(2, 3, 7)

    /**
     * Initializes the YOLO model sessions.
     *
     * @param modelName The name of the YOLO model to use (e.g., "yolo11s")
     * @throws IOException If the model files cannot be loaded
     */
    @Throws(IOException::class)
    fun initialize(modelName: String = "yolo11s") {
        // Create the sessions if they don't exist
        if (yoloSession == null || nmsSession == null || maskSession == null) {
            val segModelFile = "$modelName-seg.onnx"
            val nmsModelFile = "nms-yolov8.onnx"
            val maskModelFile = "mask-yolov8-seg.onnx"

            println("[DEBUG_LOG] YoloInference.initialize() - Loading model files:")
            println("[DEBUG_LOG] - Segmentation model: model/$segModelFile")
            println("[DEBUG_LOG] - NMS model: model/$nmsModelFile")
            println("[DEBUG_LOG] - Mask model: model/$maskModelFile")

            // Load the models from assets with better error handling
            val segModelBytes = try {
                println("[DEBUG_LOG] Loading segmentation model: model/$segModelFile")
                context.assets.open("model/$segModelFile").readBytes()
            } catch (e: IOException) {
                println("[DEBUG_LOG] Failed to load segmentation model: ${e.message}")
                throw IOException("Failed to load segmentation model 'model/$segModelFile': ${e.message}", e)
            }

            val nmsModelBytes = try {
                println("[DEBUG_LOG] Loading NMS model: model/$nmsModelFile")
                context.assets.open("model/$nmsModelFile").readBytes()
            } catch (e: IOException) {
                println("[DEBUG_LOG] Failed to load NMS model: ${e.message}")
                throw IOException("Failed to load NMS model 'model/$nmsModelFile': ${e.message}", e)
            }

            val maskModelBytes = try {
                println("[DEBUG_LOG] Loading mask model: model/$maskModelFile")
                context.assets.open("model/$maskModelFile").readBytes()
            } catch (e: IOException) {
                println("[DEBUG_LOG] Failed to load mask model: ${e.message}")
                throw IOException("Failed to load mask model 'model/$maskModelFile': ${e.message}", e)
            }

            println("[DEBUG_LOG] All model files loaded successfully, creating ONNX sessions...")

            // Log model file details for debugging
            println("[DEBUG_LOG] Model file sizes:")
            println("[DEBUG_LOG] - Segmentation model: ${segModelBytes.size} bytes")
            println("[DEBUG_LOG] - NMS model: ${nmsModelBytes.size} bytes") 
            println("[DEBUG_LOG] - Mask model: ${maskModelBytes.size} bytes")

            // Log ONNX Runtime environment details
            println("[DEBUG_LOG] ONNX Runtime Environment details:")
            try {
                println("[DEBUG_LOG] - Environment initialized: ${ortEnvironment != null}")
                println("[DEBUG_LOG] - Environment class: ${ortEnvironment.javaClass.simpleName}")
            } catch (e: Exception) {
                println("[DEBUG_LOG] - Could not get environment details: ${e.message}")
            }

            // Create the sessions
            try {
                println("[DEBUG_LOG] Creating YOLO session with segmentation model...")
                println("[DEBUG_LOG] - Model size: ${segModelBytes.size} bytes")
                println("[DEBUG_LOG] - First 10 bytes: ${segModelBytes.take(10).joinToString { "%02x".format(it) }}")

                // Validate model file format and integrity
                validateModelFile(segModelBytes, "Segmentation")

                yoloSession = ortEnvironment.createSession(segModelBytes)
                println("[DEBUG_LOG] YOLO session created successfully")
            } catch (e: Exception) {
                println("[DEBUG_LOG] ========== YOLO SESSION CREATION FAILED ==========")
                println("[DEBUG_LOG] Exception type: ${e.javaClass.simpleName}")
                println("[DEBUG_LOG] Exception message: ${e.message}")
                println("[DEBUG_LOG] Exception cause: ${e.cause?.message ?: "None"}")
                println("[DEBUG_LOG] Stack trace:")
                e.printStackTrace()
                println("[DEBUG_LOG] ================================================")

                // Provide more specific error information for Segmentation model
                val errorDetails = when {
                    e.message?.contains("corrupted", ignoreCase = true) == true -> 
                        "Segmentation model file appears to be corrupted. Try re-downloading the model files."
                    e.message?.contains("empty", ignoreCase = true) == true -> 
                        "Segmentation model file is empty or missing from assets/model/"
                    e.message?.contains("small", ignoreCase = true) == true -> 
                        "Segmentation model file is too small and likely corrupted."
                    e.message?.contains("valid ONNX", ignoreCase = true) == true -> 
                        "Segmentation model file is not a valid ONNX format."
                    e.message?.contains("unsupported", ignoreCase = true) == true -> 
                        "Segmentation model format may be unsupported by this ONNX Runtime version."
                    e.message?.contains("memory", ignoreCase = true) == true -> 
                        "Insufficient memory to load the segmentation model. Try closing other apps."
                    else -> 
                        "Check logcat for detailed stack trace."
                }

                throw IOException("Failed to create YOLO session: ${e.message}. $errorDetails", e)
            }

            try {
                println("[DEBUG_LOG] Creating NMS session...")
                println("[DEBUG_LOG] - NMS model size: ${nmsModelBytes.size} bytes")

                // Validate NMS model file
                validateModelFile(nmsModelBytes, "NMS")

                nmsSession = ortEnvironment.createSession(nmsModelBytes)
                println("[DEBUG_LOG] NMS session created successfully")
            } catch (e: Exception) {
                println("[DEBUG_LOG] ========== NMS SESSION CREATION FAILED ==========")
                println("[DEBUG_LOG] Exception type: ${e.javaClass.simpleName}")
                println("[DEBUG_LOG] Exception message: ${e.message}")
                println("[DEBUG_LOG] Exception cause: ${e.cause?.message ?: "None"}")
                e.printStackTrace()
                println("[DEBUG_LOG] ===============================================")

                // Provide more specific error information for NMS model
                val errorDetails = when {
                    e.message?.contains("corrupted", ignoreCase = true) == true -> 
                        "NMS model file appears to be corrupted. Try re-downloading the model files."
                    e.message?.contains("empty", ignoreCase = true) == true -> 
                        "NMS model file is empty or missing from assets/model/"
                    e.message?.contains("small", ignoreCase = true) == true -> 
                        "NMS model file is too small and likely corrupted."
                    e.message?.contains("valid ONNX", ignoreCase = true) == true -> 
                        "NMS model file is not a valid ONNX format."
                    else -> "Check logcat for detailed stack trace."
                }

                throw IOException("Failed to create NMS session: ${e.message}. $errorDetails", e)
            }

            try {
                println("[DEBUG_LOG] Creating Mask session...")
                println("[DEBUG_LOG] - Mask model size: ${maskModelBytes.size} bytes")

                // Validate Mask model file
                validateModelFile(maskModelBytes, "Mask")

                maskSession = ortEnvironment.createSession(maskModelBytes)
                println("[DEBUG_LOG] Mask session created successfully")
            } catch (e: Exception) {
                println("[DEBUG_LOG] ========== MASK SESSION CREATION FAILED ==========")
                println("[DEBUG_LOG] Exception type: ${e.javaClass.simpleName}")
                println("[DEBUG_LOG] Exception message: ${e.message}")
                println("[DEBUG_LOG] Exception cause: ${e.cause?.message ?: "None"}")
                e.printStackTrace()
                println("[DEBUG_LOG] ================================================")

                // Provide more specific error information for Mask model
                val errorDetails = when {
                    e.message?.contains("corrupted", ignoreCase = true) == true -> 
                        "Mask model file appears to be corrupted. Try re-downloading the model files."
                    e.message?.contains("empty", ignoreCase = true) == true -> 
                        "Mask model file is empty or missing from assets/model/"
                    e.message?.contains("small", ignoreCase = true) == true -> 
                        "Mask model file is too small and likely corrupted."
                    e.message?.contains("valid ONNX", ignoreCase = true) == true -> 
                        "Mask model file is not a valid ONNX format."
                    else -> "Check logcat for detailed stack trace."
                }

                throw IOException("Failed to create mask session: ${e.message}. $errorDetails", e)
            }

            println("[DEBUG_LOG] YoloInference.initialize() completed successfully")
        }
    }

    /**
     * Performs YOLO inference for car segmentation.
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
        scoreThreshold: Float = baseScoreThreshold,
        downshiftFactor: Float = 0.0f
    ): Mat {
        // Make sure the sessions are initialized
        if (yoloSession == null || nmsSession == null || maskSession == null) {
            initialize()
        }

        // Create a grayscale mask
        val overlayGray = Mat.ones(modelHeight, modelWidth, CvType.CV_8UC1)
        overlayGray.setTo(Scalar(255.0))  // Set to white

        // Convert the Mat to a format suitable for ONNX Runtime
        val inputTensor = createInputTensor(transformedMat)

        try {
            // Run YOLO model
            val yoloInputs = mapOf("images" to inputTensor)
            val yoloSession = this.yoloSession ?: throw IOException("YOLO session is null")
            val yoloOutputs = yoloSession.run(yoloInputs)

            try {
                // Get output tensors
                val output0 = yoloOutputs.get(0)
                val output1 = yoloOutputs.get(1)

                if (output0 == null || output1 == null) {
                    return overlayGray
                }

                // Create NMS config tensor
                val nmsConfigBuffer = FloatBuffer.allocate(4)
                nmsConfigBuffer.put(numClass.toFloat())  // num class
                nmsConfigBuffer.put(topAmountPerClass.toFloat())  // top amount per class
                nmsConfigBuffer.put(intersectionOverUnionThreshold)  // iou threshold
                nmsConfigBuffer.put(scoreThreshold)  // score threshold
                nmsConfigBuffer.rewind()

                val nmsConfigTensor = OnnxTensor.createTensor(ortEnvironment, nmsConfigBuffer, longArrayOf(4))

                try {
                    // Run NMS model
                    val nmsSession = this.nmsSession ?: throw IOException("NMS session is null")
                    val nmsInputs: Map<String, OnnxTensor> = mapOf(
                        "detection" to output0 as OnnxTensor,
                        "config" to nmsConfigTensor
                    )
                    val nmsOutputs = nmsSession.run(nmsInputs)

                    try {
                        // Process detected objects
                        val selectedOutput = nmsOutputs.get(0)
                        if (selectedOutput == null) {
                            return overlayGray
                        }

                        // Extract the selected detections
                        val selectedData = selectedOutput.value as? Array<*>
                        if (selectedData == null || selectedData.isEmpty()) {
                            return overlayGray
                        }

                        val maxSize = max(modelWidth, modelHeight)
                        val maskSession = this.maskSession ?: throw IOException("Mask session is null")

                        for (idx in 0 until selectedData.size) {
                            val data = (selectedData[idx] as? FloatArray) ?: continue

                            // Skip if data is too small
                            if (data.size < 4 + numClass + 32) {
                                continue
                            }

                            // Get bounding box
                            val box = floatArrayOf(data[0], data[1], data[2], data[3])

                            // Get class scores
                            val scores = data.sliceArray(4 until 4 + numClass)
                            val score = scores.maxOrNull() ?: 0f
                            val labelIndex = scores.indexOfFirst { it == score }

                            // Skip if not a vehicle or invalid label
                            if (labelIndex < 0 || !vehicleClassIndices.contains(labelIndex)) {
                                continue
                            }

                            // Handle box overflow
                            val adjustedBox = overflowBoxes(
                                floatArrayOf(
                                    box[0] - 0.5f * box[2],  // x
                                    box[1] - 0.5f * box[3],  // y
                                    box[2],  // width
                                    box[3]   // height
                                ),
                                maxSize.toFloat()
                            )

                            // Scale box to original image size
                            val scaledBox = overflowBoxes(
                                floatArrayOf(
                                    floor(adjustedBox[0] * xRatio.toDouble()).toFloat(),  // x
                                    floor(adjustedBox[1] * yRatio.toDouble()).toFloat(),  // y
                                    floor(adjustedBox[2] * xRatio.toDouble()).toFloat(),  // width
                                    floor(adjustedBox[3] * yRatio.toDouble()).toFloat()   // height
                                ),
                                maxSize.toFloat()
                            )

                            // Create mask input tensor
                            val maskInputBuffer = FloatBuffer.allocate(4 + 32)  // box + mask coefficients
                            maskInputBuffer.put(adjustedBox)

                            try {
                                for (i in 4 + numClass until 4 + numClass + 32) {
                                    if (i < data.size) {
                                        maskInputBuffer.put(data[i])
                                    } else {
                                        maskInputBuffer.put(0f)
                                    }
                                }
                                maskInputBuffer.rewind()

                                val maskInputTensor = OnnxTensor.createTensor(
                                    ortEnvironment, maskInputBuffer, longArrayOf(36)
                                )

                                try {
                                    // Reposition the mask with upscale
                                    val newX = scaledBox[0] - ((scaledBox[2] * upscaleFactor) - scaledBox[2]) / 2
                                    val newY = scaledBox[1] - ((scaledBox[3] * upscaleFactor) - scaledBox[3]) / 2

                                    // Create mask config tensor
                                    val maskConfigBuffer = FloatBuffer.allocate(10)
                                    maskConfigBuffer.put(maxSize.toFloat())
                                    maskConfigBuffer.put(newX)
                                    maskConfigBuffer.put(newY)
                                    maskConfigBuffer.put(scaledBox[2] * upscaleFactor)
                                    maskConfigBuffer.put(scaledBox[3] * upscaleFactor)
                                    maskConfigBuffer.put(2f)  // fixed color for mask model
                                    maskConfigBuffer.put(2f)
                                    maskConfigBuffer.put(2f)
                                    maskConfigBuffer.put(255f)
                                    maskConfigBuffer.rewind()

                                    val maskConfigTensor = OnnxTensor.createTensor(
                                        ortEnvironment, maskConfigBuffer, longArrayOf(10)
                                    )

                                    try {
                                        // Run mask model
                                        val maskInputs: Map<String, OnnxTensor> = mapOf(
                                            "detection" to maskInputTensor,
                                            "mask" to output1 as OnnxTensor,
                                            "config" to maskConfigTensor
                                        )
                                        val maskOutputs = maskSession.run(maskInputs)

                                        try {
                                            // Convert mask output to Mat
                                            val maskFilterOutput = maskOutputs.get(0)
                                            if (maskFilterOutput != null) {
                                                val maskFilter = maskFilterOutput.value as? Array<*>
                                                if (maskFilter != null) {
                                                    val maskMat = createMaskMat(maskFilter)

                                                    // Subtract mask from overlay (masked area will be black)
                                                    Core.subtract(overlayGray, maskMat, overlayGray)

                                                    maskMat.release()
                                                }
                                            }
                                        } finally {
                                            // Close all mask outputs
                                            maskOutputs.close()
                                        }
                                    } finally {
                                        maskConfigTensor.close()
                                    }
                                } finally {
                                    maskInputTensor.close()
                                }
                            } catch (e: Exception) {
                                // Log error and continue with next detection
                                e.printStackTrace()
                            }
                        }

                        // Apply downshift if specified
                        if (downshiftFactor > 0.0f) {
                            val shiftedMask = shiftDown(overlayGray, downshiftFactor)
                            overlayGray.release()
                            return shiftedMask
                        }

                        return overlayGray
                    } finally {
                        // Close all NMS outputs
                        nmsOutputs.close()
                    }
                } finally {
                    nmsConfigTensor.close()
                }
            } finally {
                // Clean up
                yoloOutputs.close()
            }
        } finally {
            inputTensor.close()
        }
    }

    /**
     * Creates an input tensor from a Mat.
     *
     * @param mat The input Mat (CV_32FC3)
     * @return An OnnxTensor suitable for model input
     */
    private fun createInputTensor(mat: Mat): OnnxTensor {
        val height = mat.rows()
        val width = mat.cols()
        val channels = mat.channels()

        // Create a float buffer for the tensor
        val buffer = FloatBuffer.allocate(height * width * channels)

        // Convert Mat to CHW format
        val matChannels = ArrayList<Mat>()
        Core.split(mat, matChannels)

        for (c in 0 until channels) {
            val channelMat = matChannels[c]
            val channelBuffer = FloatBuffer.allocate(height * width)
            channelMat.get(0, 0, channelBuffer.array())

            for (i in 0 until height * width) {
                buffer.put(i + c * height * width, channelBuffer.get(i))
            }
        }

        return OnnxTensor.createTensor(
            ortEnvironment, buffer, longArrayOf(1, channels.toLong(), height.toLong(), width.toLong())
        )
    }

    /**
     * Creates a mask Mat from the mask filter output.
     *
     * @param maskFilter The mask filter output from the mask model
     * @return A binary mask Mat
     */
    private fun createMaskMat(maskFilter: Array<*>): Mat {
        val height = (maskFilter[0] as Array<*>).size
        val width = ((maskFilter[0] as Array<*>)[0] as Array<*>).size

        // Create a Mat for the mask
        val maskMat = Mat(height, width, CvType.CV_8UC4)
        val maskData = ByteBuffer.allocate(height * width * 4)

        // Fill the buffer with mask data
        for (h in 0 until height) {
            for (w in 0 until width) {
                val pixel = ((maskFilter[0] as Array<*>)[h] as Array<*>)[w] as FloatArray
                for (c in 0 until 4) {
                    val pixelValue = pixel[c].toInt()
                    maskData.put((h * width + w) * 4 + c, pixelValue.toByte())
                }
            }
        }

        maskMat.put(0, 0, maskData.array())

        // Convert to grayscale and threshold
        val grayMat = Mat()
        Imgproc.cvtColor(maskMat, grayMat, Imgproc.COLOR_BGRA2GRAY)
        Imgproc.threshold(grayMat, grayMat, 1.0, 255.0, Imgproc.THRESH_BINARY)

        maskMat.release()

        return grayMat
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
        val srcRect = org.opencv.core.Rect(0, 0, width, height - shiftPixels)
        val dstRect = org.opencv.core.Rect(0, shiftPixels, width, height - shiftPixels)

        val srcRoi = Mat(mask, srcRect)
        val dstRoi = Mat(shiftedMask, dstRect)
        srcRoi.copyTo(dstRoi)

        // Fill the top part with white (255)
        val topRect = org.opencv.core.Rect(0, 0, width, shiftPixels)
        val topRoi = Mat(shiftedMask, topRect)
        topRoi.setTo(Scalar(255.0))

        // Clean up ROI mats
        srcRoi.release()
        dstRoi.release()
        topRoi.release()

        return shiftedMask
    }

    /**
     * Handle overflow boxes based on maxSize.
     *
     * @param box Box in [x, y, w, h] format
     * @param maxSize Maximum size
     * @return Non-overflow boxes
     */
    private fun overflowBoxes(box: FloatArray, maxSize: Float): FloatArray {
        val result = box.clone()
        result[0] = if (result[0] >= 0) result[0] else 0f
        result[1] = if (result[1] >= 0) result[1] else 0f
        result[2] = if (result[0] + result[2] <= maxSize) result[2] else maxSize - result[0]
        result[3] = if (result[1] + result[3] <= maxSize) result[3] else maxSize - result[1]
        return result
    }

    /**
     * Releases resources used by the inference sessions.
     */
    fun close() {
        yoloSession?.close()
        nmsSession?.close()
        maskSession?.close()
        ortEnvironment.close()
    }

    /**
     * Validates an ONNX model file for corruption and format issues.
     *
     * @param modelBytes The model file bytes to validate
     * @param modelType The type of model (for error messages)
     * @throws IOException If the model file is corrupted or invalid
     */
    @Throws(IOException::class)
    private fun validateModelFile(modelBytes: ByteArray, modelType: String) {
        println("[DEBUG_LOG] Validating $modelType model file...")

        // Check minimum file size
        if (modelBytes.isEmpty()) {
            throw IOException("$modelType model file is empty. The file may be missing or corrupted.")
        }

        if (modelBytes.size < 16) {
            throw IOException("$modelType model file is too small (${modelBytes.size} bytes). Expected ONNX model file (minimum ~16 bytes).")
        }

        // Check for suspiciously small files that might indicate corruption
        if (modelBytes.size < 1000) {
            throw IOException("$modelType model file is suspiciously small (${modelBytes.size} bytes). This likely indicates a corrupted or incomplete download.")
        }

        // Validate ONNX protobuf format
        // ONNX files are protobuf format, they should start with valid protobuf field tags
        // Common patterns: 0x08 (field 1, varint), 0x12 (field 2, length-delimited), etc.
        val firstByte = modelBytes[0].toInt() and 0xFF
        val isValidProtobuf = (firstByte and 0x07) <= 5 && (firstByte shr 3) > 0

        if (!isValidProtobuf) {
            println("[DEBUG_LOG] $modelType model validation failed:")
            println("[DEBUG_LOG] - File size: ${modelBytes.size} bytes")
            println("[DEBUG_LOG] - First byte: 0x${"%02x".format(firstByte)}")
            println("[DEBUG_LOG] - First 16 bytes: ${modelBytes.take(16).joinToString(" ") { "%02x".format(it.toInt() and 0xFF) }}")
            throw IOException("$modelType model file does not appear to be a valid ONNX file. The file may be corrupted or in the wrong format.")
        }

        // Additional validation: check for common corruption patterns
        val allZeros = modelBytes.take(100).all { it == 0.toByte() }
        val allSame = modelBytes.take(100).all { it == modelBytes[0] }

        if (allZeros) {
            throw IOException("$modelType model file appears to be corrupted (contains only zero bytes).")
        }

        if (allSame && modelBytes.size > 100) {
            throw IOException("$modelType model file appears to be corrupted (contains repeated identical bytes).")
        }

        // Check for text file corruption (sometimes binary files get corrupted into text)
        // Note: ONNX files legitimately contain many text strings (layer names, metadata, etc.)
        // so we use a more lenient threshold and also check for obvious text file patterns
        val firstChunk = modelBytes.take(100)
        val textBytes = firstChunk.count { byte ->
            val b = byte.toInt() and 0xFF
            b in 32..126 || b == 9 || b == 10 || b == 13 // printable ASCII + tab/newline/carriage return
        }

        // Only flag as corrupted if it's almost entirely text AND doesn't start with valid protobuf
        if (textBytes > firstChunk.size * 0.95 && !isValidProtobuf) {
            throw IOException("$modelType model file appears to be corrupted (contains mostly text characters instead of binary data).")
        }

        println("[DEBUG_LOG] $modelType model file validation passed")
        println("[DEBUG_LOG] - File size: ${modelBytes.size} bytes")
        println("[DEBUG_LOG] - First byte: 0x${"%02x".format(firstByte)}")
    }
}
