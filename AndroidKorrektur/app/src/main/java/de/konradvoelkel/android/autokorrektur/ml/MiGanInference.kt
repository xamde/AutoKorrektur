package de.konradvoelkel.android.autokorrektur.ml

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.FloatBuffer

/**
 * Handles Mi-GAN model inference for inpainting.
 * Equivalent to miGanInference.js in the web app.
 */
class MiGanInference(private val context: Context) {

    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private var miGanSession: OrtSession? = null

    /**
     * Initializes the Mi-GAN model session.
     *
     * @throws IOException If the model file cannot be loaded
     */
    @Throws(IOException::class)
    fun initialize() {
        // Create the session if it doesn't exist
        if (miGanSession == null) {
            val modelFile = "mi-gan-512.onnx"

            println("[DEBUG_LOG] MiGanInference.initialize() - Loading model file:")
            println("[DEBUG_LOG] - Mi-GAN model: model/$modelFile")

            // Load the model from assets with better error handling
            val modelBytes = try {
                println("[DEBUG_LOG] Loading Mi-GAN model: model/$modelFile")
                context.assets.open("model/$modelFile").readBytes()
            } catch (e: IOException) {
                println("[DEBUG_LOG] Failed to load Mi-GAN model: ${e.message}")
                throw IOException("Failed to load Mi-GAN model 'model/$modelFile': ${e.message}", e)
            }

            println("[DEBUG_LOG] Mi-GAN model file loaded successfully, creating ONNX session...")

            // Create the session
            try {
                miGanSession = ortEnvironment.createSession(modelBytes)
                println("[DEBUG_LOG] Mi-GAN session created successfully")
            } catch (e: Exception) {
                println("[DEBUG_LOG] Failed to create Mi-GAN session: ${e.message}")
                throw IOException("Failed to create Mi-GAN session: ${e.message}", e)
            }

            println("[DEBUG_LOG] MiGanInference.initialize() completed successfully")
        }
    }

    /**
     * Performs Mi-GAN inference for inpainting.
     *
     * @param imageMat The RGB image matrix (CV_8UC3)
     * @param maskMat The mask image matrix (CV_8UC1)
     * @return The inpainted image matrix (CV_8UC3)
     */
    @Throws(IOException::class)
    fun inferMiGan(imageMat: Mat, maskMat: Mat): Mat {
        // Make sure the session is initialized
        if (miGanSession == null) {
            initialize()
        }

        val imageHeight = imageMat.rows()
        val imageWidth = imageMat.cols()

        // Convert image and mask to CHW format for ONNX Runtime (uint8)
        val imageArray = orderInCHWAsBytes(imageMat)
        val maskArray = orderInCHWAsBytes(maskMat)

        // Create input tensors (uint8)
        val imageTensor = OnnxTensor.createTensor(
            ortEnvironment,
            ByteBuffer.wrap(imageArray),
            longArrayOf(1, 3, imageHeight.toLong(), imageWidth.toLong()),
            OnnxJavaType.UINT8
        )

        val maskTensor = OnnxTensor.createTensor(
            ortEnvironment,
            ByteBuffer.wrap(maskArray),
            longArrayOf(1, 1, imageHeight.toLong(), imageWidth.toLong()),
            OnnxJavaType.UINT8
        )

        // Run Mi-GAN model
        val inputs = mapOf(
            "image" to imageTensor,
            "mask" to maskTensor
        )

        val outputs = miGanSession!!.run(inputs)

        // Get output tensor
        val outputTensor = outputs.get(0)

        // Log the actual output type for debugging
        println("[DEBUG_LOG] Output tensor type: ${outputTensor.javaClass.name}")
        println("[DEBUG_LOG] Output tensor info: ${outputTensor.info}")

        // Get the tensor data directly (similar to JavaScript outputImageTensor.data)
        val outputData = when (outputTensor) {
            is OnnxTensor -> {
                // Try to get the data as a flat array
                try {
                    val tensorData = outputTensor.byteBuffer
                    if (tensorData != null) {
                        println("[DEBUG_LOG] Got ByteBuffer from tensor, size: ${tensorData.remaining()}")
                        val byteArray = ByteArray(tensorData.remaining())
                        tensorData.get(byteArray)
                        byteArray
                    } else {
                        println("[DEBUG_LOG] ByteBuffer is null, trying value property")
                        outputTensor.value
                    }
                } catch (e: Exception) {
                    println("[DEBUG_LOG] Failed to get ByteBuffer, using value: ${e.message}")
                    outputTensor.value
                }
            }
            else -> {
                println("[DEBUG_LOG] Unexpected tensor type, using value")
                outputTensor.value
            }
        }

        println("[DEBUG_LOG] Final output data type: ${outputData?.javaClass?.name}")

        // Convert output to HWC format
        val outputHWC = reorderToHWC(outputData, imageWidth, imageHeight)

        // Create output Mat
        val resultMat = Mat(imageHeight, imageWidth, CvType.CV_8UC3)
        resultMat.put(0, 0, outputHWC)

        // Clean up
        imageTensor.close()
        maskTensor.close()

        return resultMat
    }

    /**
     * Converts a Mat to CHW (Channel, Height, Width) format as uint8 bytes.
     *
     * @param mat The input Mat
     * @return A byte array in CHW format (uint8)
     */
    private fun orderInCHWAsBytes(mat: Mat): ByteArray {
        // Log Mat properties for debugging
        println("[DEBUG_LOG] orderInCHWAsBytes - Mat type: ${mat.type()}, Depth: ${mat.depth()}, Channels: ${mat.channels()}")
        println("[DEBUG_LOG] orderInCHWAsBytes - Mat rows: ${mat.rows()}, cols: ${mat.cols()}")

        val channels = ArrayList<Mat>()
        Core.split(mat, channels)

        val c = channels.size
        val h = mat.rows()
        val w = mat.cols()

        val chwArray = ByteArray(c * h * w)

        for (i in 0 until c) {
            val channelMat = channels[i]
            println("[DEBUG_LOG] orderInCHWAsBytes - Channel $i type: ${channelMat.type()}, depth: ${channelMat.depth()}")

            // Check the data type and use appropriate array type
            if (channelMat.type() == CvType.CV_32F || channelMat.depth() == CvType.CV_32F) {
                // Handle 32-bit float data (CV_32F) - convert to uint8
                val channelData = FloatArray(h * w)
                channelMat.get(0, 0, channelData)

                for (y in 0 until h) {
                    for (x in 0 until w) {
                        // Convert from 0.0-1.0 to 0-255 uint8
                        val value = (channelData[y * w + x] * 255.0f).toInt()
                        val clampedValue = when {
                            value > 255 -> 255
                            value < 0 -> 0
                            else -> value
                        }
                        chwArray[i * h * w + y * w + x] = clampedValue.toByte()
                    }
                }
            } else {
                // Handle 8-bit unsigned data (CV_8U) - use directly
                val channelData = ByteArray(h * w)
                channelMat.get(0, 0, channelData)

                for (y in 0 until h) {
                    for (x in 0 until w) {
                        // Use uint8 values directly (0-255)
                        chwArray[i * h * w + y * w + x] = channelData[y * w + x]
                    }
                }
            }
        }

        return chwArray
    }

    /**
     * Converts a Mat to CHW (Channel, Height, Width) format.
     *
     * @param mat The input Mat
     * @return A float array in CHW format
     */
    private fun orderInCHW(mat: Mat): FloatArray {
        // Log Mat properties for debugging
        println("[DEBUG_LOG] orderInCHW - Mat type: ${mat.type()}, Depth: ${mat.depth()}, Channels: ${mat.channels()}")
        println("[DEBUG_LOG] orderInCHW - Mat rows: ${mat.rows()}, cols: ${mat.cols()}")
        println("[DEBUG_LOG] orderInCHW - CV_32F constant: ${CvType.CV_32F}")

        val channels = ArrayList<Mat>()
        Core.split(mat, channels)

        val c = channels.size
        val h = mat.rows()
        val w = mat.cols()

        val chwArray = FloatArray(c * h * w)

        for (i in 0 until c) {
            val channelMat = channels[i]
            println("[DEBUG_LOG] orderInCHW - Channel $i type: ${channelMat.type()}, depth: ${channelMat.depth()}")

            // Check the data type and use appropriate array type
            if (channelMat.type() == CvType.CV_32F || channelMat.depth() == CvType.CV_32F) {
                // Handle 32-bit float data (CV_32F)
                val channelData = FloatArray(h * w)
                channelMat.get(0, 0, channelData)

                for (y in 0 until h) {
                    for (x in 0 until w) {
                        // Data is already normalized (0.0-1.0), so use directly
                        chwArray[i * h * w + y * w + x] = channelData[y * w + x]
                    }
                }
            } else {
                // Handle 8-bit unsigned data (CV_8U)
                val channelData = ByteArray(h * w)
                channelMat.get(0, 0, channelData)

                for (y in 0 until h) {
                    for (x in 0 until w) {
                        // Convert from 0-255 to 0.0-1.0
                        chwArray[i * h * w + y * w + x] = (channelData[y * w + x].toInt() and 0xFF) / 255.0f
                    }
                }
            }
        }

        return chwArray
    }

    /**
     * Reorders CHW image data into HWC image data.
     * Similar to the JavaScript reorderToHWC function.
     *
     * @param outputData The output data from the model
     * @param width The width of the image
     * @param height The height of the image
     * @return A byte array in HWC format
     */
    private fun reorderToHWC(outputData: Any?, width: Int, height: Int): ByteArray {
        val size = width * height
        val hwcData = ByteArray(height * width * 3)

        // Handle different output data types - prioritize direct ByteArray handling like JavaScript
        when (outputData) {
            is ByteArray -> {
                println("[DEBUG_LOG] Processing ByteArray directly, size: ${outputData.size}")
                // Treat as uint8 data (like JavaScript uint8Data)
                reorderCHWToHWCFromBytes(outputData, width, height, hwcData)
            }
            is FloatArray -> {
                println("[DEBUG_LOG] Processing FloatArray, size: ${outputData.size}")
                reorderCHWToHWCFromFloats(outputData, width, height, hwcData)
            }
            is Array<*> -> {
                println("[DEBUG_LOG] Output is nested Array, attempting to extract data")
                try {
                    val floatArray = extractFloatArrayFromNestedArray(outputData, width, height)
                    reorderCHWToHWCFromFloats(floatArray, width, height, hwcData)
                } catch (e: Exception) {
                    println("[DEBUG_LOG] Failed to extract from nested array, trying direct access")
                    // Try to access the data more directly
                    val flatData = tryExtractFlatData(outputData)
                    if (flatData != null) {
                        when (flatData) {
                            is ByteArray -> reorderCHWToHWCFromBytes(flatData, width, height, hwcData)
                            is FloatArray -> reorderCHWToHWCFromFloats(flatData, width, height, hwcData)
                            else -> throw IllegalArgumentException("Could not extract usable data from nested array")
                        }
                    } else {
                        throw e
                    }
                }
            }
            else -> {
                println("[DEBUG_LOG] Unexpected output type: ${outputData?.javaClass?.name}")
                throw IllegalArgumentException("Unexpected model output type: ${outputData?.javaClass?.name}")
            }
        }

        return hwcData
    }

    /**
     * Reorders CHW byte data to HWC format (similar to JavaScript version).
     */
    private fun reorderCHWToHWCFromBytes(uint8Data: ByteArray, width: Int, height: Int, hwcData: ByteArray) {
        val size = width * height

        for (h in 0 until height) {
            for (w in 0 until width) {
                for (c in 0 until 3) {
                    val chwIndex = c * size + h * width + w
                    val pixelVal = if (chwIndex < uint8Data.size) {
                        uint8Data[chwIndex].toInt() and 0xFF
                    } else {
                        0
                    }

                    // Clamp pixel value (like JavaScript)
                    val newPixel = when {
                        pixelVal > 255 -> 255
                        pixelVal < 0 -> 0
                        else -> pixelVal
                    }

                    hwcData[(h * width + w) * 3 + c] = newPixel.toByte()
                }
            }
        }
    }

    /**
     * Reorders CHW float data to HWC format.
     */
    private fun reorderCHWToHWCFromFloats(floatData: FloatArray, width: Int, height: Int, hwcData: ByteArray) {
        val size = width * height

        for (h in 0 until height) {
            for (w in 0 until width) {
                for (c in 0 until 3) {
                    val chwIndex = c * size + h * width + w
                    val value = if (chwIndex < floatData.size) {
                        floatData[chwIndex] * 255.0f
                    } else {
                        0.0f
                    }

                    // Clamp value to 0-255 range
                    val byteVal = when {
                        value > 255 -> 255
                        value < 0 -> 0
                        else -> value.toInt()
                    }

                    hwcData[(h * width + w) * 3 + c] = byteVal.toByte()
                }
            }
        }
    }

    /**
     * Tries to extract flat data from nested array structure.
     */
    private fun tryExtractFlatData(outputData: Array<*>): Any? {
        return try {
            // Try different ways to access the flat data
            when {
                outputData.isNotEmpty() && outputData[0] is ByteArray -> {
                    println("[DEBUG_LOG] Found ByteArray at index 0")
                    outputData[0] as ByteArray
                }
                outputData.isNotEmpty() && outputData[0] is FloatArray -> {
                    println("[DEBUG_LOG] Found FloatArray at index 0")
                    outputData[0] as FloatArray
                }
                else -> {
                    println("[DEBUG_LOG] Could not find flat data in nested structure")
                    null
                }
            }
        } catch (e: Exception) {
            println("[DEBUG_LOG] Error extracting flat data: ${e.message}")
            null
        }
    }

    /**
     * Converts a ByteArray to FloatArray assuming IEEE 754 float representation.
     */
    private fun convertByteArrayToFloatArray(byteArray: ByteArray): FloatArray {
        if (byteArray.size % 4 != 0) {
            println("[DEBUG_LOG] ByteArray size is not a multiple of 4, treating as uint8 values")
            // If not divisible by 4, treat as uint8 values and convert to 0.0-1.0 range
            return FloatArray(byteArray.size) { i ->
                (byteArray[i].toInt() and 0xFF) / 255.0f
            }
        }

        val buffer = ByteBuffer.wrap(byteArray).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        val floatArray = FloatArray(byteArray.size / 4)
        for (i in floatArray.indices) {
            floatArray[i] = buffer.getFloat(i * 4)
        }
        return floatArray
    }

    /**
     * Extracts FloatArray from nested Array structure.
     */
    private fun extractFloatArrayFromNestedArray(outputData: Array<*>, width: Int, height: Int): FloatArray {
        return try {
            val size = width * height * 3
            val floatArray = FloatArray(size)
            var index = 0

            for (c in 0 until 3) {
                val pixelVal = (outputData[0] as Array<*>)[c] as Array<*>
                for (i in 0 until (width * height)) {
                    if (index < floatArray.size) {
                        floatArray[index] = pixelVal[i] as Float
                        index++
                    }
                }
            }
            floatArray
        } catch (e: Exception) {
            println("[DEBUG_LOG] Failed to extract from nested array: ${e.message}")
            throw IllegalArgumentException("Failed to extract FloatArray from nested Array structure", e)
        }
    }

    /**
     * Releases resources used by the inference session.
     */
    fun close() {
        miGanSession?.close()
        ortEnvironment.close()
    }
}
