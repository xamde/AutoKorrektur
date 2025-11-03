package de.konradvoelkel.android.autokorrektur.ml

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.io.IOException
import java.nio.ByteBuffer

/**
 * Handles Mi-GAN model inference for inpainting.
 * Equivalent to miGanInference.js in the web app.
 */
class MiGanInference(private val context: Context) {

    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private var miGanSession: OrtSession? = null

    companion object {
        private const val MODEL_INPUT_SIZE = 512
    }

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

            AppLogger.debug("MiGanInference.initialize() - Loading model file:")
            AppLogger.debug("- Mi-GAN model: model/$modelFile")

            // Load the model from assets with better error handling
            val modelBytes = try {
                AppLogger.debug("Loading Mi-GAN model: model/$modelFile")
                context.assets.open("model/$modelFile").readBytes()
            } catch (e: IOException) {
                AppLogger.debug("Failed to load Mi-GAN model: ${e.message}")
                throw IOException("Failed to load Mi-GAN model 'model/$modelFile': ${e.message}", e)
            }

            AppLogger.debug("Mi-GAN model file loaded successfully, creating ONNX session...")

            // Create the session
            try {
                miGanSession = ortEnvironment.createSession(modelBytes)
                AppLogger.debug("Mi-GAN session created successfully")
            } catch (e: Exception) {
                AppLogger.debug("Failed to create Mi-GAN session: ${e.message}")
                throw IOException("Failed to create Mi-GAN session: ${e.message}", e)
            }

            AppLogger.debug("MiGanInference.initialize() completed successfully")
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

        // Ensure image is 8UC3 RGB and mask is 8UC1 with same spatial size as image
        val processedImage = Mat()
        when (imageMat.type()) {
            CvType.CV_8UC3 -> imageMat.copyTo(processedImage)
            CvType.CV_8UC4 -> {
                // Convert RGBA to RGB
                org.opencv.imgproc.Imgproc.cvtColor(
                    imageMat,
                    processedImage,
                    org.opencv.imgproc.Imgproc.COLOR_RGBA2RGB
                )
            }

            else -> {
                val tmp = Mat()
                imageMat.convertTo(tmp, CvType.CV_8UC3)
                tmp.copyTo(processedImage)
                tmp.release()
            }
        }

        val processedMask = Mat()
        // Convert mask to single channel 8-bit
        if (maskMat.channels() == 1 && maskMat.type() == CvType.CV_8UC1) {
            maskMat.copyTo(processedMask)
        } else {
            val gray = Mat()
            if (maskMat.channels() == 3) {
                org.opencv.imgproc.Imgproc.cvtColor(
                    maskMat,
                    gray,
                    org.opencv.imgproc.Imgproc.COLOR_RGB2GRAY
                )
            } else if (maskMat.channels() == 4) {
                org.opencv.imgproc.Imgproc.cvtColor(
                    maskMat,
                    gray,
                    org.opencv.imgproc.Imgproc.COLOR_RGBA2GRAY
                )
            } else {
                maskMat.convertTo(gray, CvType.CV_8UC1)
            }
            gray.convertTo(processedMask, CvType.CV_8UC1)
            gray.release()
        }

        // Resize mask to match image spatial dimensions if needed (use nearest for binary/label masks)
        if (processedMask.rows() != processedImage.rows() || processedMask.cols() != processedImage.cols()) {
            val resizedMask = Mat()
            org.opencv.imgproc.Imgproc.resize(
                processedMask,
                resizedMask,
                org.opencv.core.Size(
                    processedImage.cols().toDouble(),
                    processedImage.rows().toDouble()
                ),
                0.0,
                0.0,
                org.opencv.imgproc.Imgproc.INTER_NEAREST
            )
            processedMask.release()
            resizedMask.copyTo(processedMask)
            resizedMask.release()
        }

        val originalHeight = processedImage.rows()
        val originalWidth = processedImage.cols()

        // Resize image and mask to the model's expected input size
        val modelInputSize =
            org.opencv.core.Size(MODEL_INPUT_SIZE.toDouble(), MODEL_INPUT_SIZE.toDouble())

        val resizedImage = Mat()
        org.opencv.imgproc.Imgproc.resize(processedImage, resizedImage, modelInputSize)

        val resizedMask = Mat()
        org.opencv.imgproc.Imgproc.resize(
            processedMask,
            resizedMask,
            modelInputSize,
            0.0,
            0.0,
            org.opencv.imgproc.Imgproc.INTER_NEAREST
        )


        // Convert resized image and mask to CHW format for ONNX Runtime (uint8)
        val imageArray = orderInCHWAsBytes(resizedImage)
        val maskArray = orderInCHWAsBytes(resizedMask)

        // Create input tensors (uint8)
        val imageTensor = OnnxTensor.createTensor(
            ortEnvironment,
            ByteBuffer.wrap(imageArray),
            longArrayOf(1, 3, MODEL_INPUT_SIZE.toLong(), MODEL_INPUT_SIZE.toLong()),
            OnnxJavaType.UINT8
        )

        val maskTensor = OnnxTensor.createTensor(
            ortEnvironment,
            ByteBuffer.wrap(maskArray),
            longArrayOf(1, 1, MODEL_INPUT_SIZE.toLong(), MODEL_INPUT_SIZE.toLong()),
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
        AppLogger.debug("Output tensor type: ${outputTensor.javaClass.name}")
        AppLogger.debug("Output tensor info: ${outputTensor.info}")

        // Get the tensor data directly (similar to JavaScript outputImageTensor.data)
        val outputData = when (outputTensor) {
            is OnnxTensor -> {
                // Try to get the data as a flat array
                try {
                    val tensorData = outputTensor.byteBuffer
                    if (tensorData != null) {
                        AppLogger.debug("Got ByteBuffer from tensor, size: ${tensorData.remaining()}")
                        val byteArray = ByteArray(tensorData.remaining())
                        tensorData.get(byteArray)
                        byteArray
                    } else {
                        AppLogger.debug("ByteBuffer is null, trying value property")
                        outputTensor.value
                    }
                } catch (e: Exception) {
                    AppLogger.debug("Failed to get ByteBuffer, using value: ${e.message}")
                    outputTensor.value
                }
            }

            else -> {
                AppLogger.debug("Unexpected tensor type, using value")
                outputTensor.value
            }
        }

        AppLogger.debug("Final output data type: ${outputData?.javaClass?.name}")

        // Convert output to HWC format
        val outputHWC = reorderToHWC(outputData, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

        // Create output Mat from model output
        val modelOutputMat = Mat(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, CvType.CV_8UC3)
        modelOutputMat.put(0, 0, outputHWC)

        // Resize the output to the original image dimensions
        val finalResultMat = Mat()
        org.opencv.imgproc.Imgproc.resize(
            modelOutputMat,
            finalResultMat,
            org.opencv.core.Size(originalWidth.toDouble(), originalHeight.toDouble())
        )


        // Clean up
        imageTensor.close()
        maskTensor.close()
        resizedImage.release()
        resizedMask.release()
        modelOutputMat.release()
        processedImage.release()
        processedMask.release()


        return finalResultMat
    }

    /**
     * Converts a Mat to CHW (Channel, Height, Width) format as uint8 bytes.
     *
     * @param mat The input Mat
     * @return A byte array in CHW format (uint8)
     */
    private fun orderInCHWAsBytes(mat: Mat): ByteArray {
        // Log Mat properties for debugging
        AppLogger.debug("orderInCHWAsBytes - Mat type: ${mat.type()}, Depth: ${mat.depth()}, Channels: ${mat.channels()}")
        AppLogger.debug("orderInCHWAsBytes - Mat rows: ${mat.rows()}, cols: ${mat.cols()}")

        val channels = ArrayList<Mat>()
        Core.split(mat, channels)

        val c = channels.size
        val h = mat.rows()
        val w = mat.cols()

        val chwArray = ByteArray(c * h * w)

        for (i in 0 until c) {
            val channelMat = channels[i]
            AppLogger.debug("orderInCHWAsBytes - Channel $i type: ${channelMat.type()}, depth: ${channelMat.depth()}")

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
     * Reorders CHW image data into HWC image data.
     * Similar to the JavaScript reorderToHWC function.
     *
     * @param outputData The output data from the model
     * @param width The width of the image
     * @param height The height of the image
     * @return A byte array in HWC format
     */
    private fun reorderToHWC(outputData: Any?, width: Int, height: Int): ByteArray {
        width * height
        val hwcData = ByteArray(height * width * 3)

        // Handle different output data types - prioritize direct ByteArray handling like JavaScript
        when (outputData) {
            is ByteArray -> {
                AppLogger.debug("Processing ByteArray directly, size: ${outputData.size}")
                // Treat as uint8 data (like JavaScript uint8Data)
                reorderCHWToHWCFromBytes(outputData, width, height, hwcData)
            }

            is FloatArray -> {
                AppLogger.debug("Processing FloatArray, size: ${outputData.size}")
                reorderCHWToHWCFromFloats(outputData, width, height, hwcData)
            }

            is Array<*> -> {
                AppLogger.debug("Output is nested Array, attempting to extract data")
                try {
                    val floatArray = extractFloatArrayFromNestedArray(outputData, width, height)
                    reorderCHWToHWCFromFloats(floatArray, width, height, hwcData)
                } catch (e: Exception) {
                    AppLogger.debug("Failed to extract from nested array, trying direct access")
                    // Try to access the data more directly
                    val flatData = tryExtractFlatData(outputData)
                    if (flatData != null) {
                        when (flatData) {
                            is ByteArray -> reorderCHWToHWCFromBytes(
                                flatData,
                                width,
                                height,
                                hwcData
                            )

                            is FloatArray -> reorderCHWToHWCFromFloats(
                                flatData,
                                width,
                                height,
                                hwcData
                            )

                            else -> throw IllegalArgumentException("Could not extract usable data from nested array")
                        }
                    } else {
                        throw e
                    }
                }
            }

            else -> {
                AppLogger.debug("Unexpected output type: ${outputData?.javaClass?.name}")
                throw IllegalArgumentException("Unexpected model output type: ${outputData?.javaClass?.name}")
            }
        }

        return hwcData
    }

    /**
     * Reorders CHW byte data to HWC format (similar to JavaScript version).
     */
    private fun reorderCHWToHWCFromBytes(
        uint8Data: ByteArray,
        width: Int,
        height: Int,
        hwcData: ByteArray
    ) {
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

                    // pixelVal is already in 0..255 (uint8), assign directly
                    hwcData[(h * width + w) * 3 + c] = pixelVal.toByte()
                }
            }
        }
    }

    /**
     * Reorders CHW float data to HWC format.
     */
    private fun reorderCHWToHWCFromFloats(
        floatData: FloatArray,
        width: Int,
        height: Int,
        hwcData: ByteArray
    ) {
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
                    AppLogger.debug("Found ByteArray at index 0")
                    outputData[0] as ByteArray
                }

                outputData.isNotEmpty() && outputData[0] is FloatArray -> {
                    AppLogger.debug("Found FloatArray at index 0")
                    outputData[0] as FloatArray
                }

                else -> {
                    AppLogger.debug("Could not find flat data in nested structure")
                    null
                }
            }
        } catch (e: Exception) {
            AppLogger.debug("Error extracting flat data: ${e.message}")
            null
        }
    }


    /**
     * Extracts FloatArray from nested Array structure.
     */
    private fun extractFloatArrayFromNestedArray(
        outputData: Array<*>,
        width: Int,
        height: Int
    ): FloatArray {
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
            AppLogger.debug("Failed to extract from nested array: ${e.message}")
            throw IllegalArgumentException(
                "Failed to extract FloatArray from nested Array structure",
                e
            )
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
