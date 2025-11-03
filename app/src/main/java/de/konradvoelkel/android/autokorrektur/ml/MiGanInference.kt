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
import org.opencv.imgproc.Imgproc
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
        private const val MODEL_FILE = "mi-gan-512.onnx"
    }

    /**
     * Initializes the Mi-GAN model session.
     *
     * @throws IOException If the model file cannot be loaded
     */
    @Throws(IOException::class)
    fun initialize() {
        if (miGanSession != null) return

        AppLogger.debug("Initializing MiGanInference...")

        val modelBytes = try {
            context.assets.open("model/$MODEL_FILE").readBytes()
        } catch (e: IOException) {
            AppLogger.debug("Failed to load Mi-GAN model: ${e.message}")
            throw IOException("Failed to load Mi-GAN model 'model/$MODEL_FILE': ${e.message}", e)
        }

        miGanSession = try {
            ortEnvironment.createSession(modelBytes)
        } catch (e: Exception) {
            AppLogger.debug("Failed to create Mi-GAN session: ${e.message}")
            throw IOException("Failed to create Mi-GAN session: ${e.message}", e)
        }

        AppLogger.debug("MiGanInference initialized successfully.")
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
        initialize() // Ensure session is initialized

        val processedImage = preprocessImage(imageMat)
        val processedMask = preprocessMask(maskMat, processedImage)

        val originalHeight = processedImage.rows()
        val originalWidth = processedImage.cols()

        val modelInputSize = org.opencv.core.Size(MODEL_INPUT_SIZE.toDouble(), MODEL_INPUT_SIZE.toDouble())
        val resizedImage = Mat()
        Imgproc.resize(processedImage, resizedImage, modelInputSize)
        val resizedMask = Mat()
        Imgproc.resize(processedMask, resizedMask, modelInputSize, 0.0, 0.0, Imgproc.INTER_NEAREST)

        val imageArray = orderInCHWAsBytes(resizedImage)
        val maskArray = orderInCHWAsBytes(resizedMask)

        val inputs = mapOf(
            "image" to createTensor(imageArray, 1, 3, MODEL_INPUT_SIZE.toLong(), MODEL_INPUT_SIZE.toLong()),
            "mask" to createTensor(maskArray, 1, 1, MODEL_INPUT_SIZE.toLong(), MODEL_INPUT_SIZE.toLong())
        )

        val outputs = miGanSession!!.run(inputs)
        val outputTensor = outputs[0]

        val outputData = getOutputData(outputTensor)
        val outputHWC = reorderToHWC(outputData, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

        val modelOutputMat = Mat(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, CvType.CV_8UC3)
        modelOutputMat.put(0, 0, outputHWC)

        val finalResultMat = Mat()
        Imgproc.resize(modelOutputMat, finalResultMat, org.opencv.core.Size(originalWidth.toDouble(), originalHeight.toDouble()))

        // Clean up
        inputs.values.forEach { it.close() }
        resizedImage.release()
        resizedMask.release()
        modelOutputMat.release()
        processedImage.release()
        processedMask.release()

        return finalResultMat
    }

    private fun preprocessImage(imageMat: Mat): Mat {
        val processedImage = Mat()
        when (imageMat.type()) {
            CvType.CV_8UC3 -> imageMat.copyTo(processedImage)
            CvType.CV_8UC4 -> Imgproc.cvtColor(imageMat, processedImage, Imgproc.COLOR_RGBA2RGB)
            else -> imageMat.convertTo(processedImage, CvType.CV_8UC3)
        }
        return processedImage
    }

    private fun preprocessMask(maskMat: Mat, imageMat: Mat): Mat {
        val processedMask = Mat()
        if (maskMat.channels() == 1 && maskMat.type() == CvType.CV_8UC1) {
            maskMat.copyTo(processedMask)
        } else {
            val gray = Mat()
            when (maskMat.channels()) {
                3 -> Imgproc.cvtColor(maskMat, gray, Imgproc.COLOR_RGB2GRAY)
                4 -> Imgproc.cvtColor(maskMat, gray, Imgproc.COLOR_RGBA2GRAY)
                else -> maskMat.convertTo(gray, CvType.CV_8UC1)
            }
            gray.convertTo(processedMask, CvType.CV_8UC1)
            gray.release()
        }

        if (processedMask.rows() != imageMat.rows() || processedMask.cols() != imageMat.cols()) {
            val resizedMask = Mat()
            Imgproc.resize(processedMask, resizedMask, imageMat.size(), 0.0, 0.0, Imgproc.INTER_NEAREST)
            processedMask.release()
            return resizedMask
        }

        return processedMask
    }
    
    private fun createTensor(data: ByteArray, batchSize: Long, channels: Long, height: Long, width: Long): OnnxTensor {
        val shape = longArrayOf(batchSize, channels, height, width)
        return OnnxTensor.createTensor(ortEnvironment, ByteBuffer.wrap(data), shape, OnnxJavaType.UINT8)
    }

    private fun getOutputData(outputTensor: ai.onnxruntime.OnnxValue): Any? {
        return if (outputTensor is OnnxTensor) {
            try {
                outputTensor.byteBuffer?.let {
                    val byteArray = ByteArray(it.remaining())
                    it.get(byteArray)
                    byteArray
                } ?: outputTensor.value
            } catch (e: Exception) {
                AppLogger.debug("Failed to get ByteBuffer from tensor, using value: ${e.message}")
                outputTensor.value
            }
        } else {
            outputTensor.value
        }
    }

    /**
     * Converts a Mat to CHW (Channel, Height, Width) format as uint8 bytes.
     *
     * @param mat The input Mat
     * @return A byte array in CHW format (uint8)
     */
    private fun orderInCHWAsBytes(mat: Mat): ByteArray {
        val channels = ArrayList<Mat>()
        Core.split(mat, channels)

        val c = channels.size
        val h = mat.rows()
        val w = mat.cols()

        val chwArray = ByteArray(c * h * w)

        for (i in 0 until c) {
            val channelMat = channels[i]
            val channelData = ByteArray(h * w)
            channelMat.get(0, 0, channelData)
            System.arraycopy(channelData, 0, chwArray, i * h * w, h * w)
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
        val hwcData = ByteArray(height * width * 3)

        when (outputData) {
            is ByteArray -> reorderCHWToHWCFromBytes(outputData, width, height, hwcData)
            is FloatArray -> reorderCHWToHWCFromFloats(outputData, width, height, hwcData)
            is Array<*> -> {
                when (val flatData = tryExtractFlatData(outputData)) {
                    is ByteArray -> reorderCHWToHWCFromBytes(flatData, width, height, hwcData)
                    is FloatArray -> reorderCHWToHWCFromFloats(flatData, width, height, hwcData)
                    else -> throw IllegalArgumentException("Could not extract usable data from nested array")
                }
            }
            else -> throw IllegalArgumentException("Unexpected model output type: ${outputData?.javaClass?.name}")
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
                    val value = if (chwIndex < floatData.size) floatData[chwIndex] * 255.0f else 0.0f
                    hwcData[(h * width + w) * 3 + c] = value.coerceIn(0f, 255f).toInt().toByte()
                }
            }
        }
    }

    /**
     * Tries to extract flat data from nested array structure.
     */
    private fun tryExtractFlatData(outputData: Array<*>): Any? {
        return when {
            outputData.isNotEmpty() && outputData[0] is ByteArray -> outputData[0] as ByteArray
            outputData.isNotEmpty() && outputData[0] is FloatArray -> outputData[0] as FloatArray
            else -> null
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