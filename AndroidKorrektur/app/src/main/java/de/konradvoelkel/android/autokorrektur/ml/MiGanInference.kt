package de.konradvoelkel.android.autokorrektur.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.io.IOException
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

        // Convert image and mask to CHW format for ONNX Runtime
        val imageArray = orderInCHW(imageMat)
        val maskArray = orderInCHW(maskMat)

        // Create input tensors
        val imageTensor = OnnxTensor.createTensor(
            ortEnvironment,
            FloatBuffer.wrap(imageArray),
            longArrayOf(1, 3, imageHeight.toLong(), imageWidth.toLong())
        )

        val maskTensor = OnnxTensor.createTensor(
            ortEnvironment,
            FloatBuffer.wrap(maskArray),
            longArrayOf(1, 1, imageHeight.toLong(), imageWidth.toLong())
        )

        // Run Mi-GAN model
        val inputs = mapOf(
            "image" to imageTensor,
            "mask" to maskTensor
        )

        val outputs = miGanSession!!.run(inputs)

        // Get output tensor
        val outputTensor = outputs.get(0)
        val outputData = outputTensor.value as Array<*>

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
     * Converts a Mat to CHW (Channel, Height, Width) format.
     *
     * @param mat The input Mat
     * @return A float array in CHW format
     */
    private fun orderInCHW(mat: Mat): FloatArray {
        val channels = ArrayList<Mat>()
        Core.split(mat, channels)

        val c = channels.size
        val h = mat.rows()
        val w = mat.cols()

        val chwArray = FloatArray(c * h * w)

        for (i in 0 until c) {
            val channelMat = channels[i]
            val channelData = ByteArray(h * w)
            channelMat.get(0, 0, channelData)

            for (y in 0 until h) {
                for (x in 0 until w) {
                    chwArray[i * h * w + y * w + x] = (channelData[y * w + x].toInt() and 0xFF) / 255.0f
                }
            }
        }

        return chwArray
    }

    /**
     * Reorders CHW image data into HWC image data.
     *
     * @param outputData The output data from the model
     * @param width The width of the image
     * @param height The height of the image
     * @return A byte array in HWC format
     */
    private fun reorderToHWC(outputData: Array<*>, width: Int, height: Int): ByteArray {
        val size = width * height
        val hwcData = ByteArray(height * width * 3)

        for (h in 0 until height) {
            for (w in 0 until width) {
                for (c in 0 until 3) {
                    val chwIndex = c * size + h * width + w
                    val pixelVal = (outputData[0] as Array<*>)[c] as Array<*>
                    val value = (pixelVal[h * width + w] as Float) * 255.0f

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

        return hwcData
    }

    /**
     * Releases resources used by the inference session.
     */
    fun close() {
        miGanSession?.close()
        ortEnvironment.close()
    }
}
