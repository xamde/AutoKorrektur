package de.konradvoelkel.android.autokorrektur.ml

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.IOException
import java.nio.ByteBuffer

/**
 * Handles Mi-GAN model inference for inpainting.
 * Equivalent to miGanInference.js in the web app.
 */
class MiGanInference(private val context: Context) : InpaintingEngine {

    private val ortEnvironment = OrtEnvironment.getEnvironment()

    @Volatile
    private var miGanSession: OrtSession? = null
    private val lock = Any()

    companion object {
        private const val MODEL_INPUT_SIZE = 512
        private const val MODEL_FILE = "mi-gan-512.onnx"
        @Volatile
        private var sharedSession: OrtSession? = null
        private val globalLock = Any()
    }

    /**
     * Initializes the Mi-GAN model session.
     *
     * @throws IOException If the model file cannot be loaded
     */
    @Throws(IOException::class)
    override suspend fun initialize() = withContext(kotlinx.coroutines.Dispatchers.IO) {
        synchronized(globalLock) {
            if (sharedSession != null) {
                miGanSession = sharedSession
                return@withContext
            }

            AppLogger.debug("Initializing MiGanInference...")

            val modelBytes = try {
                context.assets.open("model/$MODEL_FILE").use { it.readBytes() }
            } catch (e: IOException) {
                AppLogger.debug("Failed to load Mi-GAN model: ${e.message}")
                throw IOException(
                    "Failed to load Mi-GAN model 'model/$MODEL_FILE': ${e.message}",
                    e
                )
            }

            val session = try {
                if (de.konradvoelkel.android.autokorrektur.utils.DevicePerformanceHelper.isNnapiSupported()) {
                    try {
                        val sessionOptions = OrtSession.SessionOptions().apply { addNnapi() }
                        AppLogger.info("MiGanInference: Attempting NNAPI EP...")
                        ortEnvironment.createSession(modelBytes, sessionOptions)
                    } catch (e: Exception) {
                        AppLogger.warn("MiGanInference: Failed to initialize NNAPI EP, falling back to CPU: ${e.message}")
                        ortEnvironment.createSession(modelBytes)
                    }
                } else {
                    ortEnvironment.createSession(modelBytes)
                }
            } catch (e: Exception) {
                AppLogger.debug("Failed to create Mi-GAN session: ${e.message}")
                throw IOException("Failed to create Mi-GAN session: ${e.message}", e)
            }

            sharedSession = session
            miGanSession = session
            AppLogger.info("MiGanInference session inputInfo: ${session.inputInfo}")
            AppLogger.info("MiGanInference session outputInfo: ${session.outputInfo}")
            AppLogger.debug("MiGanInference initialized successfully.")
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
    override suspend fun inpaint(imageMat: Mat, maskMat: Mat): Mat =
        withContext(Dispatchers.Default) {
            val session = miGanSession ?: run {
                initialize()
                miGanSession ?: throw IllegalStateException("Failed to initialize Mi-GAN session")
            }

            val matsToRelease = mutableListOf<Mat>()
            val tensorsToClose = mutableListOf<OnnxTensor>()

            try {
                val processedImage = preprocessImage(imageMat).also { matsToRelease.add(it) }
                val processedMask =
                    preprocessMask(maskMat, processedImage).also { matsToRelease.add(it) }

                val origWidth = processedImage.cols()
                val origHeight = processedImage.rows()
                val maxSize = kotlin.math.max(origWidth, origHeight)

                // 1. Prepare square inputs
                val (resizedImage, resizedMask) = prepareSquareInputs(
                    processedImage,
                    processedMask,
                    maxSize,
                    matsToRelease
                )

                // 2. Run ONNX Session
                val outputHWC = synchronized(lock) {
                    runOnnxSession(session, resizedImage, resizedMask, tensorsToClose, matsToRelease)
                }

                // 3. Process output Mat
                val unpaddedInpaintedMat =
                    processOutputMat(outputHWC, maxSize, origWidth, origHeight, matsToRelease)

                try {
                    // 4. Blend result
                    return@withContext blendResult(
                        processedImage,
                        processedMask,
                        unpaddedInpaintedMat
                    )
                } finally {
                    unpaddedInpaintedMat.release()
                }
            } finally {
                matsToRelease.forEach { it.release() }
                tensorsToClose.forEach { it.close() }
            }
        }

    /**
     * Pads image and subtractive mask to a square aspect ratio and resizes them to 512x512 for MI-GAN.
     */
    private fun prepareSquareInputs(
        processedImage: Mat,
        processedMask: Mat,
        maxSize: Int,
        matsToRelease: MutableList<Mat>
    ): Pair<Mat, Mat> {
        val xPad = maxSize - processedImage.cols()
        val yPad = maxSize - processedImage.rows()

        val squareImage = Mat().also { matsToRelease.add(it) }
        Core.copyMakeBorder(processedImage, squareImage, 0, yPad, 0, xPad, Core.BORDER_REFLECT_101)

        val squareMask = Mat().also { matsToRelease.add(it) }
        // Pad mask with 255.0 (Preserved Background) so non-image padding is never inpainted
        Core.copyMakeBorder(processedMask, squareMask, 0, yPad, 0, xPad, Core.BORDER_CONSTANT, org.opencv.core.Scalar(255.0))

        val modelInputSize = Size(MODEL_INPUT_SIZE.toDouble(), MODEL_INPUT_SIZE.toDouble())
        val resizedImage = Mat().also { matsToRelease.add(it) }
        Imgproc.resize(squareImage, resizedImage, modelInputSize)
        val resizedMask = Mat().also { matsToRelease.add(it) }
        Imgproc.resize(squareMask, resizedMask, modelInputSize, 0.0, 0.0, Imgproc.INTER_NEAREST)

        return Pair(resizedImage, resizedMask)
    }

    /**
     * Executes the ONNX Runtime session for MI-GAN inpainting and returns raw HWC byte buffer.
     */
    private fun runOnnxSession(
        session: OrtSession,
        resizedImage: Mat,
        resizedMask: Mat,
        tensorsToClose: MutableList<OnnxTensor>,
        matsToRelease: MutableList<Mat>
    ): ByteArray {
        // resizedMask has 0 on vehicle hole and 255 on preserved background, matching MI-GAN input convention.
        val imageArray = orderInCHWAsBytes(resizedImage)
        val maskArray = orderInCHWAsBytes(resizedMask)

        val imageTensor =
            createTensor(imageArray, 1, 3, MODEL_INPUT_SIZE.toLong(), MODEL_INPUT_SIZE.toLong())
                .also { tensorsToClose.add(it) }
        val maskTensor =
            createTensor(maskArray, 1, 1, MODEL_INPUT_SIZE.toLong(), MODEL_INPUT_SIZE.toLong())
                .also { tensorsToClose.add(it) }

        val zeroHoleCount = maskArray.count { it == 0.toByte() }
        AppLogger.info("MiGanInference: maskArray zeroHoleCount = $zeroHoleCount / ${maskArray.size}")

        val inputs = mapOf("image" to imageTensor, "mask" to maskTensor)

        return session.run(inputs).use { outputs ->
            val outputTensor = outputs[0]
            val outputData = getOutputData(outputTensor)
            AppLogger.info("MiGanInference: outputTensor class = ${outputTensor.javaClass.name}, outputData class = ${outputData?.javaClass?.name}")
            val hwc = reorderToHWC(outputData, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
            AppLogger.info("MiGanInference: hwc nonZeroCount = ${hwc.count { it != 0.toByte() }} / ${hwc.size}")
            hwc
        }
    }

    /**
     * Unpads and resizes the raw 512x512 MI-GAN output back to the original image dimensions.
     */
    private fun processOutputMat(
        outputHWC: ByteArray,
        maxSize: Int,
        origWidth: Int,
        origHeight: Int,
        matsToRelease: MutableList<Mat>
    ): Mat {
        val modelOutputMat =
            Mat(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, CvType.CV_8UC3).also { matsToRelease.add(it) }
        modelOutputMat.put(0, 0, outputHWC)

        val squareResultMat = Mat().also { matsToRelease.add(it) }
        Imgproc.resize(
            modelOutputMat,
            squareResultMat,
            Size(maxSize.toDouble(), maxSize.toDouble())
        )

        val roi = Rect(0, 0, origWidth, origHeight)
        return Mat(squareResultMat, roi).clone()
    }

    /**
     * Blends inpainted vehicle patches strictly into mask regions while preserving untouched background pixels.
     */
    private fun blendResult(processedImage: Mat, processedMask: Mat, unpaddedInpainted: Mat): Mat {
        val finalBlendedMat = processedImage.clone()
        // processedMask has 0 on car and 255 on background.
        // Invert so carMask has 255 on car, copying inpainting strictly onto the vehicle region.
        val carMask = Mat()
        Core.bitwise_not(processedMask, carMask)
        unpaddedInpainted.copyTo(finalBlendedMat, carMask)
        carMask.release()
        return finalBlendedMat
    }

    /**
     * Normalizes depth and color channels of the input image Mat to standard 8-bit RGB format.
     */
    private fun preprocessImage(imageMat: Mat): Mat {
        val converted = Mat()
        var current = imageMat
        var rgb: Mat? = null
        try {
            if (current.depth() != CvType.CV_8U) {
                val scale = if (current.depth() == CvType.CV_32F || current.depth() == CvType.CV_64F) 255.0 else 1.0
                current.convertTo(converted, CvType.CV_8U, scale)
                current = converted
            }
            rgb = Mat()
            when (current.channels()) {
                3 -> {
                    if (current !== imageMat) {
                        current.copyTo(rgb)
                    } else {
                        return imageMat.clone()
                    }
                }
                4 -> {
                    Imgproc.cvtColor(current, rgb, Imgproc.COLOR_RGBA2RGB)
                }
                1 -> {
                    Imgproc.cvtColor(current, rgb, Imgproc.COLOR_GRAY2RGB)
                }
                else -> {
                    val channels = mutableListOf<Mat>()
                    try {
                        Core.split(current, channels)
                        if (channels.size >= 3) {
                            val rgb3 = listOf(channels[0], channels[1], channels[2])
                            Core.merge(rgb3, rgb)
                        } else if (channels.isNotEmpty()) {
                            Imgproc.cvtColor(channels[0], rgb, Imgproc.COLOR_GRAY2RGB)
                        }
                    } finally {
                        channels.forEach { it.release() }
                    }
                }
            }
            val result = rgb
            rgb = null
            return result
        } finally {
            rgb?.release()
            if (converted !== imageMat) {
                converted.release()
            }
        }
    }

    /**
     * Converts mask to single-channel 8UC1 and ensures spatial dimensions match the source image.
     */
    private fun preprocessMask(maskMat: Mat, imageMat: Mat): Mat {
        val processedMask = Mat()
        if (maskMat.channels() == 1 && maskMat.type() == CvType.CV_8UC1) {
            maskMat.copyTo(processedMask)
        } else {
            val gray = when (maskMat.channels()) {
                1 -> maskMat
                else -> {
                    val g = Mat()
                    val code = if (maskMat.channels() == 3) Imgproc.COLOR_RGB2GRAY else Imgproc.COLOR_RGBA2GRAY
                    Imgproc.cvtColor(maskMat, g, code)
                    g
                }
            }
            val scale = if (gray.type() == CvType.CV_32F || gray.type() == CvType.CV_32FC1) 255.0 else 1.0
            gray.convertTo(processedMask, CvType.CV_8U, scale)
            if (gray != maskMat) {
                gray.release()
            }
        }

        if (processedMask.rows() != imageMat.rows() || processedMask.cols() != imageMat.cols()) {
            val resizedMask = Mat()
            Imgproc.resize(
                processedMask,
                resizedMask,
                imageMat.size(),
                0.0,
                0.0,
                Imgproc.INTER_NEAREST
            )
            processedMask.release()
            return resizedMask
        }

        return processedMask
    }

    /**
     * Creates a direct native ByteBuffer OnnxTensor with UINT8 payload and given tensor shape.
     */
    private fun createTensor(
        data: ByteArray,
        batchSize: Long,
        channels: Long,
        height: Long,
        width: Long
    ): OnnxTensor {
        val shape = longArrayOf(batchSize, channels, height, width)
        val directBuffer = java.nio.ByteBuffer.allocateDirect(data.size).order(java.nio.ByteOrder.nativeOrder())
        directBuffer.put(data)
        directBuffer.rewind()
        return OnnxTensor.createTensor(
            ortEnvironment,
            directBuffer,
            shape,
            OnnxJavaType.UINT8
        )
    }

    /**
     * Safely retrieves the byte or value payload from an output OnnxValue.
     */
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
     * Extracts and flattens image or mask channels into Planar CHW Byte order for ONNX tensor ingestion.
     */
    private fun orderInCHWAsBytes(mat: Mat): ByteArray {
        val c = mat.channels()
        val h = mat.rows()
        val w = mat.cols()
        val chwArray = ByteArray(c * h * w)

        if (c == 1) {
            mat.get(0, 0, chwArray)
            return chwArray
        }

        val planeSize = h * w
        for (i in 0 until c) {
            val channelMat = Mat()
            Core.extractChannel(mat, channelMat, i)
            val channelData = ByteArray(planeSize)
            channelMat.get(0, 0, channelData)
            channelMat.release()
            System.arraycopy(channelData, 0, chwArray, i * planeSize, planeSize)
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
                    val value =
                        if (chwIndex < floatData.size) floatData[chwIndex] * 255.0f else 0.0f
                    hwcData[(h * width + w) * 3 + c] = value.coerceIn(0f, 255f).toInt().toByte()
                }
            }
        }
    }

    /**
     * Tries to extract flat data from multi-dimensional nested array structure.
     */
    private fun tryExtractFlatData(outputData: Any?): Any? {
        if (outputData == null) return null
        if (outputData is ByteArray) return outputData
        if (outputData is FloatArray) return outputData
        if (outputData is Array<*>) {
            val byteList = mutableListOf<Byte>()
            val floatList = mutableListOf<Float>()
            var isFloat = false

            fun flatten(item: Any?) {
                when (item) {
                    is ByteArray -> item.forEach { byteList.add(it) }
                    is FloatArray -> {
                        isFloat = true
                        item.forEach { floatList.add(it) }
                    }
                    is Array<*> -> item.forEach { flatten(it) }
                }
            }
            outputData.forEach { flatten(it) }
            return if (isFloat) floatList.toFloatArray() else byteList.toByteArray()
        }
        return null
    }

    /**
     * Releases resources used by the inference session.
     */
    override fun close() {
        synchronized(globalLock) {
            try {
                miGanSession?.close()
            } catch (e: Exception) {
                AppLogger.warn("MiGanInference: Exception while closing OrtSession: ${e.message}")
            } finally {
                miGanSession = null
                sharedSession = null
                AppLogger.debug("MiGanInference: Released ONNX session.")
            }
        }
    }
}
