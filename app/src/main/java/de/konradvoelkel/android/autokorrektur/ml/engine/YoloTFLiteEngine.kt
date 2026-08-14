package de.konradvoelkel.android.autokorrektur.ml.engine

import android.content.Context
import android.content.pm.ApplicationInfo
import de.konradvoelkel.android.autokorrektur.ml.errors.InferenceException
import de.konradvoelkel.android.autokorrektur.ml.errors.ModelLoadException
import de.konradvoelkel.android.autokorrektur.ml.errors.ShapeMismatchException
import de.konradvoelkel.android.autokorrektur.ml.model.RawOutputs
import de.konradvoelkel.android.autokorrektur.ml.model.Shapes
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * TFLite engine that owns the Interpreter, shapes, and reusable I/O buffers.
 * It accepts a preprocessed RGB Mat (HxWx3, 8UC3) matching the model input size.
 */
class YoloTFLiteEngine(private val context: Context) : YoloEngine {

    private val isDebugBuild: Boolean by lazy {
        (context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0
    }

    @Volatile
    private var interpreter: Interpreter? = null

    // Discovered shapes
    private var inputW: Int = 640
    private var inputH: Int = 640
    private var inputC: Int = 3
    private var detShape: IntArray = intArrayOf()
    private var protoShape: IntArray = intArrayOf()

    // Reusable buffers
    private var inputBuffer: ByteBuffer? = null
    private var outputDetections: ByteBuffer? = null
    private var outputPrototypes: ByteBuffer? = null

    // Synchronization lock for lifecycle and run serialization
    private val lock = Any()

    override val isInitialized: Boolean
        get() = interpreter != null

    override val isClosed: Boolean
        get() = interpreter == null

    @Throws(ModelLoadException::class)
    override suspend fun initialize(modelName: String, useFP16: Boolean) =
        withContext(Dispatchers.IO) {
        synchronized(lock) {
            if (isInitialized) {
                AppLogger.debug("YoloTFLiteEngine already initialized")
                return@withContext
            }
            val modelFile = if (useFP16) {
                "model/${modelName}-seg_saved_model/${modelName}-seg_float16.tflite"
            } else {
                "model/${modelName}-seg_saved_model/${modelName}-seg_float32.tflite"
            }
            try {
                val modelBytes = try {
                    context.assets.openFd(modelFile).use { afd ->
                        afd.createInputStream().use { inputStream ->
                            inputStream.readBytes()
                        }
                    }
                } catch (e: Exception) {
                    context.assets.open(modelFile).use { inputStream ->
                        inputStream.readBytes()
                    }
                }

                val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size)
                modelBuffer.order(ByteOrder.nativeOrder())
                modelBuffer.put(modelBytes)
                modelBuffer.rewind()

                val threads = if (isDebugBuild) 2 else Runtime.getRuntime().availableProcessors()

                interpreter = try {
                    if (de.konradvoelkel.android.autokorrektur.utils.DevicePerformanceHelper.isNnapiSupported()) {
                        val nnapiOptions = Interpreter.Options().apply {
                            setNumThreads(threads)
                            setUseNNAPI(true)
                        }
                        AppLogger.info("YoloTFLiteEngine: Attempting NNAPI acceleration...")
                        Interpreter(modelBuffer, nnapiOptions)
                    } else {
                        null
                    }
                } catch (e: Exception) {
                    AppLogger.warn("YoloTFLiteEngine: NNAPI delegate initialization failed, falling back to CPU: ${e.message}")
                    null
                } ?: run {
                    val cpuOptions = Interpreter.Options().apply {
                        setNumThreads(threads)
                    }
                    AppLogger.info("YoloTFLiteEngine: Initializing CPU interpreter")
                    Interpreter(modelBuffer, cpuOptions)
                }

                // Input shape [1, H, W, C]
                val inTensor = interpreter!!.getInputTensor(0)
                val inShape = inTensor.shape()
                if (inShape.size != 4) throw ShapeMismatchException("Unexpected input shape: ${inShape.joinToString()}")
                inputH = inShape[1]
                inputW = inShape[2]
                inputC = inShape[3]
                AppLogger.debug("Engine input shape: [${inShape.joinToString()}]")

                // Output shapes
                detShape = interpreter!!.getOutputTensor(0).shape()
                protoShape = interpreter!!.getOutputTensor(1).shape()
                AppLogger.debug("Engine output shapes: det=${detShape.joinToString()}, proto=${protoShape.joinToString()}")

                allocateBuffers()
                AppLogger.debug("YoloTFLiteEngine initialized. Threads=$threads")
            } catch (e: Exception) {
                throw ModelLoadException("Failed to initialize TFLite YOLO model: ${e.message}", e)
            }
        }
    }

    private fun allocateBuffers() {
        val inBytes = inputH * inputW * inputC * 4 // float32
        if (inputBuffer == null || inputBuffer!!.capacity() < inBytes) {
            inputBuffer = ByteBuffer.allocateDirect(inBytes).order(ByteOrder.nativeOrder())
        }
        // Assume float32 outputs
        val detFloats = detShape.fold(1) { acc, v -> acc * v }
        val protoFloats = protoShape.fold(1) { acc, v -> acc * v }
        val detBytes = detFloats * 4
        val protoBytes = protoFloats * 4
        if (outputDetections == null || outputDetections!!.capacity() < detBytes) {
            outputDetections = ByteBuffer.allocateDirect(detBytes).order(ByteOrder.nativeOrder())
        }
        if (outputPrototypes == null || outputPrototypes!!.capacity() < protoBytes) {
            outputPrototypes = ByteBuffer.allocateDirect(protoBytes).order(ByteOrder.nativeOrder())
        }
        // Reset positions
        inputBuffer!!.rewind()
        outputDetections!!.rewind()
        outputPrototypes!!.rewind()
    }

    override fun shapes(): Shapes = Shapes(
        inputH = inputH,
        inputW = inputW,
        inputC = inputC,
        detShape = detShape,
        protoShape = protoShape
    )

    /**
     * Run inference for a preprocessed RGB Mat sized to inputW x inputH (8UC3).
     */
    @Throws(InferenceException::class)
    override fun run(rgbMat: Mat): RawOutputs {
        synchronized(lock) {
            try {
                if (!isInitialized) {
                    throw InferenceException("YoloTFLiteEngine.run() called before initialize()")
                }

                if (inputBuffer == null || outputDetections == null || outputPrototypes == null) {
                    // Allocate or reallocate buffers if needed (defensive)
                    allocateBuffers()
                }

                if (rgbMat.rows() != inputH || rgbMat.cols() != inputW) {
                    // Resize into a temporary Mat to match model input
                    val resized = Mat()
                    Imgproc.resize(
                        rgbMat,
                        resized,
                        org.opencv.core.Size(inputW.toDouble(), inputH.toDouble())
                    )
                    try {
                        matToByteBuffer(resized, inputBuffer!!)
                    } finally {
                        resized.release()
                    }
                } else {
                    matToByteBuffer(rgbMat, inputBuffer!!)
                }

                val inputs = arrayOf<Any>(inputBuffer as Any)
                val outputs = HashMap<Int, Any>(2)
                // Ensure output buffers are rewound
                outputDetections!!.rewind()
                outputPrototypes!!.rewind()
                outputs[0] = outputDetections as Any
                outputs[1] = outputPrototypes as Any

                interpreter!!.runForMultipleInputsOutputs(inputs, outputs)

                // B6: Return real copies to avoid aliasing with internal reusable buffers
                val detResult = ByteBuffer.allocateDirect(outputDetections!!.capacity())
                    .order(ByteOrder.nativeOrder())
                outputDetections!!.rewind()
                detResult.put(outputDetections!!)
                detResult.rewind()

                val protoResult = ByteBuffer.allocateDirect(outputPrototypes!!.capacity())
                    .order(ByteOrder.nativeOrder())
                outputPrototypes!!.rewind()
                protoResult.put(outputPrototypes!!)
                protoResult.rewind()

                return RawOutputs(detResult, protoResult, shapes())
            } catch (e: Exception) {
                val matType = try {
                    CvType.typeToString(rgbMat.type())
                } catch (_: Exception) {
                    "?"
                }
                val msg = buildString {
                    append("TFLite run failed: ")
                    append(e.message)
                    append(" | inputMat=")
                    append(rgbMat.rows()).append("x").append(rgbMat.cols()).append(", ")
                    append(matType)
                    append(" | expectedInput=")
                    append(inputH).append("x").append(inputW).append("x").append(inputC)
                    append(" | detShape=")
                    append(detShape.joinToString())
                    append(" | protoShape=")
                    append(protoShape.joinToString())
                }
                throw InferenceException(msg, e)
            }
        }
    }

    override fun close() {
        synchronized(lock) {
            try {
                interpreter?.close()
            } catch (_: Exception) {
            } finally {
                interpreter = null
                // Release buffers and reset discovered shapes to safe defaults
                inputBuffer = null
                outputDetections = null
                outputPrototypes = null
                detShape = intArrayOf()
                protoShape = intArrayOf()
            }
        }
    }

    /** Converts an 8UC3 RGB Mat into a float32 NHWC buffer scaled to 0..1. */
    private fun matToByteBuffer(mat: Mat, buffer: ByteBuffer) {
        var input = mat
        val matsToRelease = mutableListOf<Mat>()
        try {
            if (input.type() != CvType.CV_8UC3) {
                val tmp = Mat().also { matsToRelease.add(it) }
                if (input.channels() == 4) {
                    Imgproc.cvtColor(input, tmp, Imgproc.COLOR_RGBA2RGB)
                } else if (input.channels() == 1) {
                    Imgproc.cvtColor(input, tmp, Imgproc.COLOR_GRAY2RGB)
                } else {
                    val scale = if (input.depth() == CvType.CV_32F) 255.0 else 1.0
                    input.convertTo(tmp, CvType.CV_8U, scale)
                }
                input = tmp
            }
            val rows = input.rows()
            val cols = input.cols()
            val channels = input.channels()
            if (channels != 3) throw ShapeMismatchException("Expected 3 channels, got $channels")

            // Copy pixels
            val totalBytes = rows * cols * channels
            val pixels = ByteArray(totalBytes)
            val reshaped = input.reshape(1, totalBytes)
            reshaped.get(0, 0, pixels)
            reshaped.release()
            buffer.rewind()
            // Write normalized floats in NHWC order
            var idx = 0
            val scale = 1f / 255f
            repeat(rows) {
                repeat(cols) {
                    val r = (pixels[idx].toInt() and 0xFF) * scale
                    val g = (pixels[idx + 1].toInt() and 0xFF) * scale
                    val b = (pixels[idx + 2].toInt() and 0xFF) * scale
                    buffer.putFloat(r)
                    buffer.putFloat(g)
                    buffer.putFloat(b)
                    idx += 3
                }
            }
            buffer.rewind()
        } finally {
            matsToRelease.forEach { it.release() }
        }
    }
}