package de.konradvoelkel.android.autokorrektur.ml.engine

import android.content.Context
import android.content.pm.ApplicationInfo
import de.konradvoelkel.android.autokorrektur.ml.errors.InferenceException
import de.konradvoelkel.android.autokorrektur.ml.errors.ModelLoadException
import de.konradvoelkel.android.autokorrektur.ml.errors.ShapeMismatchException
import de.konradvoelkel.android.autokorrektur.ml.model.RawOutputs
import de.konradvoelkel.android.autokorrektur.ml.model.Shapes
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
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
class YoloTFLiteEngine(private val context: Context) {

    private val isDebugBuild: Boolean by lazy {
        (context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0
    }

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

    val isInitialized: Boolean
        get() = interpreter != null

    @Throws(ModelLoadException::class)
    fun initialize(modelName: String = "yolo11s", useFP16: Boolean = false) {
        if (isInitialized) {
            AppLogger.debug("YoloTFLiteEngine already initialized")
            return
        }
        val modelFile = if (useFP16) {
            "model/${modelName}-seg_saved_model/${modelName}-seg_float16.tflite"
        } else {
            "model/${modelName}-seg_saved_model/${modelName}-seg_float32.tflite"
        }
        try {
            val afd = context.assets.openFd(modelFile)
            val inputStream = afd.createInputStream()
            val modelBytes = inputStream.readBytes()
            inputStream.close()
            afd.close()

            val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size)
            modelBuffer.order(ByteOrder.nativeOrder())
            modelBuffer.put(modelBytes)
            modelBuffer.rewind()

            val options = Interpreter.Options()
            val threads = if (isDebugBuild) 2 else Runtime.getRuntime().availableProcessors()
            options.setNumThreads(threads)
            interpreter = Interpreter(modelBuffer, options)

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

    fun shapes(): Shapes = Shapes(
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
    fun run(rgbMat: Mat): RawOutputs {
        try {
            if (rgbMat.rows() != inputH || rgbMat.cols() != inputW) {
                // Resize into a temporary Mat to match model input
                val resized = Mat()
                Imgproc.resize(rgbMat, resized, org.opencv.core.Size(inputW.toDouble(), inputH.toDouble()))
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

            // Duplicate read-only views to return without disturbing internal positions
            val detCopy = outputDetections!!.duplicate().order(ByteOrder.nativeOrder())
            detCopy.rewind()
            val protoCopy = outputPrototypes!!.duplicate().order(ByteOrder.nativeOrder())
            protoCopy.rewind()
            return RawOutputs(detCopy, protoCopy, shapes())
        } catch (e: Exception) {
            throw InferenceException("TFLite run failed: ${e.message}", e)
        }
    }

    fun close() {
        try {
            interpreter?.close()
        } catch (_: Exception) {
        } finally {
            interpreter = null
        }
    }

    /** Converts an 8UC3 RGB Mat into a float32 NHWC buffer scaled to 0..1. */
    private fun matToByteBuffer(mat: Mat, buffer: ByteBuffer) {
        // Ensure type
        require(mat.type() == CvType.CV_8UC3) { "Expected CV_8UC3 input Mat (RGB)" }
        val rows = mat.rows()
        val cols = mat.cols()
        val channels = mat.channels()
        if (channels != 3) throw ShapeMismatchException("Expected 3 channels, got $channels")

        // Copy pixels
        val pixels = ByteArray(rows * cols * channels)
        mat.get(0, 0, pixels)
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
    }
}