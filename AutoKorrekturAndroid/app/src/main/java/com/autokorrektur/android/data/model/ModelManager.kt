package com.autokorrektur.android.data.model

import android.content.Context
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import timber.log.Timber
import java.io.InputStream
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ModelManager @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private var ortEnvironment: OrtEnvironment? = null
    private var yoloSession: OrtSession? = null
    private var nmsSession: OrtSession? = null
    private var maskSession: OrtSession? = null
    private var miGanSession: OrtSession? = null

    private var isInitialized = false

    companion object {
        private const val YOLO_MODEL_SMALL = "yolo11s-seg.onnx"
        private const val YOLO_MODEL_NANO = "yolo11n-seg.onnx"
        private const val YOLO_MODEL_MEDIUM = "yolo11m-seg.onnx"
        private const val NMS_MODEL = "nms-yolov8.onnx"
        private const val MASK_MODEL = "mask-yolov8-seg.onnx"
        private const val MIGAN_MODEL = "mi-gan-512.onnx"
    }

    enum class YoloModelSize {
        NANO, SMALL, MEDIUM
    }

    suspend fun initializeModels(yoloModelSize: YoloModelSize = YoloModelSize.SMALL) = withContext(Dispatchers.IO) {
        try {
            Timber.d("Initializing ONNX Runtime models...")

            // Initialize ONNX Runtime environment
            ortEnvironment = OrtEnvironment.getEnvironment()

            val yoloModelName = when (yoloModelSize) {
                YoloModelSize.NANO -> YOLO_MODEL_NANO
                YoloModelSize.SMALL -> YOLO_MODEL_SMALL
                YoloModelSize.MEDIUM -> YOLO_MODEL_MEDIUM
            }

            // Load YOLO segmentation model
            yoloSession = createSessionFromAsset(yoloModelName)
            Timber.d("YOLO model loaded: $yoloModelName")

            // Load NMS model
            nmsSession = createSessionFromAsset(NMS_MODEL)
            Timber.d("NMS model loaded")

            // Load mask processing model
            maskSession = createSessionFromAsset(MASK_MODEL)
            Timber.d("Mask model loaded")

            // Load MI-GAN inpainting model
            miGanSession = createSessionFromAsset(MIGAN_MODEL)
            Timber.d("MI-GAN model loaded")

            isInitialized = true
            Timber.d("All models initialized successfully")

        } catch (e: Exception) {
            Timber.e(e, "Failed to initialize models")
            cleanup()
            throw e
        }
    }

    private fun createSessionFromAsset(modelFileName: String): OrtSession {
        val inputStream: InputStream = context.assets.open(modelFileName)
        val modelBytes = inputStream.readBytes()
        inputStream.close()

        return ortEnvironment?.createSession(modelBytes)
            ?: throw IllegalStateException("ORT Environment not initialized")
    }

    fun getYoloSession(): OrtSession {
        return yoloSession ?: throw IllegalStateException("YOLO model not initialized")
    }

    fun getNmsSession(): OrtSession {
        return nmsSession ?: throw IllegalStateException("NMS model not initialized")
    }

    fun getMaskSession(): OrtSession {
        return maskSession ?: throw IllegalStateException("Mask model not initialized")
    }

    fun getMiGanSession(): OrtSession {
        return miGanSession ?: throw IllegalStateException("MI-GAN model not initialized")
    }

    fun isModelsInitialized(): Boolean = isInitialized

    suspend fun switchYoloModel(yoloModelSize: YoloModelSize) = withContext(Dispatchers.IO) {
        try {
            Timber.d("Switching YOLO model to: $yoloModelSize")

            // Close current YOLO session
            yoloSession?.close()

            val yoloModelName = when (yoloModelSize) {
                YoloModelSize.NANO -> YOLO_MODEL_NANO
                YoloModelSize.SMALL -> YOLO_MODEL_SMALL
                YoloModelSize.MEDIUM -> YOLO_MODEL_MEDIUM
            }

            // Load new YOLO model
            yoloSession = createSessionFromAsset(yoloModelName)
            Timber.d("YOLO model switched to: $yoloModelName")

        } catch (e: Exception) {
            Timber.e(e, "Failed to switch YOLO model")
            throw e
        }
    }

    fun cleanup() {
        try {
            yoloSession?.close()
            nmsSession?.close()
            maskSession?.close()
            miGanSession?.close()
            ortEnvironment?.close()

            yoloSession = null
            nmsSession = null
            maskSession = null
            miGanSession = null
            ortEnvironment = null
            isInitialized = false

            Timber.d("Model manager cleaned up")
        } catch (e: Exception) {
            Timber.e(e, "Error during cleanup")
        }
    }

    fun getModelInfo(): Map<String, Any> {
        return mapOf(
            "isInitialized" to isInitialized,
            "yoloInputs" to (yoloSession?.inputNames?.toList() ?: emptyList()),
            "yoloOutputs" to (yoloSession?.outputNames?.toList() ?: emptyList()),
            "miGanInputs" to (miGanSession?.inputNames?.toList() ?: emptyList()),
            "miGanOutputs" to (miGanSession?.outputNames?.toList() ?: emptyList())
        )
    }
}
