package de.konradvoelkel.android.autokorrektur.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import de.konradvoelkel.android.autokorrektur.utils.DevicePerformanceHelper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Mat
import androidx.core.graphics.createBitmap
import de.konradvoelkel.android.autokorrektur.ml.api.ServerSdxlApi

data class PipelineResult(
    val originalBitmap: Bitmap,
    val maskBitmap: Bitmap,
    val inpaintedBitmap: Bitmap?,
    val isServerProcessed: Boolean = false,
    val errorMessage: String? = null
)

class StaticImagePipeline(private val context: Context) {
    private val imageProcessor = ImageProcessor(context)
    private val yoloInference: YoloService = YoloServiceImpl(context)
    private val miGanInference = MiGanInference(context)
    private val serverSdxlApi = ServerSdxlApi(context)
    
    val isInitialized: Boolean
        get() = yoloInference.isInitialized
    
    suspend fun initialize() = withContext(Dispatchers.Default) {
        val modelName = "yolo11s"
        AppLogger.info("StaticImagePipeline initializing with model: $modelName")
        
        yoloInference.initialize(modelName = modelName, useFP16 = false)
        miGanInference.initialize()
    }

    suspend fun processImage(
        uri: Uri,
        downscaleMp: Float?,
        maskUpscale: Float,
        scoreThreshold: Float,
        useServerSdxl: Boolean,
        onMaskGenerated: ((Bitmap) -> Unit)? = null
    ): PipelineResult = withContext(Dispatchers.Default) {
        if (!isInitialized) {
            AppLogger.info("StaticImagePipeline processImage called on uninitialized pipeline; auto-initializing now...")
            initialize()
        }

        var processedImage: ImageProcessor.ProcessedImage? = null
        var maskMat: Mat? = null
        try {
            // 1. Process Input
            processedImage = imageProcessor.processInputImage(
                imageUri = uri,
                modelWidth = 640,
                modelHeight = 640,
                downscaleMp = downscaleMp
            )
            
            // 2. YOLO Mask Generation
            val config = YoloConfig(scoreThreshold = scoreThreshold)
            val yoloResult = yoloInference.inferDetailed(
                transformedMat = processedImage.transformedMat,
                xRatio = processedImage.xRatio,
                yRatio = processedImage.yRatio,
                upscaleFactor = maskUpscale,
                originalWidth = processedImage.originalMat.cols(),
                originalHeight = processedImage.originalMat.rows(),
                overrideConfig = config
            )
            maskMat = yoloResult.mask
            
            // Convert mask to Bitmap
            val maskBitmap = createBitmap(maskMat.cols(), maskMat.rows())
            Utils.matToBitmap(maskMat, maskBitmap)
            
            onMaskGenerated?.invoke(maskBitmap)

            
            // 3. Inpainting
            val inpaintedBitmap: Bitmap?
            if (useServerSdxl) {
                AppLogger.info("Using Server SDXL for inpainting")
                // Generate a preview with MiGan first to use as structural prior
                val miGanPreview = miGanInference.inferMiGan(processedImage.originalMat, maskMat)
                val previewBitmap = createBitmap(miGanPreview.cols(), miGanPreview.rows())
                Utils.matToBitmap(miGanPreview, previewBitmap)
                miGanPreview.release()
                
                // Send to server
                inpaintedBitmap = serverSdxlApi.processWithSdxl(processedImage.originalBitmap, maskBitmap, previewBitmap)
            } else {
                AppLogger.info("Using Local Mi-GAN for inpainting")
                val miGanResult = miGanInference.inferMiGan(processedImage.originalMat, maskMat)
                inpaintedBitmap = createBitmap(miGanResult.cols(), miGanResult.rows())
                Utils.matToBitmap(miGanResult, inpaintedBitmap)
                miGanResult.release()
            }
            
            return@withContext PipelineResult(
                originalBitmap = processedImage.originalBitmap,
                maskBitmap = maskBitmap,
                inpaintedBitmap = inpaintedBitmap,
                isServerProcessed = useServerSdxl
            )
            
        } catch (e: Exception) {
            AppLogger.error("StaticImagePipeline Error", e)
            return@withContext PipelineResult(
                originalBitmap = processedImage?.originalBitmap ?: createBitmap(1, 1),
                maskBitmap = createBitmap(1, 1),
                inpaintedBitmap = null,
                errorMessage = e.message
            )
        } finally {
            processedImage?.release()
            maskMat?.release()
        }
    }
    
    fun close() {
        yoloInference.close()
        miGanInference.close()
    }
}
