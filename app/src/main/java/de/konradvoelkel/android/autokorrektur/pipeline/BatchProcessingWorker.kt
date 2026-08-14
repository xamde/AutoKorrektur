package de.konradvoelkel.android.autokorrektur.pipeline

import android.content.Context
import android.net.Uri
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import de.konradvoelkel.android.autokorrektur.ml.api.ServerSdxlApi
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import androidx.core.net.toUri

/**
 * Background WorkManager worker to process multiple queued vehicle photos sequentially.
 */
class BatchProcessingWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        val imageUrisStr = inputData.getStringArray(KEY_IMAGE_URIS)
        if (imageUrisStr.isNullOrEmpty()) {
            AppLogger.info("BatchProcessingWorker: No image URIs provided")
            return Result.failure(workDataOf(KEY_ERROR to "No image URIs provided"))
        }

        val useServerSdxl = inputData.getBoolean(KEY_USE_SERVER_SDXL, false)
        val scoreThreshold = inputData.getFloat(KEY_SCORE_THRESHOLD, 0.25f)
        val maskUpscale = inputData.getFloat(KEY_MASK_UPSCALE, 1.0f)
        val downscaleMp = if (inputData.keyValueMap.containsKey(KEY_DOWNSCALE_MP)) {
            inputData.getFloat(KEY_DOWNSCALE_MP, 2.0f)
        } else null

        AppLogger.info("BatchProcessingWorker starting batch for ${imageUrisStr.size} images")
        val pipeline = StaticImagePipeline(
            ImageProcessor(applicationContext),
            YoloServiceImpl(YoloTFLiteEngine(applicationContext)),
            MiGanInference(applicationContext),
            ServerSdxlApi(applicationContext)
        )

        return try {
            pipeline.initialize()
            var successCount = 0

            imageUrisStr.forEachIndexed { index, uriStr ->
                if (isStopped) {
                    AppLogger.info("BatchProcessingWorker cancelled at index $index")
                    return Result.failure(workDataOf(KEY_ERROR to "Worker cancelled"))
                }

                val progressPercent = ((index.toFloat() / imageUrisStr.size) * 100).toInt()
                setProgress(workDataOf(
                    KEY_PROGRESS_PERCENT to progressPercent,
                    KEY_CURRENT_INDEX to index,
                    KEY_TOTAL_COUNT to imageUrisStr.size
                ))

                val uri = uriStr.toUri()
                AppLogger.info("BatchProcessingWorker processing image ${index + 1}/${imageUrisStr.size}: $uri")

                val result = pipeline.processImage(
                    uri = uri,
                    downscaleMp = downscaleMp,
                    maskUpscale = maskUpscale,
                    scoreThreshold = scoreThreshold,
                    useServerSdxl = useServerSdxl
                )

                if (result.inpaintedBitmap != null && result.errorMessage == null) {
                    successCount++
                }
            }

            setProgress(workDataOf(
                KEY_PROGRESS_PERCENT to 100,
                KEY_CURRENT_INDEX to imageUrisStr.size,
                KEY_TOTAL_COUNT to imageUrisStr.size
            ))

            AppLogger.info("BatchProcessingWorker completed successfully: $successCount/${imageUrisStr.size} processed")
            Result.success(workDataOf(
                KEY_SUCCESS_COUNT to successCount,
                KEY_TOTAL_COUNT to imageUrisStr.size
            ))
        } catch (e: Exception) {
            AppLogger.error("BatchProcessingWorker failed", e)
            Result.failure(workDataOf(KEY_ERROR to (e.message ?: "Unknown batch error")))
        } finally {
            pipeline.close()
        }
    }

    companion object {
        const val KEY_IMAGE_URIS = "image_uris"
        const val KEY_USE_SERVER_SDXL = "use_server_sdxl"
        const val KEY_SCORE_THRESHOLD = "score_threshold"
        const val KEY_MASK_UPSCALE = "mask_upscale"
        const val KEY_DOWNSCALE_MP = "downscale_mp"

        const val KEY_PROGRESS_PERCENT = "progress_percent"
        const val KEY_CURRENT_INDEX = "current_index"
        const val KEY_TOTAL_COUNT = "total_count"
        const val KEY_SUCCESS_COUNT = "success_count"
        const val KEY_ERROR = "error_message"
    }
}
