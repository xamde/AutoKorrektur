package de.konradvoelkel.android.autokorrektur.video

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import de.konradvoelkel.android.autokorrektur.ar.TemporalBackgroundAccumulator
import de.konradvoelkel.android.autokorrektur.ml.InpaintingEngine
import de.konradvoelkel.android.autokorrektur.ml.MatScaler
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig
import de.konradvoelkel.android.autokorrektur.ml.preprocess.DefaultPreprocessor
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.io.File

/**
 * Result data from the HQ video inpainting process.
 */
data class VideoProcessingResult(
    val outputFile: File,
    val totalFrames: Int,
    val durationMs: Long,
    val width: Int,
    val height: Int
)

/**
 * High-Quality offline video post-processing engine.
 * Takes a raw video snippet captured in AR mode, processes each frame with YOLO vehicle detection,
 * temporal background accumulation, and neural inpainting, and encodes a stabilized 30 FPS car-free MP4.
 */
class VideoInpaintProcessor(
    private val yoloService: YoloService,
    private val inpaintingEngine: InpaintingEngine
) {
    /**
     * Processes a video file into an in-painted car-free MP4 video.
     *
     * @param inputFile Raw captured video snippet.
     * @param outputFile Output destination for the processed MP4.
     * @param maxFps Target frame rate to extract and encode (defaults to 15-30 FPS).
     * @param onProgress Progress reporter callback (stage, percent 0..100).
     */
    suspend fun processVideo(
        inputFile: File,
        outputFile: File,
        maxFps: Int = 15,
        onProgress: ((stage: String, percent: Int) -> Unit)? = null
    ): VideoProcessingResult = withContext(Dispatchers.Default) {
        val retriever = MediaMetadataRetriever()
        val accumulator = TemporalBackgroundAccumulator()
        val preprocessor = DefaultPreprocessor()

        try {
            retriever.setDataSource(inputFile.absolutePath)
            val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val durationMs = durationStr?.toLongOrNull() ?: 5000L
            val widthStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)
            val heightStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
            val rotationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)

            var rawW = widthStr?.toIntOrNull() ?: 1080
            var rawH = heightStr?.toIntOrNull() ?: 1920
            val rotation = rotationStr?.toIntOrNull() ?: 0

            if (rotation == 90 || rotation == 270) {
                val tmp = rawW
                rawW = rawH
                rawH = tmp
            }

            // Ensure dimensions are even (required by H.264 / AVC encoders)
            val targetW = (rawW / 2) * 2
            val targetH = (rawH / 2) * 2

            val frameIntervalMs = 1000L / maxFps
            val totalFrames = ((durationMs / frameIntervalMs).toInt()).coerceIn(1, 300)

            AppLogger.info("Starting VideoInpaintProcessor: $totalFrames frames, ${targetW}x${targetH} @ ${maxFps}fps")
            onProgress?.invoke("Preparing Video Encoder", 2)

            val encoder = VideoEncoder(
                width = targetW,
                height = targetH,
                frameRate = maxFps,
                bitRate = 8_000_000
            )
            encoder.start(outputFile)

            try {
                for (frameIdx in 0 until totalFrames) {
                    currentCoroutineContext().ensureActive()

                    val timeUs = (frameIdx * frameIntervalMs) * 1000L
                    val frameBitmap = retriever.getFrameAtTime(
                        timeUs,
                        MediaMetadataRetriever.OPTION_CLOSEST
                    ) ?: continue

                    // Scale to target dimensions if necessary
                    val scaledBitmap = if (frameBitmap.width != targetW || frameBitmap.height != targetH) {
                        val scaled = Bitmap.createScaledBitmap(frameBitmap, targetW, targetH, true)
                        if (scaled != frameBitmap) frameBitmap.recycle()
                        scaled
                    } else {
                        frameBitmap
                    }

                    val frameMat = Mat()
                    Utils.bitmapToMat(scaledBitmap, frameMat)

                    // 1. YOLO segmentation
                    val preprocessed = preprocessor.prepare(frameMat, 640, 640)
                    val yoloResult = yoloService.inferDetailed(
                        transformedMat = preprocessed.forEngine,
                        xRatio = preprocessed.xRatio,
                        yRatio = preprocessed.yRatio,
                        upscaleFactor = 1.05f,
                        originalWidth = targetW,
                        originalHeight = targetH,
                        overrideConfig = YoloConfig(scoreThreshold = 0.35f)
                    )

                    // 2. Temporal background stabilization + inpainting
                    val carFreeOverlay = accumulator.accumulateAndBlend(frameMat, yoloResult.mask)

                    // Composite into output frame
                    val cleanFrameMat = Mat()
                    frameMat.copyTo(cleanFrameMat)

                    // Blend carFreeOverlay onto cleanFrameMat where mask was active
                    val overlayChannels = mutableListOf<Mat>()
                    Core.split(carFreeOverlay, overlayChannels)
                    if (overlayChannels.size >= 4) {
                        val alphaMask = overlayChannels[3]
                        carFreeOverlay.copyTo(cleanFrameMat, alphaMask)
                        alphaMask.release()
                    }
                    overlayChannels.forEach { it.release() }

                    val outBmp = MatScaler.createDisplayBitmap(cleanFrameMat)
                    encoder.encodeFrame(outBmp)

                    // Memory cleanup
                    frameMat.release()
                    cleanFrameMat.release()
                    carFreeOverlay.release()
                    yoloResult.release()
                    preprocessed.release()
                    scaledBitmap.recycle()
                    outBmp.recycle()

                    val percent = (5 + ((frameIdx + 1) * 90 / totalFrames)).coerceAtMost(98)
                    onProgress?.invoke("Inpainting Frame ${frameIdx + 1}/$totalFrames", percent)
                }

                onProgress?.invoke("Finalizing MP4 Video", 99)
                encoder.finish()
            } finally {
                encoder.release()
            }

            onProgress?.invoke("Video Inpainting Complete", 100)
            return@withContext VideoProcessingResult(
                outputFile = outputFile,
                totalFrames = totalFrames,
                durationMs = durationMs,
                width = targetW,
                height = targetH
            )
        } finally {
            retriever.release()
            accumulator.close()
        }
    }
}
