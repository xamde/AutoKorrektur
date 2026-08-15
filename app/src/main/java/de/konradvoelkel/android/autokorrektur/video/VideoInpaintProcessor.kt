package de.konradvoelkel.android.autokorrektur.video

import android.graphics.Bitmap
import android.media.Image
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
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
import org.opencv.imgproc.Imgproc
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
        val extractor = MediaExtractor()

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

            extractor.setDataSource(inputFile.absolutePath)
            var videoTrackIndex = -1
            for (i in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME)
                if (mime?.startsWith("video/") == true) {
                    videoTrackIndex = i
                    break
                }
            }
            if (videoTrackIndex < 0) {
                throw IllegalStateException("No video track found")
            }

            extractor.selectTrack(videoTrackIndex)
            val format = extractor.getTrackFormat(videoTrackIndex)
            val mime = format.getString(MediaFormat.KEY_MIME)!!

            val codec = MediaCodec.createDecoderByType(mime)
            codec.configure(format, null, null, 0)
            codec.start()

            var decodedFrameCount = 0

            try {
                var isExtractorEOS = false
                var isDecoderEOS = false
                val info = MediaCodec.BufferInfo()
                var nextExpectedTimeUs = 0L
                val frameIntervalUs = frameIntervalMs * 1000L

                while (!isDecoderEOS && decodedFrameCount < totalFrames) {
                    currentCoroutineContext().ensureActive()

                    if (!isExtractorEOS) {
                        val inIndex = codec.dequeueInputBuffer(10000)
                        if (inIndex >= 0) {
                            val buffer = codec.getInputBuffer(inIndex)
                            val sampleSize = extractor.readSampleData(buffer!!, 0)
                            if (sampleSize < 0) {
                                codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                isExtractorEOS = true
                            } else {
                                codec.queueInputBuffer(inIndex, 0, sampleSize, extractor.sampleTime, 0)
                                extractor.advance()
                            }
                        }
                    }

                    val outIndex = codec.dequeueOutputBuffer(info, 10000)
                    when {
                        outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> { /* ignore */ }
                        outIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> { /* ignore */ }
                        outIndex >= 0 -> {
                            if ((info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                                isDecoderEOS = true
                            }

                            if (info.size > 0 && info.presentationTimeUs >= nextExpectedTimeUs - 10000L) {
                                val image = codec.getOutputImage(outIndex)
                                if (image != null) {
                                    val mat = imageToMat(image)
                                    val rotatedMat = de.konradvoelkel.android.autokorrektur.ar.ArFrameConverter.rotateMat(mat, rotation)

                                    // Scale to target dimensions if necessary
                                    val scaledMat = if (rotatedMat.cols() != targetW || rotatedMat.rows() != targetH) {
                                        val scaled = Mat()
                                        Imgproc.resize(rotatedMat, scaled, org.opencv.core.Size(targetW.toDouble(), targetH.toDouble()))
                                        scaled
                                    } else {
                                        rotatedMat
                                    }
                                    if (scaledMat !== rotatedMat) rotatedMat.release()
                                    mat.release()

                                    // 1. YOLO segmentation
                                    val preprocessed = preprocessor.prepare(scaledMat, 640, 640)
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
                                    val carFreeOverlay = accumulator.accumulateAndBlend(scaledMat, yoloResult.mask)

                                    // Composite into output frame
                                    val cleanFrameMat = Mat()
                                    scaledMat.copyTo(cleanFrameMat)

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
                                    scaledMat.release()
                                    cleanFrameMat.release()
                                    carFreeOverlay.release()
                                    yoloResult.release()
                                    preprocessed.release()
                                    outBmp.recycle()

                                    decodedFrameCount++
                                    nextExpectedTimeUs += frameIntervalUs

                                    val percent = (5 + (decodedFrameCount * 90 / totalFrames)).coerceAtMost(98)
                                    onProgress?.invoke("Inpainting Frame $decodedFrameCount/$totalFrames", percent)
                                }
                            }
                            codec.releaseOutputBuffer(outIndex, false)
                        }
                    }
                }

                onProgress?.invoke("Finalizing MP4 Video", 99)
                encoder.finish()
            } finally {
                encoder.release()
                codec.stop()
                codec.release()
            }

            onProgress?.invoke("Video Inpainting Complete", 100)
            return@withContext VideoProcessingResult(
                outputFile = outputFile,
                totalFrames = decodedFrameCount,
                durationMs = durationMs,
                width = targetW,
                height = targetH
            )
        } finally {
            retriever.release()
            extractor.release()
            accumulator.close()
        }
    }

    private fun imageToMat(image: Image): Mat {
        val width = image.width
        val height = image.height

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val nv21Size = width * height * 3 / 2
        val nv21 = ByteArray(nv21Size)

        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride

        var pos = 0
        if (yPixelStride == 1 && yRowStride == width) {
            yBuffer.rewind()
            yBuffer.get(nv21, 0, width * height)
            pos = width * height
        } else {
            val yBytes = ByteArray(yRowStride)
            for (row in 0 until height) {
                yBuffer.position(row * yRowStride)
                yBuffer.get(yBytes, 0, width)
                System.arraycopy(yBytes, 0, nv21, pos, width)
                pos += width
            }
        }

        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride
        val uvWidth = width / 2
        val uvHeight = height / 2

        val uBytes = ByteArray(uvRowStride)
        val vBytes = ByteArray(vPlane.rowStride)

        for (row in 0 until uvHeight) {
            uBuffer.position(row * uvRowStride)
            uBuffer.get(uBytes, 0, kotlin.math.max(0, kotlin.math.min(uBuffer.remaining(), uvRowStride)))
            vBuffer.position(row * vPlane.rowStride)
            vBuffer.get(vBytes, 0, kotlin.math.max(0, kotlin.math.min(vBuffer.remaining(), vPlane.rowStride)))

            for (col in 0 until uvWidth) {
                val vVal = vBytes[col * uvPixelStride]
                val uVal = uBytes[col * uvPixelStride]
                nv21[pos++] = vVal
                nv21[pos++] = uVal
            }
        }

        val yuvMat = Mat(height + height / 2, width, CvType.CV_8UC1)
        yuvMat.put(0, 0, nv21)

        val rgbaMat = Mat()
        Imgproc.cvtColor(yuvMat, rgbaMat, Imgproc.COLOR_YUV2RGBA_NV21)
        yuvMat.release()

        return rgbaMat
    }
}
