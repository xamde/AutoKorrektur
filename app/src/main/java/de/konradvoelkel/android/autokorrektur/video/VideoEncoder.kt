package de.konradvoelkel.android.autokorrektur.video

import android.graphics.Bitmap
import android.graphics.Canvas
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.view.Surface
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import java.io.File

/**
 * High-performance hardware-accelerated MP4 (H.264) video encoder using Android [MediaCodec] and [MediaMuxer].
 * Encodes a sequence of [Bitmap] frames directly to an MP4 video file.
 */
class VideoEncoder(
    private val width: Int,
    private val height: Int,
    private val frameRate: Int = 30,
    private val bitRate: Int = 6_000_000,
    private val iFrameInterval: Int = 1
) {
    private var mediaCodec: MediaCodec? = null
    private var mediaMuxer: MediaMuxer? = null
    private var inputSurface: Surface? = null
    private var trackIndex = -1
    private var isMuxerStarted = false
    private val bufferInfo = MediaCodec.BufferInfo()
    private var frameIndex = 0L

    /**
     * Prepares the encoder and output destination.
     */
    fun start(outputFile: File) {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, bitRate)
            setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, iFrameInterval)
        }

        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        inputSurface = codec.createInputSurface()
        codec.start()
        mediaCodec = codec

        mediaMuxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        isMuxerStarted = false
        trackIndex = -1
        frameIndex = 0L
    }

    /**
     * Draws a [Bitmap] frame onto the encoder input surface and drains available encoded buffers.
     */
    fun encodeFrame(bitmap: Bitmap) {
        val surface = inputSurface ?: return
        val canvas: Canvas = surface.lockHardwareCanvas()
        try {
            canvas.drawBitmap(bitmap, 0f, 0f, null)
        } finally {
            surface.unlockCanvasAndPost(canvas)
        }
        drainEncoder(endOfStream = false)
        frameIndex++
    }

    /**
     * Signals End-of-Stream, writes remaining encoded packets, and closes codec/muxer.
     */
    fun finish() {
        try {
            mediaCodec?.signalEndOfInputStream()
            drainEncoder(endOfStream = true)
        } catch (e: Exception) {
            AppLogger.error("Error signaling EOS to VideoEncoder", e)
        } finally {
            release()
        }
    }

    private fun drainEncoder(endOfStream: Boolean) {
        val codec = mediaCodec ?: return
        val muxer = mediaMuxer ?: return
        val timeoutUs = 10_000L

        while (true) {
            val encoderStatus = codec.dequeueOutputBuffer(bufferInfo, timeoutUs)
            if (encoderStatus == MediaCodec.INFO_TRY_AGAIN_LATER) {
                if (!endOfStream) break
            } else if (encoderStatus == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                if (isMuxerStarted) {
                    throw IllegalStateException("Format changed twice in VideoEncoder")
                }
                val newFormat = codec.outputFormat
                trackIndex = muxer.addTrack(newFormat)
                muxer.start()
                isMuxerStarted = true
            } else if (encoderStatus >= 0) {
                val encodedData = codec.getOutputBuffer(encoderStatus) ?: continue
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                    bufferInfo.size = 0
                }

                if (bufferInfo.size != 0) {
                    if (!isMuxerStarted) {
                        throw RuntimeException("Muxer hasn't started before buffer arrived")
                    }
                    encodedData.position(bufferInfo.offset)
                    encodedData.limit(bufferInfo.offset + bufferInfo.size)
                    muxer.writeSampleData(trackIndex, encodedData, bufferInfo)
                }

                codec.releaseOutputBuffer(encoderStatus, false)

                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                    break
                }
            }
        }
    }

    fun release() {
        try {
            mediaCodec?.stop()
            mediaCodec?.release()
            mediaCodec = null
        } catch (_: Exception) {}

        try {
            if (isMuxerStarted) {
                mediaMuxer?.stop()
            }
            mediaMuxer?.release()
            mediaMuxer = null
        } catch (_: Exception) {}

        try {
            inputSurface?.release()
            inputSurface = null
        } catch (_: Exception) {}
    }
}
