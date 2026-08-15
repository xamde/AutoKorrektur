package de.konradvoelkel.android.autokorrektur.video

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
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
    private var trackIndex = -1
    private var isMuxerStarted = false
    private val bufferInfo = MediaCodec.BufferInfo()
    private var frameIndex = 0L

    /**
     * Prepares the encoder and output destination.
     */
    fun start(outputFile: File) {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            setInteger(MediaFormat.KEY_BIT_RATE, bitRate)
            setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, iFrameInterval)
        }

        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        try {
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            codec.start()
            mediaCodec = codec
        } catch (e: Exception) {
            codec.release()
            throw e
        }

        mediaMuxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        isMuxerStarted = false
        trackIndex = -1
        frameIndex = 0L
    }

    /**
     * Encodes a [Bitmap] frame by converting it to YUV and queueing it to the codec input buffer.
     */
    fun encodeFrame(bitmap: Bitmap) {
        val codec = mediaCodec ?: return
        
        var inputBufferIndex = -1
        while (inputBufferIndex < 0) {
            inputBufferIndex = codec.dequeueInputBuffer(10_000L)
        }
        
        val inputBuffer = codec.getInputBuffer(inputBufferIndex) ?: return
        inputBuffer.clear()
        
        val yuvData = getNV12(width, height, bitmap)
        inputBuffer.put(yuvData)
        
        val ptsUs = frameIndex * (1_000_000L / frameRate)
        codec.queueInputBuffer(inputBufferIndex, 0, yuvData.size, ptsUs, 0)
        
        drainEncoder(endOfStream = false)
        frameIndex++
    }

    /**
     * Signals End-of-Stream, writes remaining encoded packets, and closes codec/muxer.
     */
    fun finish() {
        try {
            val codec = mediaCodec
            if (codec != null) {
                var inputBufferIndex = -1
                while (inputBufferIndex < 0) {
                    inputBufferIndex = codec.dequeueInputBuffer(10_000L)
                }
                codec.queueInputBuffer(inputBufferIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
            }
            drainEncoder(endOfStream = true)
        } catch (e: Exception) {
            AppLogger.error("Error signaling EOS to VideoEncoder", e)
        } finally {
            release()
        }
    }

    private fun getNV12(inputWidth: Int, inputHeight: Int, bitmap: Bitmap): ByteArray {
        val argb = IntArray(inputWidth * inputHeight)
        bitmap.getPixels(argb, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        val yuv = ByteArray(inputWidth * inputHeight * 3 / 2)
        var yIndex = 0
        var uvIndex = inputWidth * inputHeight
        var index = 0
        for (y in 0 until inputHeight) {
            for (x in 0 until inputWidth) {
                val color = argb[index]
                val R = (color and 0xff0000) shr 16
                val G = (color and 0xff00) shr 8
                val B = (color and 0xff)
                
                val Y = ((66 * R + 129 * G + 25 * B + 128) shr 8) + 16
                val U = ((-38 * R - 74 * G + 112 * B + 128) shr 8) + 128
                val V = ((112 * R - 94 * G - 18 * B + 128) shr 8) + 128
                
                yuv[yIndex++] = Y.coerceIn(0, 255).toByte()
                if (y % 2 == 0 && x % 2 == 0) {
                    yuv[uvIndex++] = U.coerceIn(0, 255).toByte()
                    yuv[uvIndex++] = V.coerceIn(0, 255).toByte()
                }
                index++
            }
        }
        return yuv
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
    }
}
