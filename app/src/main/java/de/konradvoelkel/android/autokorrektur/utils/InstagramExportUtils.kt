package de.konradvoelkel.android.autokorrektur.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Rect
import android.graphics.RectF
import android.net.Uri
import androidx.core.content.FileProvider
import androidx.core.graphics.withClip
import de.konradvoelkel.android.autokorrektur.video.VideoEncoder
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.math.sin

/**
 * Utility for rendering Instagram-ready (1:1, 4:5, 9:16) comparison graphics and animated sweep videos
 * combining "VORHER" and "NACHHER" photo bitmaps.
 */
object InstagramExportUtils {

    /**
     * Common aspect ratios supported by Instagram posts, carousels, and stories.
     */
    enum class AspectRatio(val width: Int, val height: Int) {
        /** 1:1 square format (1080x1080). */
        SQUARE_1_1(1080, 1080),
        /** 4:5 vertical portrait feed format (1080x1350). */
        PORTRAIT_4_5(1080, 1350),
        /** 9:16 vertical full-screen Story/Reel format (1080x1920). */
        STORY_9_16(1080, 1920)
    }

    /**
     * Layout arrangement of Before and After images.
     */
    enum class LayoutStyle {
        /** Left and right comparison panes. */
        SIDE_BY_SIDE,
        /** Top and bottom comparison panes. */
        STACKED
    }

    /**
     * Composes a Before/After side-by-side or stacked split comparison graphic matching the specified aspect ratio.
     */
    fun createComparisonBitmap(
        beforeBitmap: Bitmap,
        afterBitmap: Bitmap,
        ratio: AspectRatio = AspectRatio.SQUARE_1_1,
        layout: LayoutStyle = LayoutStyle.SIDE_BY_SIDE
    ): Bitmap {
        val targetWidth = ratio.width
        val targetHeight = ratio.height

        val resultBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(resultBitmap)

        // 1. Draw dark background
        val bgPaint = Paint().apply {
            color = Color.parseColor("#121212")
            style = Paint.Style.FILL
        }
        canvas.drawRect(0f, 0f, targetWidth.toFloat(), targetHeight.toFloat(), bgPaint)

        // 2. Draw Before & After regions based on layout
        val dividerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(220, 255, 255, 255)
            strokeWidth = 4f
            style = Paint.Style.STROKE
        }

        when (layout) {
            LayoutStyle.SIDE_BY_SIDE -> {
                val halfWidth = targetWidth / 2f
                val leftRect = RectF(0f, 0f, halfWidth, targetHeight.toFloat())
                val rightRect = RectF(halfWidth, 0f, targetWidth.toFloat(), targetHeight.toFloat())

                drawScaledCenterCrop(canvas, beforeBitmap, leftRect)
                drawScaledCenterCrop(canvas, afterBitmap, rightRect)

                canvas.drawLine(halfWidth, 0f, halfWidth, targetHeight.toFloat(), dividerPaint)

                drawBadge(canvas, "VORHER", leftRect, isTop = true)
                drawBadge(canvas, "AUTOFREI", rightRect, isTop = true)
            }
            LayoutStyle.STACKED -> {
                val halfHeight = targetHeight / 2f
                val topRect = RectF(0f, 0f, targetWidth.toFloat(), halfHeight)
                val bottomRect = RectF(0f, halfHeight, targetWidth.toFloat(), targetHeight.toFloat())

                drawScaledCenterCrop(canvas, beforeBitmap, topRect)
                drawScaledCenterCrop(canvas, afterBitmap, bottomRect)

                canvas.drawLine(0f, halfHeight, targetWidth.toFloat(), halfHeight, dividerPaint)

                drawBadge(canvas, "VORHER", topRect, isTop = true)
                drawBadge(canvas, "AUTOFREI", bottomRect, isTop = true)
            }
        }

        return resultBitmap
    }

    /**
     * Generates a 2-slide carousel image pair (Slide 1: Before, Slide 2: After) styled for Instagram.
     */
    fun createCarouselPair(
        beforeBitmap: Bitmap,
        afterBitmap: Bitmap,
        ratio: AspectRatio = AspectRatio.PORTRAIT_4_5
    ): Pair<Bitmap, Bitmap> {
        val targetW = ratio.width
        val targetH = ratio.height
        val fullRect = RectF(0f, 0f, targetW.toFloat(), targetH.toFloat())

        val slide1 = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        val canvas1 = Canvas(slide1)
        drawScaledCenterCrop(canvas1, beforeBitmap, fullRect)
        drawBadge(canvas1, "1/2  VORHER", fullRect, isTop = true)

        val slide2 = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        val canvas2 = Canvas(slide2)
        drawScaledCenterCrop(canvas2, afterBitmap, fullRect)
        drawBadge(canvas2, "2/2  AUTOFREI", fullRect, isTop = true)

        return Pair(slide1, slide2)
    }

    /**
     * Generates a 3.5s looping MP4 video where a vertical split slider sweeps across the screen.
     */
    fun createAnimatedSweepVideo(
        beforeBitmap: Bitmap,
        afterBitmap: Bitmap,
        outputFile: File,
        ratio: AspectRatio = AspectRatio.STORY_9_16,
        fps: Int = 30,
        durationSeconds: Float = 3.5f
    ): File {
        val targetW = (ratio.width / 2) * 2
        val targetH = (ratio.height / 2) * 2
        val totalFrames = (durationSeconds * fps).toInt().coerceAtLeast(30)

        val encoder = VideoEncoder(
            width = targetW,
            height = targetH,
            frameRate = fps,
            bitRate = 8_000_000
        )
        encoder.start(outputFile)

        val frameBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(frameBitmap)
        val fullRect = RectF(0f, 0f, targetW.toFloat(), targetH.toFloat())

        val dividerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            strokeWidth = 6f
            style = Paint.Style.STROKE
        }

        try {
            for (i in 0 until totalFrames) {
                val t = i.toFloat() / totalFrames.toFloat()
                // Smooth sine oscillation between 0.1 and 0.9
                val splitPosition = (0.5f - 0.42f * sin(2.0 * Math.PI * t)).toFloat()
                val splitX = targetW * splitPosition

                // 1. Draw After Bitmap everywhere
                drawScaledCenterCrop(canvas, afterBitmap, fullRect)

                // 2. Clip left region and draw Before Bitmap
                canvas.withClip(0f, 0f, splitX, targetH.toFloat()) {
                    drawScaledCenterCrop(this, beforeBitmap, fullRect)
                }

                // 3. Draw divider line
                canvas.drawLine(splitX, 0f, splitX, targetH.toFloat(), dividerPaint)

                // 4. Draw Badges
                if (splitX > 200f) {
                    drawBadge(canvas, "VORHER", RectF(0f, 0f, splitX, targetH.toFloat()), isTop = true)
                }
                if (splitX < targetW - 200f) {
                    drawBadge(canvas, "AUTOFREI", RectF(splitX, 0f, targetW.toFloat(), targetH.toFloat()), isTop = true)
                }

                encoder.encodeFrame(frameBitmap)
            }
            encoder.finish()
        } finally {
            encoder.release()
            frameBitmap.recycle()
        }

        return outputFile
    }

    /**
     * Draws a bitmap cropped & scaled to fill the target destination rectangle while maintaining aspect ratio.
     */
    private fun drawScaledCenterCrop(canvas: Canvas, srcBitmap: Bitmap, destRect: RectF) {
        val srcWidth = srcBitmap.width.toFloat()
        val srcHeight = srcBitmap.height.toFloat()

        val srcAspect = srcWidth / srcHeight
        val destAspect = destRect.width() / destRect.height()

        val srcCrop: Rect = if (srcAspect > destAspect) {
            val cropWidth = srcHeight * destAspect
            val left = (srcWidth - cropWidth) / 2f
            Rect(left.toInt(), 0, (left + cropWidth).toInt(), srcHeight.toInt())
        } else {
            val cropHeight = srcWidth / destAspect
            val top = (srcHeight - cropHeight) / 2f
            Rect(0, top.toInt(), srcWidth.toInt(), (top + cropHeight).toInt())
        }

        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        canvas.drawBitmap(srcBitmap, srcCrop, destRect, paint)
    }

    /**
     * Draws a styled "VORHER" or "AUTOFREI" pill badge inside the destination region.
     */
    private fun drawBadge(canvas: Canvas, text: String, region: RectF, isTop: Boolean) {
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 28f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
        }

        val textBounds = Rect()
        textPaint.getTextBounds(text, 0, text.length, textBounds)

        val paddingX = 24f
        val paddingY = 14f
        val badgeWidth = textBounds.width() + (paddingX * 2)
        val badgeHeight = textBounds.height() + (paddingY * 2)

        val margin = 20f
        val badgeLeft = region.left + margin
        val badgeTop = if (isTop) region.top + margin else region.bottom - margin - badgeHeight
        val badgeRect = RectF(badgeLeft, badgeTop, badgeLeft + badgeWidth, badgeTop + badgeHeight)

        // Background pill badge
        val badgeBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(200, 15, 15, 20)
            style = Paint.Style.FILL
        }
        val cornerRadius = 14f
        canvas.drawRoundRect(badgeRect, cornerRadius, cornerRadius, badgeBgPaint)

        // Border around badge
        val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(120, 255, 255, 255)
            style = Paint.Style.STROKE
            strokeWidth = 2f
        }
        canvas.drawRoundRect(badgeRect, cornerRadius, cornerRadius, borderPaint)

        // Draw text
        val textX = badgeRect.left + paddingX
        val textY = badgeRect.top + paddingY + textBounds.height() - 4f
        canvas.drawText(text, textX, textY, textPaint)
    }

    /**
     * Saves a bitmap to temporary app cache and returns a content Uri for sharing via FileProvider.
     */
    @Throws(IOException::class)
    fun saveBitmapForSharing(
        context: Context,
        bitmap: Bitmap,
        fileName: String = "autokorrektur_share_${System.currentTimeMillis()}.jpg"
    ): Uri {
        val imagesDir = File(context.cacheDir, "images").apply { mkdirs() }
        val imageFile = File(imagesDir, fileName)

        FileOutputStream(imageFile).use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 92, out)
        }

        val authority = "${context.packageName}.fileprovider"
        return FileProvider.getUriForFile(context, authority, imageFile)
    }
}
