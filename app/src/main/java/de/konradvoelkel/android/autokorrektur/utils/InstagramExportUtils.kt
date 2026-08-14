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
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * Utility for rendering Instagram-ready (1:1, 4:5, 9:16) comparison graphics
 * combining "BEFORE" and "AFTER" photo bitmaps.
 */
object InstagramExportUtils {

    /**
     * Common aspect ratios supported by Instagram posts and stories.
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
     * Composes a Before/After comparison graphic matching the specified Instagram aspect ratio and layout.
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
        val dividerPaint = Paint().apply {
            color = Color.argb(180, 255, 255, 255)
            strokeWidth = 4f
            style = Paint.Style.STROKE
            isAntiAlias = true
        }

        when (layout) {
            LayoutStyle.SIDE_BY_SIDE -> {
                val halfWidth = targetWidth / 2f
                val leftRect = RectF(0f, 0f, halfWidth, targetHeight.toFloat())
                val rightRect = RectF(halfWidth, 0f, targetWidth.toFloat(), targetHeight.toFloat())

                drawScaledCenterCrop(canvas, beforeBitmap, leftRect)
                drawScaledCenterCrop(canvas, afterBitmap, rightRect)

                // Divider line
                canvas.drawLine(halfWidth, 0f, halfWidth, targetHeight.toFloat(), dividerPaint)

                // Badges
                drawBadge(canvas, "BEFORE", leftRect, isTop = true)
                drawBadge(canvas, "AFTER", rightRect, isTop = true)
            }
            LayoutStyle.STACKED -> {
                val halfHeight = targetHeight / 2f
                val topRect = RectF(0f, 0f, targetWidth.toFloat(), halfHeight)
                val bottomRect = RectF(0f, halfHeight, targetWidth.toFloat(), targetHeight.toFloat())

                drawScaledCenterCrop(canvas, beforeBitmap, topRect)
                drawScaledCenterCrop(canvas, afterBitmap, bottomRect)

                // Divider line
                canvas.drawLine(0f, halfHeight, targetWidth.toFloat(), halfHeight, dividerPaint)

                // Badges
                drawBadge(canvas, "BEFORE", topRect, isTop = true)
                drawBadge(canvas, "AFTER", bottomRect, isTop = true)
            }
        }

        return resultBitmap
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
            // Source is wider than dest: crop left and right
            val cropWidth = srcHeight * destAspect
            val left = (srcWidth - cropWidth) / 2f
            Rect(left.toInt(), 0, (left + cropWidth).toInt(), srcHeight.toInt())
        } else {
            // Source is taller than dest: crop top and bottom
            val cropHeight = srcWidth / destAspect
            val top = (srcHeight - cropHeight) / 2f
            Rect(0, top.toInt(), srcWidth.toInt(), (top + cropHeight).toInt())
        }

        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        canvas.drawBitmap(srcBitmap, srcCrop, destRect, paint)
    }

    /**
     * Draws a styled "BEFORE" or "AFTER" pill badge inside the destination region.
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
            color = Color.argb(190, 0, 0, 0)
            style = Paint.Style.FILL
        }
        val cornerRadius = 12f
        canvas.drawRoundRect(badgeRect, cornerRadius, cornerRadius, badgeBgPaint)

        // Border around badge
        val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(100, 255, 255, 255)
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
