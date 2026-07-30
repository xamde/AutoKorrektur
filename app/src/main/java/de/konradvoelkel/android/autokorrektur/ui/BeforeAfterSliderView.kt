package de.konradvoelkel.android.autokorrektur.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.min

/**
 * Interactive Before/After split-slider view allowing users to drag a vertical handle
 * across the view to reveal the original (Before) and inpainted (After) bitmaps in real time.
 */
class BeforeAfterSliderView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var beforeBitmap: Bitmap? = null
    private var afterBitmap: Bitmap? = null

    // Normalized slider position between 0.0 (all After) and 1.0 (all Before)
    private var sliderPosition: Float = 0.5f

    private val dividerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        strokeWidth = 6f
        style = Paint.Style.STROKE
    }

    private val handleBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(220, 0, 0, 0)
        style = Paint.Style.FILL
    }

    private val handleBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        strokeWidth = 4f
        style = Paint.Style.STROKE
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 32f
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }

    private val bitmapPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)

    /**
     * Sets the Before (original with car) and After (processed car-free) bitmaps.
     */
    fun setBitmaps(before: Bitmap, after: Bitmap) {
        this.beforeBitmap = before
        this.afterBitmap = after
        invalidate()
    }

    /**
     * Updates the normalized slider position (0.0 to 1.0).
     */
    fun setSliderPosition(position: Float) {
        this.sliderPosition = position.coerceIn(0f, 1f)
        invalidate()
    }

    fun getSliderPosition(): Float = sliderPosition

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (width <= 0) return super.onTouchEvent(event)

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                sliderPosition = (event.x / width.toFloat()).coerceIn(0f, 1f)
                parent?.requestDisallowInterceptTouchEvent(true)
                invalidate()
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                parent?.requestDisallowInterceptTouchEvent(false)
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val before = beforeBitmap
        val after = afterBitmap

        if (before == null || after == null || width <= 0 || height <= 0) return

        val viewRect = RectF(0f, 0f, width.toFloat(), height.toFloat())
        val splitX = width * sliderPosition

        // 1. Draw After Bitmap across full view bounds
        drawCenterCropBitmap(canvas, after, viewRect)

        // 2. Clip and draw Before Bitmap on the left side of splitX
        canvas.save()
        canvas.clipRect(0f, 0f, splitX, height.toFloat())
        drawCenterCropBitmap(canvas, before, viewRect)
        canvas.restore()

        // 3. Draw vertical divider line
        canvas.drawLine(splitX, 0f, splitX, height.toFloat(), dividerPaint)

        // 4. Draw circular handle thumb in center of divider
        val handleRadius = 40f
        val handleCenterY = height / 2f
        canvas.drawCircle(splitX, handleCenterY, handleRadius, handleBgPaint)
        canvas.drawCircle(splitX, handleCenterY, handleRadius, handleBorderPaint)

        // Draw left/right arrows inside handle thumb ("< >")
        val text = "⮜ ⮞"
        val textBounds = Rect()
        textPaint.getTextBounds(text, 0, text.length, textBounds)
        val textX = splitX - (textBounds.width() / 2f)
        val textY = handleCenterY + (textBounds.height() / 2f) - 2f
        canvas.drawText(text, textX, textY, textPaint)
    }

    private fun drawCenterCropBitmap(canvas: Canvas, bitmap: Bitmap, destRect: RectF) {
        val srcW = bitmap.width.toFloat()
        val srcH = bitmap.height.toFloat()
        val srcAspect = srcW / srcH
        val destAspect = destRect.width() / destRect.height()

        val srcCrop: Rect = if (srcAspect > destAspect) {
            val cropW = srcH * destAspect
            val left = (srcW - cropW) / 2f
            Rect(left.toInt(), 0, (left + cropW).toInt(), srcH.toInt())
        } else {
            val cropH = srcW / destAspect
            val top = (srcH - cropH) / 2f
            Rect(0, top.toInt(), srcW.toInt(), (top + cropH).toInt())
        }
        canvas.drawBitmap(bitmap, srcCrop, destRect, bitmapPaint)
    }
}
