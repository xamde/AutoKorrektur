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
import androidx.core.graphics.withClip

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
        color = Color.argb(230, 20, 20, 25)
        style = Paint.Style.FILL
    }

    private val handleBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        strokeWidth = 4f
        style = Paint.Style.STROKE
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 28f
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }

    private val badgeBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(175, 0, 0, 0)
        style = Paint.Style.FILL
    }

    private val badgeTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 24f
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }

    private val bitmapPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)

    // Cached objects for onDraw to avoid GC churn
    private val viewRect = RectF()
    private val textBounds = Rect()
    private val srcCrop = Rect()
    private val badgeRect = RectF()
    private val handleText = "◀  ▶"
    private var revealAnimator: android.animation.ValueAnimator? = null

    init {
        isFocusable = true
        isFocusableInTouchMode = true
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_YES
    }

    /**
     * Sets the Before (original with car) and After (processed car-free) bitmaps.
     */
    fun setBitmaps(before: Bitmap, after: Bitmap, animate: Boolean = true) {
        this.beforeBitmap = before
        this.afterBitmap = after
        invalidate()
        if (animate) {
            animateReveal()
        }
    }

    /**
     * Sweeps the slider to visually showcase the inpainting transition.
     */
    fun animateReveal() {
        revealAnimator?.cancel()
        revealAnimator = android.animation.ValueAnimator.ofFloat(1.0f, 0.0f, 0.5f).apply {
            duration = 1000L
            interpolator = android.view.animation.DecelerateInterpolator()
            addUpdateListener {
                sliderPosition = it.animatedValue as Float
                invalidate()
            }
            start()
        }
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
                revealAnimator?.cancel()
                updateSliderPosition(event.x)
                parent?.requestDisallowInterceptTouchEvent(true)
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                parent?.requestDisallowInterceptTouchEvent(false)
                performClick()
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    private fun updateSliderPosition(x: Float) {
        sliderPosition = (x / width.toFloat()).coerceIn(0f, 1f)
        contentDescription = "Before/After slider at ${(sliderPosition * 100).toInt()}%"
        invalidate()
    }

    override fun onKeyDown(keyCode: Int, event: android.view.KeyEvent?): Boolean {
        return when (keyCode) {
            android.view.KeyEvent.KEYCODE_DPAD_LEFT -> {
                setSliderPosition(sliderPosition - 0.05f)
                true
            }

            android.view.KeyEvent.KEYCODE_DPAD_RIGHT -> {
                setSliderPosition(sliderPosition + 0.05f)
                true
            }

            else -> super.onKeyDown(keyCode, event)
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val before = beforeBitmap
        val after = afterBitmap

        if (before == null || after == null || width <= 0 || height <= 0) return

        val viewW = width.toFloat()
        val viewH = height.toFloat()
        val splitX = viewW * sliderPosition

        // Calculate aspect ratio fit bounding box (FIT_CENTER)
        val srcW = before.width.toFloat()
        val srcH = before.height.toFloat()
        val srcAspect = srcW / srcH
        val destAspect = viewW / viewH

        val drawRect = RectF()
        if (srcAspect > destAspect) {
            val drawH = viewW / srcAspect
            val top = (viewH - drawH) / 2f
            drawRect.set(0f, top, viewW, top + drawH)
        } else {
            val drawW = viewH * srcAspect
            val left = (viewW - drawW) / 2f
            drawRect.set(left, 0f, left + drawW, viewH)
        }

        // 1. Draw After Bitmap across destination fit rectangle
        srcCrop.set(0, 0, after.width, after.height)
        canvas.drawBitmap(after, srcCrop, drawRect, bitmapPaint)

        // 2. Clip and draw Before Bitmap on the left side of splitX
        canvas.withClip(0f, 0f, splitX, viewH) {
            srcCrop.set(0, 0, before.width, before.height)
            drawBitmap(before, srcCrop, drawRect, bitmapPaint)
        }

        // 3. Draw vertical divider line across image area
        canvas.drawLine(splitX, drawRect.top, splitX, drawRect.bottom, dividerPaint)

        // 4. Draw circular handle thumb in center of divider
        val handleRadius = 40f
        val handleCenterY = (drawRect.top + drawRect.bottom) / 2f
        canvas.drawCircle(splitX, handleCenterY, handleRadius, handleBgPaint)
        canvas.drawCircle(splitX, handleCenterY, handleRadius, handleBorderPaint)

        // Draw left/right arrows inside handle thumb ("◀  ▶")
        textPaint.getTextBounds(handleText, 0, handleText.length, textBounds)
        val textX = splitX - (textBounds.width() / 2f)
        val textY = handleCenterY + (textBounds.height() / 2f) - 2f
        canvas.drawText(handleText, textX, textY, textPaint)

        // 5. Draw "VORHER" badge on left if visible
        if (splitX > drawRect.left + 80f) {
            drawBadge(canvas, "VORHER", drawRect.left + 24f, drawRect.top + 24f)
        }

        // 6. Draw "NACHHER" badge on right if visible
        if (splitX < drawRect.right - 80f) {
            val badgeW = calculateBadgeWidth("NACHHER")
            drawBadge(canvas, "NACHHER", drawRect.right - badgeW - 24f, drawRect.top + 24f)
        }
    }

    private fun calculateBadgeWidth(text: String): Float {
        badgeTextPaint.getTextBounds(text, 0, text.length, textBounds)
        return textBounds.width() + 32f
    }

    private fun drawBadge(canvas: Canvas, text: String, left: Float, top: Float) {
        badgeTextPaint.getTextBounds(text, 0, text.length, textBounds)
        val badgeW = textBounds.width() + 32f
        val badgeH = textBounds.height() + 20f
        badgeRect.set(left, top, left + badgeW, top + badgeH)
        canvas.drawRoundRect(badgeRect, 12f, 12f, badgeBgPaint)
        val textX = left + 16f
        val textY = top + badgeH - 12f
        canvas.drawText(text, textX, textY, badgeTextPaint)
    }

    override fun onSaveInstanceState(): android.os.Parcelable {
        val bundle = android.os.Bundle()
        bundle.putParcelable("superState", super.onSaveInstanceState())
        bundle.putFloat("sliderPosition", sliderPosition)
        return bundle
    }

    override fun onRestoreInstanceState(state: android.os.Parcelable?) {
        var viewState = state
        if (viewState is android.os.Bundle) {
            sliderPosition = viewState.getFloat("sliderPosition", 0.5f)
            @Suppress("DEPRECATION")
            viewState = viewState.getParcelable("superState")
        }
        super.onRestoreInstanceState(viewState)
    }
}
