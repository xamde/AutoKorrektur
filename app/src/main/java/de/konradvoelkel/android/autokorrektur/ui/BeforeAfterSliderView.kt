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

    private val density: Float by lazy { context.resources.displayMetrics.density }

    private val dividerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        strokeWidth = 3f * context.resources.displayMetrics.density
        style = Paint.Style.STROKE
    }

    private val handleBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(230, 20, 20, 25)
        style = Paint.Style.FILL
    }

    private val handleBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        strokeWidth = 2f * context.resources.displayMetrics.density
        style = Paint.Style.STROKE
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 14f * context.resources.displayMetrics.scaledDensity
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }

    private val badgeBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(175, 0, 0, 0)
        style = Paint.Style.FILL
    }

    private val badgeTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 12f * context.resources.displayMetrics.scaledDensity
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
     * Dynamically updates the After bitmap during progressive inpainting without re-triggering animation.
     */
    fun updateAfterBitmap(after: Bitmap) {
        this.afterBitmap = after
        invalidate()
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
     * Sets slider position programmatically (0.0 to 1.0).
     */
    fun setSliderPosition(position: Float) {
        this.sliderPosition = position.coerceIn(0.0f, 1.0f)
        invalidate()
    }

    /**
     * Returns current slider position (0.0 to 1.0).
     */
    fun getSliderPosition(): Float = sliderPosition

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val before = beforeBitmap ?: return super.onTouchEvent(event)
        val viewW = width.toFloat()
        val viewH = height.toFloat()
        if (viewW <= 0 || viewH <= 0) return super.onTouchEvent(event)

        val srcW = before.width.toFloat()
        val srcH = before.height.toFloat()
        val srcAspect = srcW / srcH
        val destAspect = viewW / viewH

        val left: Float
        val right: Float
        if (srcAspect > destAspect) {
            left = 0f
            right = viewW
        } else {
            val drawW = viewH * srcAspect
            left = (viewW - drawW) / 2f
            right = left + drawW
        }

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN,
            MotionEvent.ACTION_MOVE -> {
                parent?.requestDisallowInterceptTouchEvent(true)
                val clampedX = event.x.coerceIn(left, right)
                sliderPosition = if (right > left) (clampedX - left) / (right - left) else 0.5f
                invalidate()
                return true
            }
            MotionEvent.ACTION_UP,
            MotionEvent.ACTION_CANCEL -> {
                parent?.requestDisallowInterceptTouchEvent(false)
                return true
            }
        }
        return super.onTouchEvent(event)
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

        val before = beforeBitmap ?: return
        val after = afterBitmap ?: return

        val viewW = width.toFloat()
        val viewH = height.toFloat()
        if (viewW <= 0 || viewH <= 0) return

        // Compute aspect-fit destination rectangle inside this View
        val srcW = before.width.toFloat()
        val srcH = before.height.toFloat()
        val srcAspect = srcW / srcH
        val destAspect = viewW / viewH

        if (srcAspect > destAspect) {
            val drawH = viewW / srcAspect
            val top = (viewH - drawH) / 2f
            viewRect.set(0f, top, viewW, top + drawH)
        } else {
            val drawW = viewH * srcAspect
            val left = (viewW - drawW) / 2f
            viewRect.set(left, 0f, left + drawW, viewH)
        }

        val splitX = viewRect.left + (viewRect.width() * sliderPosition)

        // 1. Draw After Bitmap across destination fit rectangle
        srcCrop.set(0, 0, after.width, after.height)
        canvas.drawBitmap(after, srcCrop, viewRect, bitmapPaint)

        // 2. Clip and draw Before Bitmap on the left side of splitX
        canvas.withClip(0f, 0f, splitX, viewH) {
            srcCrop.set(0, 0, before.width, before.height)
            drawBitmap(before, srcCrop, viewRect, bitmapPaint)
        }

        // 3. Draw vertical divider line across image area
        canvas.drawLine(splitX, viewRect.top, splitX, viewRect.bottom, dividerPaint)

        // 4. Draw circular handle thumb in center of divider
        val handleRadius = 20f * density
        val handleCenterY = (viewRect.top + viewRect.bottom) / 2f
        canvas.drawCircle(splitX, handleCenterY, handleRadius, handleBgPaint)
        canvas.drawCircle(splitX, handleCenterY, handleRadius, handleBorderPaint)

        // Draw left/right arrows inside handle thumb ("◀  ▶")
        textPaint.getTextBounds(handleText, 0, handleText.length, textBounds)
        val textX = splitX - (textBounds.width() / 2f)
        val textY = handleCenterY + (textBounds.height() / 2f) - (1f * density)
        canvas.drawText(handleText, textX, textY, textPaint)

        // 5. Draw "VORHER" badge on left if visible
        val badgeMargin = 40f * density
        val badgeOffset = 12f * density
        if (splitX > viewRect.left + badgeMargin) {
            drawBadge(canvas, "VORHER", viewRect.left + badgeOffset, viewRect.top + badgeOffset)
        }

        // 6. Draw "NACHHER" badge on right if visible
        if (splitX < viewRect.right - badgeMargin) {
            val badgeW = calculateBadgeWidth("NACHHER")
            drawBadge(canvas, "NACHHER", viewRect.right - badgeW - badgeOffset, viewRect.top + badgeOffset)
        }
    }

    private fun calculateBadgeWidth(text: String): Float {
        badgeTextPaint.getTextBounds(text, 0, text.length, textBounds)
        return textBounds.width() + (16f * density)
    }

    private fun drawBadge(canvas: Canvas, text: String, left: Float, top: Float) {
        badgeTextPaint.getTextBounds(text, 0, text.length, textBounds)
        val badgeW = textBounds.width() + (16f * density)
        val badgeH = textBounds.height() + (10f * density)
        badgeRect.set(left, top, left + badgeW, top + badgeH)
        canvas.drawRoundRect(badgeRect, 6f * density, 6f * density, badgeBgPaint)
        val textX = left + (8f * density)
        val textY = top + badgeH - (6f * density)
        canvas.drawText(text, textX, textY, badgeTextPaint)
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        revealAnimator?.cancel()
        revealAnimator = null
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
