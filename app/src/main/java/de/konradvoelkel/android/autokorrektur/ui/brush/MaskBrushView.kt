package de.konradvoelkel.android.autokorrektur.ui.brush

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

/**
 * Interactive canvas overlay allowing users to paint or erase manual vehicle inpainting mask regions.
 */
class MaskBrushView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    enum class ToolMode {
        BRUSH,
        ERASER
    }

    private var currentMode: ToolMode = ToolMode.BRUSH
    private var brushSizePx: Float = 40f * resources.displayMetrics.density

    private var baseBitmap: Bitmap? = null
    private var maskOverlayBitmap: Bitmap? = null
    private var maskCanvas: Canvas? = null

    private val currentPath = Path()
    private var lastX = 0f
    private var lastY = 0f

    private val brushPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(180, 239, 68, 68) // Semi-transparent red
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
    }

    private val eraserPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
    }

    private val drawPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
    private val viewRect = RectF()
    private val srcCrop = Rect()

    /**
     * Initializes the brush canvas with the background image and optional starting mask.
     */
    fun setup(imageBitmap: Bitmap, initialMaskBitmap: Bitmap? = null) {
        this.baseBitmap = imageBitmap
        val w = imageBitmap.width
        val h = imageBitmap.height

        val maskBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(maskBmp)

        if (initialMaskBitmap != null) {
            // Render initial mask as red overlay
            val initialPaint = Paint(Paint.ANTI_ALIAS_FLAG)
            canvas.drawBitmap(initialMaskBitmap, 0f, 0f, initialPaint)
        }

        this.maskOverlayBitmap = maskBmp
        this.maskCanvas = canvas
        invalidate()
    }

    fun setToolMode(mode: ToolMode) {
        this.currentMode = mode
    }

    fun setBrushSize(sizeDp: Float) {
        this.brushSizePx = sizeDp * resources.displayMetrics.density
    }

    fun clearMask() {
        maskCanvas?.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
        invalidate()
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val base = baseBitmap ?: return super.onTouchEvent(event)
        val viewW = width.toFloat()
        val viewH = height.toFloat()
        if (viewW <= 0 || viewH <= 0) return super.onTouchEvent(event)

        // Convert touch coordinates from View space to Bitmap space
        val srcW = base.width.toFloat()
        val srcH = base.height.toFloat()
        val srcAspect = srcW / srcH
        val destAspect = viewW / viewH

        val drawLeft: Float
        val drawTop: Float
        val drawW: Float
        val drawH: Float

        if (srcAspect > destAspect) {
            drawW = viewW
            drawH = viewW / srcAspect
            drawLeft = 0f
            drawTop = (viewH - drawH) / 2f
        } else {
            drawH = viewH
            drawW = viewH * srcAspect
            drawLeft = (viewW - drawW) / 2f
            drawTop = 0f
        }

        val bmpX = ((event.x - drawLeft) / drawW) * srcW
        val bmpY = ((event.y - drawTop) / drawH) * srcH

        val canvas = maskCanvas ?: return false
        val activePaint = if (currentMode == ToolMode.BRUSH) {
            brushPaint.apply { strokeWidth = brushSizePx * (srcW / drawW) }
        } else {
            eraserPaint.apply { strokeWidth = brushSizePx * (srcW / drawW) }
        }

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                parent?.requestDisallowInterceptTouchEvent(true)
                currentPath.reset()
                currentPath.moveTo(bmpX, bmpY)
                lastX = bmpX
                lastY = bmpY
                canvas.drawPoint(bmpX, bmpY, activePaint)
                invalidate()
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                val dx = Math.abs(bmpX - lastX)
                val dy = Math.abs(bmpY - lastY)
                if (dx >= 2 || dy >= 2) {
                    currentPath.quadTo(lastX, lastY, (bmpX + lastX) / 2f, (bmpY + lastY) / 2f)
                    canvas.drawPath(currentPath, activePaint)
                    currentPath.reset()
                    currentPath.moveTo((bmpX + lastX) / 2f, (bmpY + lastY) / 2f)
                    lastX = bmpX
                    lastY = bmpY
                    invalidate()
                }
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                parent?.requestDisallowInterceptTouchEvent(false)
                currentPath.lineTo(bmpX, bmpY)
                canvas.drawPath(currentPath, activePaint)
                currentPath.reset()
                invalidate()
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val base = baseBitmap ?: return
        val viewW = width.toFloat()
        val viewH = height.toFloat()
        if (viewW <= 0 || viewH <= 0) return

        val srcW = base.width.toFloat()
        val srcH = base.height.toFloat()
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

        // 1. Draw base photo
        srcCrop.set(0, 0, base.width, base.height)
        canvas.drawBitmap(base, srcCrop, viewRect, drawPaint)

        // 2. Draw mask overlay
        maskOverlayBitmap?.let { maskBmp ->
            canvas.drawBitmap(maskBmp, srcCrop, viewRect, drawPaint)
        }
    }

    /**
     * Converts the current drawn brush overlay into a binary OpenCV subtractive mask Mat
     * (0 for vehicle/mask hole, 255 for clean background).
     */
    fun exportSubtractiveMaskMat(): Mat {
        val base = baseBitmap ?: return Mat()
        val maskBmp = maskOverlayBitmap ?: return Mat(base.height, base.width, CvType.CV_8UC1, Scalar(255.0))

        val rgbaMat = Mat()
        Utils.bitmapToMat(maskBmp, rgbaMat)

        val channels = mutableListOf<Mat>()
        org.opencv.core.Core.split(rgbaMat, channels)

        val resultMask = Mat(base.height, base.width, CvType.CV_8UC1, Scalar(255.0))
        if (channels.size >= 4) {
            val alphaChannel = channels[3]
            // Where alpha > 0 (user painted red), invert to 0 in subtractive mask
            val thresholdedHoles = Mat()
            Imgproc.threshold(alphaChannel, thresholdedHoles, 10.0, 255.0, Imgproc.THRESH_BINARY)
            org.opencv.core.Core.bitwise_not(thresholdedHoles, resultMask)
            thresholdedHoles.release()
            alphaChannel.release()
        }

        channels.forEach { it.release() }
        rgbaMat.release()
        return resultMask
    }
}
