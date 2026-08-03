package de.konradvoelkel.android.autokorrektur.ml.mask

import kotlin.math.max
import kotlin.math.min

/**
 * Pure Kotlin math for YOLO mask operations.
 * Separated from OpenCV dependencies to allow JVM unit testing.
 */
object YoloMaskMath {

    data class IntRect(val x: Int, val y: Int, val width: Int, val height: Int)

    data class MaskPlacement(
        val dst: IntRect,
        val src: IntRect
    )

    /**
     * Calculates the crop rectangle in the prototype grid for a given normalized bounding box.
     */
    fun calculateCropRect(
        boxX: Float,
        boxY: Float,
        boxW: Float,
        boxH: Float,
        protoW: Int,
        protoH: Int
    ): IntRect {
        val cropX = (boxX * protoW).toInt().coerceIn(0, protoW - 1)
        val cropY = (boxY * protoH).toInt().coerceIn(0, protoH - 1)
        val cropW = (boxW * protoW).toInt().coerceAtLeast(1).coerceAtMost(protoW - cropX)
        val cropH = (boxH * protoH).toInt().coerceAtLeast(1).coerceAtMost(protoH - cropY)
        return IntRect(cropX, cropY, cropW, cropH)
    }

    /**
     * Calculates where to place the upscaled mask on the final overlay.
     */
    fun calculatePlacement(
        boxX: Float,
        boxY: Float,
        boxW: Float,
        boxH: Float,
        maskW: Int,
        maskH: Int,
        inputW: Int,
        inputH: Int
    ): MaskPlacement {
        val xModel = (boxX * inputW).toInt()
        val yModel = (boxY * inputH).toInt()
        val wModel = (boxW * inputW).toInt()
        val hModel = (boxH * inputH).toInt()

        val targetX = xModel + (wModel / 2.0) - (maskW / 2.0)
        val targetY = yModel + (hModel / 2.0) - (maskH / 2.0)

        val dstX = max(0, targetX.toInt())
        val dstY = max(0, targetY.toInt())
        val dstW = min(maskW, inputW - dstX)
        val dstH = min(maskH, inputH - dstY)

        val srcW = min(maskW, dstW)
        val srcH = min(maskH, dstH)

        return MaskPlacement(
            dst = IntRect(dstX, dstY, dstW, dstH),
            src = IntRect(0, 0, srcW, srcH)
        )
    }
}
