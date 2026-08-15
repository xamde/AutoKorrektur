package de.konradvoelkel.android.autokorrektur.ml.progressive

import android.graphics.Bitmap
import de.konradvoelkel.android.autokorrektur.ml.InpaintingEngine
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

/**
 * High-resolution progressive tile-based neural inpainter.
 * Decomposes high-resolution car regions into contextual tiles, executes multi-pass
 * neural synthesis with boundary feathering, and streams intermediate progress to the UI.
 */
class ProgressiveTileInpainter(
    private val inpaintingEngine: InpaintingEngine
) {

    /**
     * Executes progressive high-resolution inpainting on [fullImageMat] guided by [subtractiveMaskMat].
     *
     * @param fullImageMat High-resolution original image (CV_8UC3 or CV_8UC4).
     * @param subtractiveMaskMat Binary subtractive mask (0 = vehicle region, 255 = keep background).
     * @param onProgress Callback invoked with (stage, percent, intermediateBitmap).
     * @return Fully inpainted high-resolution Mat (caller must release).
     */
    suspend fun inpaintProgressive(
        fullImageMat: Mat,
        subtractiveMaskMat: Mat,
        onProgress: ((stage: String, percent: Int, intermediateBitmap: Bitmap?) -> Unit)? = null
    ): Mat {
        currentCoroutineContext().ensureActive()
        val origW = fullImageMat.cols()
        val origH = fullImageMat.rows()

        // 1. Invert mask to get binary vehicle region (255 = car, 0 = bg)
        val carHoleMask = Mat()
        Core.bitwise_not(subtractiveMaskMat, carHoleMask)

        val nonZeroCount = Core.countNonZero(carHoleMask)
        if (nonZeroCount == 0) {
            carHoleMask.release()
            return fullImageMat.clone()
        }

        onProgress?.invoke("Extracting Vehicle Regions", 5, null)

        // 2. Find contours and bounding boxes of all detected vehicles
        val contours = java.util.ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        val maskCopy = carHoleMask.clone()
        Imgproc.findContours(
            maskCopy,
            contours,
            hierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        maskCopy.release()
        hierarchy.release()

        val outputMat = fullImageMat.clone()
        val matsToRelease = mutableListOf<Mat>()

        try {
            // Filter significant vehicle regions
            val minArea = (origW * origH * 0.0005).toInt().coerceAtLeast(100)
            val boundingBoxes = mutableListOf<Rect>()
            for (i in 0 until contours.size) {
                val c = contours[i]
                val rect = computeBoundingBox(c)
                if (rect.width * rect.height >= minArea) {
                    boundingBoxes.add(rect)
                }
                c.release()
            }

            if (boundingBoxes.isEmpty()) {
                // Fallback to full inpainting
                val directResult = inpaintingEngine.inpaint(fullImageMat, subtractiveMaskMat)
                carHoleMask.release()
                return directResult
            }

            // Merge closely overlapping bounding boxes
            val mergedBoxes = mergeBoundingBoxes(boundingBoxes, paddingRatio = 0.20f, imgW = origW, imgH = origH)
            val totalBoxes = mergedBoxes.size

            onProgress?.invoke("Progressive Neural Inpainting (0/$totalBoxes)", 15, null)

            mergedBoxes.forEachIndexed { index, bbox ->
                currentCoroutineContext().ensureActive()

                val progressStart = 15 + (index * 75 / totalBoxes)
                val progressEnd = 15 + ((index + 1) * 75 / totalBoxes)

                onProgress?.invoke("Inpainting Region ${index + 1}/$totalBoxes", progressStart, null)

                // Extract ROI from image and subtractive mask
                val imageRoi = Mat(outputMat, bbox)
                val maskRoi = Mat(subtractiveMaskMat, bbox)

                // Execute neural inpainting on ROI
                val inpaintedRoi = inpaintingEngine.inpaint(imageRoi, maskRoi)

                // Create feathered alpha mask along ROI borders for seamless blending
                val blendMask = createFeatheredMask(bbox.width, bbox.height, featherPx = 16)

                // Blend inpainted ROI into outputMat
                blendRoi(outputMat, inpaintedRoi, bbox, blendMask)

                imageRoi.release()
                maskRoi.release()
                inpaintedRoi.release()
                blendMask.release()

                // Generate intermediate preview bitmap
                val previewBitmap = Bitmap.createBitmap(origW, origH, Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(outputMat, previewBitmap)
                onProgress?.invoke("Refining Textures (${index + 1}/$totalBoxes)", progressEnd, previewBitmap)
            }

            onProgress?.invoke("Finalizing High-Res Output", 95, null)
            return outputMat
        } catch (e: Exception) {
            AppLogger.error("ProgressiveTileInpainter error, applying fallback", e)
            outputMat.release()
            return inpaintingEngine.inpaint(fullImageMat, subtractiveMaskMat)
        } finally {
            carHoleMask.release()
            matsToRelease.forEach { it.release() }
        }
    }

    companion object {
        /**
         * Computes bounding rectangle from contour point list.
         */
        fun computeBoundingBox(contour: MatOfPoint): Rect {
            val pts = contour.toArray()
            if (pts.isEmpty()) return Rect(0, 0, 0, 0)
            var minX = pts[0].x.toInt()
            var maxX = pts[0].x.toInt()
            var minY = pts[0].y.toInt()
            var maxY = pts[0].y.toInt()
            for (p in pts) {
                val px = p.x.toInt()
                val py = p.y.toInt()
                if (px < minX) minX = px
                if (px > maxX) maxX = px
                if (py < minY) minY = py
                if (py > maxY) maxY = py
            }
            return Rect(minX, minY, max(1, maxX - minX + 1), max(1, maxY - minY + 1))
        }

        /**
         * Expands bounding boxes with contextual padding and merges overlapping rectangles.
         */
        fun mergeBoundingBoxes(
            boxes: List<Rect>,
            paddingRatio: Float = 0.20f,
            imgW: Int,
            imgH: Int
        ): List<Rect> {
            if (boxes.isEmpty()) return emptyList()

            val expanded = boxes.map { box ->
                val padX = (box.width * paddingRatio).toInt()
                val padY = (box.height * paddingRatio).toInt()
                val x = (box.x - padX).coerceIn(0, imgW - 1)
                val y = (box.y - padY).coerceIn(0, imgH - 1)
                val w = (box.width + 2 * padX).coerceAtMost(imgW - x)
                val h = (box.height + 2 * padY).coerceAtMost(imgH - y)
                Rect(x, y, w, h)
            }

            val merged = mutableListOf<Rect>()
            expanded.forEach { current ->
                var mergedWithExisting = false
                for (i in merged.indices) {
                    val existing = merged[i]
                    val intersection = rectIntersection(existing, current)
                    if (intersection != null) {
                        merged[i] = rectUnion(existing, current)
                        mergedWithExisting = true
                        break
                    }
                }
                if (!mergedWithExisting) {
                    merged.add(current)
                }
            }
            return merged
        }

        private fun rectIntersection(r1: Rect, r2: Rect): Rect? {
            val x = max(r1.x, r2.x)
            val y = max(r1.y, r2.y)
            val w = min(r1.x + r1.width, r2.x + r2.width) - x
            val h = min(r1.y + r1.height, r2.y + r2.height) - y
            return if (w > 0 && h > 0) Rect(x, y, w, h) else null
        }

        private fun rectUnion(r1: Rect, r2: Rect): Rect {
            val x = min(r1.x, r2.x)
            val y = min(r1.y, r2.y)
            val w = max(r1.x + r1.width, r2.x + r2.width) - x
            val h = max(r1.y + r1.height, r2.y + r2.height) - y
            return Rect(x, y, w, h)
        }

        /**
         * Creates a 1-channel CV_8UC1 feathered blend mask with soft edges to eliminate seam artifacts.
         */
        fun createFeatheredMask(width: Int, height: Int, featherPx: Int = 16): Mat {
            val feather = featherPx.coerceAtMost(min(width, height) / 4).coerceAtLeast(1)

            // Draw border inset rectangle with Gaussian feathering
            val insetRect = Rect(feather, feather, width - 2 * feather, height - 2 * feather)
            val baseMat = Mat.zeros(height, width, CvType.CV_8UC1)
            val whiteRoi = baseMat.submat(insetRect)
            whiteRoi.setTo(Scalar(255.0))
            whiteRoi.release()

            val blurred = Mat()
            val kSize = (feather * 2 + 1)
            Imgproc.GaussianBlur(baseMat, blurred, Size(kSize.toDouble(), kSize.toDouble()), 0.0)
            baseMat.release()
            return blurred
        }

        private fun blendRoi(dst: Mat, src: Mat, roiRect: Rect, featherMask: Mat) {
            val dstRoi = dst.submat(roiRect)
            val srcMatching = if (src.size() != dstRoi.size() || src.type() != dstRoi.type()) {
                val resized = Mat()
                Imgproc.resize(src, resized, dstRoi.size())
                if (resized.type() != dstRoi.type()) {
                    val converted = Mat()
                    if (dstRoi.channels() == 4 && resized.channels() == 3) {
                        Imgproc.cvtColor(resized, converted, Imgproc.COLOR_RGB2RGBA)
                    } else if (dstRoi.channels() == 3 && resized.channels() == 4) {
                        Imgproc.cvtColor(resized, converted, Imgproc.COLOR_RGBA2RGB)
                    } else {
                        resized.convertTo(converted, dstRoi.type())
                    }
                    resized.release()
                    converted
                } else {
                    resized
                }
            } else {
                src
            }

            srcMatching.copyTo(dstRoi, featherMask)

            if (srcMatching !== src) {
                srcMatching.release()
            }
            dstRoi.release()
        }
    }
}
