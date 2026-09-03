package de.konradvoelkel.android.autokorrektur.utils

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SmallTest
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs

@RunWith(AndroidJUnit4::class)
@SmallTest
class InstagramExportUtilsInstrumentedTest : AndroidInstrumentedBaseTest() {

    /**
     * Regression test for a real bug: the two existing tests below only ever asserted output
     * dimensions, never pixel content, so a previous version of this export silently shipped
     * with the "AUTOFREI" (after/inpainted) side not actually showing the after-image. Uses
     * solid, maximally distinct colors (not real photos) so pixel comparison is exact and the
     * test is unambiguous about which side is wrong if it fails.
     */
    @Test
    fun createComparisonBitmap_sideBySide_rightHalfShowsAfterImage_leftHalfShowsBeforeImage() {
        val before = Bitmap.createBitmap(200, 200, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.RED)
        }
        val after = Bitmap.createBitmap(200, 200, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.BLUE)
        }

        val result = InstagramExportUtils.createComparisonBitmap(
            beforeBitmap = before,
            afterBitmap = after,
            ratio = InstagramExportUtils.AspectRatio.SQUARE_1_1,
            layout = InstagramExportUtils.LayoutStyle.SIDE_BY_SIDE
        )

        // Sample well inside each half, away from the center divider and the corner badges.
        val leftPixel = result.getPixel(result.width / 4, result.height * 3 / 4)
        val rightPixel = result.getPixel(result.width * 3 / 4, result.height * 3 / 4)

        assertColorMatches("Left half must show the BEFORE (VORHER) image", Color.RED, leftPixel)
        assertColorMatches("Right half must show the AFTER (AUTOFREI / inpainted) image", Color.BLUE, rightPixel)
        assertNotEquals(
            "Left and right halves must show different content, not a duplicate of one side",
            leftPixel,
            rightPixel
        )

        before.recycle()
        after.recycle()
        result.recycle()
    }

    /**
     * Same regression, for the STACKED layout used by the 4:5 portrait export.
     */
    @Test
    fun createComparisonBitmap_stacked_bottomHalfShowsAfterImage_topHalfShowsBeforeImage() {
        val before = Bitmap.createBitmap(200, 200, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.RED)
        }
        val after = Bitmap.createBitmap(200, 200, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.BLUE)
        }

        val result = InstagramExportUtils.createComparisonBitmap(
            beforeBitmap = before,
            afterBitmap = after,
            ratio = InstagramExportUtils.AspectRatio.PORTRAIT_4_5,
            layout = InstagramExportUtils.LayoutStyle.STACKED
        )

        val topPixel = result.getPixel(result.width / 2, result.height / 4)
        val bottomPixel = result.getPixel(result.width / 2, result.height * 3 / 4)

        assertColorMatches("Top half must show the BEFORE (VORHER) image", Color.RED, topPixel)
        assertColorMatches("Bottom half must show the AFTER (AUTOFREI / inpainted) image", Color.BLUE, bottomPixel)

        before.recycle()
        after.recycle()
        result.recycle()
    }

    /**
     * End-to-end version using the real ML pipeline's actual output bitmaps (not synthetic
     * colors), so this also catches a bug in how FirstFragment/InstagramExportBottomSheet wire
     * the pipeline's originalBitmap/inpaintedBitmap into this utility, not just a bug inside the
     * utility itself.
     */
    @Test
    fun createComparisonBitmap_withRealPipelineOutput_rightHalfReflectsActualInpainting() = kotlinx.coroutines.runBlocking {
        val yoloService = de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl(
            de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine(appContext)
        )
        val miGanInference = de.konradvoelkel.android.autokorrektur.ml.MiGanInference(appContext)
        try {
            yoloService.initialize("yolo11s")
            miGanInference.initialize()
            val imageProcessor = de.konradvoelkel.android.autokorrektur.ml.ImageProcessor(appContext)
            val serverSdxlApi = de.konradvoelkel.android.autokorrektur.ml.api.ServerSdxlApi(appContext)
            val pipeline = de.konradvoelkel.android.autokorrektur.pipeline.StaticImagePipeline(
                imageProcessor = imageProcessor,
                yoloService = yoloService,
                miGanInference = miGanInference,
                serverSdxlApi = serverSdxlApi
            )

            val tempFile = cacheAsset("sample_street_with_car.jpg")
            val result = pipeline.processImage(
                uri = android.net.Uri.fromFile(tempFile),
                downscaleMp = null,
                maskUpscale = 1.0f,
                scoreThreshold = 0.5f,
                useServerSdxl = false
            )
            val inpaintBmp = result.inpaintedBitmap
            assertNotNull("Pipeline must produce an inpainted bitmap", inpaintBmp)
            requireNotNull(inpaintBmp)

            // Locate a source pixel that the mask marks as vehicle AND that inpainting actually
            // changed — the same technique MiGanDisplayBitmapPipelineTest already uses to prove
            // inpainting occurred. Only these masked-and-changed pixels are guaranteed to differ
            // between original and inpainted; most of a real street photo is unchanged
            // background, so comparing whole-image halves (an earlier version of this test did)
            // gives a false pass/fail depending on how much of the frame the vehicle occupies.
            val origBmp = result.originalBitmap
            val maskBmp = result.maskBitmap
            var changedSourceX = -1
            var changedSourceY = -1
            outer@ for (y in 0 until minOf(origBmp.height, inpaintBmp.height, maskBmp.height) step 2) {
                for (x in 0 until minOf(origBmp.width, inpaintBmp.width, maskBmp.width) step 2) {
                    val maskPx = Color.red(maskBmp.getPixel(x, y))
                    if (maskPx < 128 && origBmp.getPixel(x, y) != inpaintBmp.getPixel(x, y)) {
                        changedSourceX = x
                        changedSourceY = y
                        break@outer
                    }
                }
            }
            assertTrue(
                "Test setup: sample_street_with_car.jpg must yield at least one masked pixel " +
                    "actually changed by inpainting, otherwise this test can't verify anything",
                changedSourceX >= 0
            )

            val exported = InstagramExportUtils.createComparisonBitmap(
                beforeBitmap = origBmp,
                afterBitmap = inpaintBmp,
                ratio = InstagramExportUtils.AspectRatio.SQUARE_1_1,
                layout = InstagramExportUtils.LayoutStyle.SIDE_BY_SIDE
            )

            // origBmp and inpaintBmp are guaranteed identical in size (asserted elsewhere in the
            // suite), so createComparisonBitmap's private center-crop computes the *same* crop
            // rectangle for both — replicate that here to map our known-changed source pixel
            // into its corresponding position in each exported half.
            val halfWidth = exported.width / 2
            val destAspect = halfWidth.toFloat() / exported.height.toFloat()
            val srcAspect = origBmp.width.toFloat() / origBmp.height.toFloat()
            val cropLeft: Float
            val cropTop: Float
            val cropWidth: Float
            val cropHeight: Float
            if (srcAspect > destAspect) {
                cropHeight = origBmp.height.toFloat()
                cropWidth = cropHeight * destAspect
                cropLeft = (origBmp.width - cropWidth) / 2f
                cropTop = 0f
            } else {
                cropWidth = origBmp.width.toFloat()
                cropHeight = cropWidth / destAspect
                cropLeft = 0f
                cropTop = (origBmp.height - cropHeight) / 2f
            }
            val fx = (changedSourceX - cropLeft) / cropWidth
            val fy = (changedSourceY - cropTop) / cropHeight
            assertTrue(
                "Test setup: the changed source pixel must fall inside the center-crop region " +
                    "actually used by the export, otherwise this test can't verify anything",
                fx in 0.0f..1.0f && fy in 0.0f..1.0f
            )
            val exportedLeftX = (fx * halfWidth).toInt().coerceIn(0, halfWidth - 1)
            val exportedY = (fy * exported.height).toInt().coerceIn(0, exported.height - 1)
            val exportedRightX = (halfWidth + fx * halfWidth).toInt().coerceIn(halfWidth, exported.width - 1)

            val exportedLeftPixel = exported.getPixel(exportedLeftX, exportedY)
            val exportedRightPixel = exported.getPixel(exportedRightX, exportedY)

            assertColorMatches(
                "Exported left (VORHER) half at the known-changed vehicle pixel must show the " +
                    "ORIGINAL color",
                origBmp.getPixel(changedSourceX, changedSourceY),
                exportedLeftPixel,
                tolerance = 40
            )
            assertColorMatches(
                "Exported right (AUTOFREI) half at the known-changed vehicle pixel must show " +
                    "the INPAINTED color, not the original — this is the bug this test guards " +
                    "against: the right side must actually reflect inpainting",
                inpaintBmp.getPixel(changedSourceX, changedSourceY),
                exportedRightPixel,
                tolerance = 40
            )
            assertNotEquals(
                "Exported left and right halves must differ at a pixel inpainting actually changed",
                exportedLeftPixel,
                exportedRightPixel
            )

            exported.recycle()
        } finally {
            yoloService.close()
            miGanInference.close()
        }
    }

    private fun assertColorMatches(message: String, expected: Int, actual: Int, tolerance: Int = 10) {
        val closeEnough = abs(Color.red(expected) - Color.red(actual)) <= tolerance &&
            abs(Color.green(expected) - Color.green(actual)) <= tolerance &&
            abs(Color.blue(expected) - Color.blue(actual)) <= tolerance
        assertTrue(
            "$message (expected ~#${Integer.toHexString(expected)}, got #${Integer.toHexString(actual)})",
            closeEnough
        )
    }

    @Test
    fun createComparisonBitmap_squareOneToOne_generatesValid1080x1080Bitmap() {
        val context = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().context
        val beforeStream = context.assets.open("sample_street_with_car.jpg")
        val afterStream = context.assets.open("sample_street_without_car.jpg")
        val before = android.graphics.BitmapFactory.decodeStream(beforeStream)
        val after = android.graphics.BitmapFactory.decodeStream(afterStream)
        beforeStream.close(); afterStream.close()

        val result = InstagramExportUtils.createComparisonBitmap(
            beforeBitmap = before,
            afterBitmap = after,
            ratio = InstagramExportUtils.AspectRatio.SQUARE_1_1,
            layout = InstagramExportUtils.LayoutStyle.SIDE_BY_SIDE
        )

        assertNotNull("Generated Instagram bitmap should not be null", result)
        assertEquals(1080, result.width)
        assertEquals(1080, result.height)

        val targetFile = java.io.File(androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().targetContext.externalCacheDir, "instagram_square_preview.png")
        java.io.FileOutputStream(targetFile).use { out ->
            result.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, out)
        }
    }

    @Test
    fun createComparisonBitmap_portraitFourFive_generatesValid1080x1350Bitmap() {
        val context = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().context
        val beforeStream = context.assets.open("sample_suburb_with_car.jpg")
        val afterStream = context.assets.open("sample_suburb_without_car.jpg")
        val before = android.graphics.BitmapFactory.decodeStream(beforeStream)
        val after = android.graphics.BitmapFactory.decodeStream(afterStream)
        beforeStream.close(); afterStream.close()

        val result = InstagramExportUtils.createComparisonBitmap(
            beforeBitmap = before,
            afterBitmap = after,
            ratio = InstagramExportUtils.AspectRatio.PORTRAIT_4_5,
            layout = InstagramExportUtils.LayoutStyle.STACKED
        )

        assertNotNull("Generated Instagram bitmap should not be null", result)
        assertEquals(1080, result.width)
        assertEquals(1350, result.height)

        val targetFile = java.io.File(androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().targetContext.externalCacheDir, "instagram_portrait_preview.png")
        java.io.FileOutputStream(targetFile).use { out ->
            result.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, out)
        }
    }
}
