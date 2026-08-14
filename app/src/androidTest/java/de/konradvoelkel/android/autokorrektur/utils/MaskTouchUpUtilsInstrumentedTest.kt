package de.konradvoelkel.android.autokorrektur.utils

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SmallTest
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
@SmallTest
class MaskTouchUpUtilsInstrumentedTest : AndroidInstrumentedBaseTest() {

    @Test
    fun createDilatedMask_expandsNonZeroRegions() {
        val src = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
        src.eraseColor(Color.BLACK)
        src.setPixel(50, 50, Color.WHITE)

        val dilated = MaskTouchUpUtils.createDilatedMask(src, radiusPx = 5)
        assertNotNull(dilated)
        assertEquals(100, dilated.width)
        assertEquals(100, dilated.height)

        // Pixels around (50, 50) within radius 5 should now be white/non-black
        val centerColor = dilated.getPixel(50, 50)
        val neighborColor = dilated.getPixel(53, 50)
        assertEquals(Color.WHITE, centerColor)
        assertEquals(Color.WHITE, neighborColor)

        // Pixel far away should remain black
        val farColor = dilated.getPixel(10, 10)
        assertEquals(Color.BLACK, farColor)

        src.recycle()
        dilated.recycle()
    }

    @Test
    fun mergeMaskWithStrokes_overlaysBrushStrokes() {
        val baseMask = Bitmap.createBitmap(50, 50, Bitmap.Config.ARGB_8888)
        baseMask.eraseColor(Color.BLACK)

        val strokeBitmap = Bitmap.createBitmap(50, 50, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(strokeBitmap)
        val paint = Paint().apply { color = Color.RED }
        canvas.drawCircle(25f, 25f, 10f, paint)

        val merged = MaskTouchUpUtils.mergeMaskWithStrokes(baseMask, strokeBitmap)
        assertNotNull(merged)
        assertEquals(50, merged.width)
        assertEquals(50, merged.height)

        val centerMerged = merged.getPixel(25, 25)
        assertEquals(Color.RED, centerMerged)

        baseMask.recycle()
        strokeBitmap.recycle()
        merged.recycle()
    }
}
