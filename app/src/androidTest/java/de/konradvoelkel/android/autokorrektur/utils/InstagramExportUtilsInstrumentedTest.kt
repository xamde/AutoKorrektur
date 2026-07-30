package de.konradvoelkel.android.autokorrektur.utils

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SmallTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
@SmallTest
class InstagramExportUtilsInstrumentedTest {

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
