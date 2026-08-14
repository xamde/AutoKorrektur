package de.konradvoelkel.android.autokorrektur.ml

import android.graphics.Bitmap
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

@RunWith(AndroidJUnit4::class)
class ImageProcessorInstrumentedTest : AndroidInstrumentedBaseTest() {

    @Test
    fun testProcessInputImage_validBitmapUri_returnsProcessedImage() {
        val testFile = File(appContext.cacheDir, "test_input_${System.currentTimeMillis()}.png")
        baseTempFiles.add(testFile)
        val bmp = Bitmap.createBitmap(800, 600, Bitmap.Config.ARGB_8888)
        val fos = FileOutputStream(testFile)
        try {
            bmp.compress(Bitmap.CompressFormat.PNG, 100, fos)
        } finally {
            fos.close()
            bmp.recycle()
        }

        val processor = ImageProcessor(appContext)
        val processed = processor.processInputImage(
            imageUri = Uri.fromFile(testFile),
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )

        try {
            assertNotNull(processed.originalBitmap)
            assertNotNull(processed.transformedBitmap)
            assertEquals(800, processed.originalBitmap.width)
            assertEquals(600, processed.originalBitmap.height)
            assertEquals(640, processed.transformedBitmap.width)
            assertEquals(640, processed.transformedBitmap.height)
            assertTrue(processed.xRatio > 0f)
            assertTrue(processed.yRatio > 0f)
        } finally {
            processed.release()
        }
    }

    @Test(expected = IOException::class)
    fun testProcessInputImage_nonExistentUri_throwsIOException() {
        val processor = ImageProcessor(appContext)
        processor.processInputImage(
            imageUri = Uri.parse("file:///non_existent_path_${System.currentTimeMillis()}.jpg"),
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )
    }

    @Test
    fun testProcessInputImage_withDownscaleMegapixels_limitsResolution() {
        val testFile = File(appContext.cacheDir, "test_large_${System.currentTimeMillis()}.png")
        baseTempFiles.add(testFile)
        val largeBmp = Bitmap.createBitmap(2000, 2000, Bitmap.Config.ARGB_8888)
        val fos = FileOutputStream(testFile)
        try {
            largeBmp.compress(Bitmap.CompressFormat.PNG, 100, fos)
        } finally {
            fos.close()
            largeBmp.recycle()
        }

        val processor = ImageProcessor(appContext)
        val processed = processor.processInputImage(
            imageUri = Uri.fromFile(testFile),
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = 1.0f // Max 1 MP
        )

        try {
            assertNotNull(processed.originalBitmap)
            assertNotNull(processed.transformedBitmap)
            val mp = (processed.originalMat.cols() * processed.originalMat.rows()) / 1_000_000f
            assertTrue("Processed matrix should be scaled near 1.0 MP, got $mp MP", mp <= 1.2f)
        } finally {
            processed.release()
        }
    }
}
