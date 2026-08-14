package de.konradvoelkel.android.autokorrektur.utils

import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SmallTest
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

@RunWith(AndroidJUnit4::class)
@SmallTest
class UriLoaderInstrumentedTest : AndroidInstrumentedBaseTest() {

    @Test
    fun loadRotatedBitmap_unsupportedSchemeThrowsIOException() {
        val loader = UriLoader(appContext)
        val unsupportedUri = Uri.parse("http://example.com/image.jpg")
        try {
            loader.loadRotatedBitmap(unsupportedUri, maxMegapixels = 2.0f)
            fail("Expected IOException for unsupported URI scheme")
        } catch (e: IOException) {
            assertTrue(e.message?.contains("Unsupported URI scheme") == true)
        }
    }

    @Test
    fun loadRotatedBitmap_loadsFileUriSuccessfully() {
        val loader = UriLoader(appContext)
        val file = File(appContext.cacheDir, "test_file_uri.jpg")
        baseTempFiles.add(file)

        val src = Bitmap.createBitmap(200, 100, Bitmap.Config.ARGB_8888)
        src.eraseColor(Color.BLUE)
        FileOutputStream(file).use { out ->
            src.compress(Bitmap.CompressFormat.JPEG, 90, out)
        }
        src.recycle()

        val loaded = loader.loadRotatedBitmap(Uri.fromFile(file), maxMegapixels = 1.0f)
        assertNotNull(loaded)
        assertEquals(200, loaded.width)
        assertEquals(100, loaded.height)
        loaded.recycle()
    }

    @Test
    fun loadRotatedBitmap_exifRotationApplied() {
        val loader = UriLoader(appContext)
        val file = File(appContext.cacheDir, "test_exif_rotation.jpg")
        baseTempFiles.add(file)

        val src = Bitmap.createBitmap(200, 100, Bitmap.Config.ARGB_8888)
        src.eraseColor(Color.MAGENTA)
        FileOutputStream(file).use { out ->
            src.compress(Bitmap.CompressFormat.JPEG, 90, out)
        }
        src.recycle()

        // Set EXIF orientation to 90 degrees clockwise
        val exif = ExifInterface(file.absolutePath)
        exif.setAttribute(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_ROTATE_90.toString())
        exif.saveAttributes()

        val loaded = loader.loadRotatedBitmap(Uri.fromFile(file), maxMegapixels = 1.0f)
        assertNotNull(loaded)
        // Original 200x100 rotated 90 degrees becomes 100x200
        assertEquals(100, loaded.width)
        assertEquals(200, loaded.height)
        loaded.recycle()
    }
}
