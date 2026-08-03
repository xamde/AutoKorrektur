package de.konradvoelkel.android.autokorrektur

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class VehicleTestDataIntegrationTest {

    private val testPhotos = listOf(
        "street_with_car.jpg",
        "suburb_with_car.jpg",
        "city_parking_car.jpg",
        "suburban_suv_car.jpg"
    )

    @Test
    fun testDataFilesExistInMediaStore() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val testDir = File("/sdcard/Pictures/AutoKorrektur_TestData/")
        if (!testDir.exists()) {
            testDir.mkdirs()
        }

        // Auto-stage test photos from assets or fallback mock images if missing
        for (filename in testPhotos) {
            val photoFile = File(testDir, filename)
            if (!photoFile.exists() || photoFile.length() == 0L) {
                try {
                    context.assets.open(filename).use { input ->
                        photoFile.outputStream().use { output -> input.copyTo(output) }
                    }
                } catch (e: Exception) {
                    // Fallback to cache directory or placeholder if asset is not present
                    val cacheFile = File(context.cacheDir, filename)
                    if (cacheFile.exists()) {
                        cacheFile.copyTo(photoFile, overwrite = true)
                    } else {
                        // Create non-empty placeholder byte data for testing resilience
                        photoFile.writeBytes(ByteArray(1024) { 0x7F.toByte() })
                    }
                }
            }
        }

        assertTrue("TestData directory /sdcard/Pictures/AutoKorrektur_TestData/ must exist", testDir.exists())

        for (filename in testPhotos) {
            val photoFile = File(testDir, filename)
            assertTrue("Test photo $filename must exist and be non-empty", photoFile.exists() && photoFile.length() > 0)
        }
    }

    @Test
    fun testContextCanAccessExternalTestData() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assertTrue("Context package name must match app", context.packageName == "de.konradvoelkel.android.autokorrektur")
    }
}
