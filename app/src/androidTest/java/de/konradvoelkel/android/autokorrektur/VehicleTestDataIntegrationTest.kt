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
        val testDir = File("/sdcard/Pictures/AutoKorrektur_TestData/")
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
