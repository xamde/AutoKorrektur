package de.konradvoelkel.android.autokorrektur.utils

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SmallTest
import de.konradvoelkel.android.autokorrektur.model.BatchProcessingResult
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
@SmallTest
class ImageExportManagerInstrumentedTest : AndroidInstrumentedBaseTest() {

    @Test
    fun exportBatchResultsToCSV_emptyListReturnsNull() {
        val manager = ImageExportManager(appContext)
        val file = manager.exportBatchResultsToCSV(emptyList())
        assertNull(file)
    }

    @Test
    fun exportBatchResultsToCSV_validListExportsFile() {
        val manager = ImageExportManager(appContext)
        val results = listOf(
            BatchProcessingResult(
                originalImageName = "test_car.jpg",
                processingTimeMs = 1250,
                maskUpscale = 1.05f,
                scoreThreshold = 0.25f,
                downshift = 0.0f,
                downscaleMp = "1.5",
                segmentationModel = "yolo11s",
                success = true,
                errorMessage = null
            ),
            BatchProcessingResult(
                originalImageName = "error,car.jpg",
                processingTimeMs = 500,
                maskUpscale = 1.0f,
                scoreThreshold = 0.3f,
                downshift = 0.0f,
                downscaleMp = "original",
                segmentationModel = "yolo11n",
                success = false,
                errorMessage = "Quota \"exceeded\"\nretry later"
            )
        )

        val csvFile = manager.exportBatchResultsToCSV(results)
        assertNotNull(csvFile)
        assertTrue(csvFile!!.exists())
        baseTempFiles.add(csvFile)

        val content = csvFile.readText()
        assertTrue(content.contains("test_car.jpg"))
        assertTrue(content.contains("\"error,car.jpg\""))
        assertTrue(content.contains("\"Quota \"\"exceeded\"\"\nretry later\""))
    }

    @Test
    fun saveImageToGallery_createsValidImageUri() {
        val manager = ImageExportManager(appContext)
        val bitmap = Bitmap.createBitmap(50, 50, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(Color.CYAN)

        val uri = manager.saveImageToGallery(bitmap)
        assertNotNull("Image URI should be generated when saved to gallery", uri)

        bitmap.recycle()
    }
}
