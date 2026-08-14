package de.konradvoelkel.android.autokorrektur.ui.delegate

import android.graphics.Bitmap
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import de.konradvoelkel.android.autokorrektur.MainActivity
import de.konradvoelkel.android.autokorrektur.model.BatchProcessingResult
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import de.konradvoelkel.android.autokorrektur.utils.ImageExportManager
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class UiDelegateInstrumentedTest : AndroidInstrumentedBaseTest() {

    @Test
    fun testBatchUiDelegate_emptyResults_triggersOnMessage() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                var messageReceived: String? = null
                val exportManager = ImageExportManager(activity)
                val delegate = BatchUiDelegate(activity, exportManager) { message ->
                    messageReceived = message
                }

                delegate.showCsvExportDialog(emptyList())
                assertTrue(messageReceived?.contains("No batch results") == true)
            }
        }
    }

    @Test
    fun testBatchUiDelegate_withResults_showsDialog() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                var messageReceived: String? = null
                val exportManager = ImageExportManager(activity)
                val delegate = BatchUiDelegate(activity, exportManager) { message ->
                    messageReceived = message
                }

                val sampleResults = listOf(
                    BatchProcessingResult(
                        originalImageName = "sample.jpg",
                        processingTimeMs = 120L,
                        maskUpscale = 1.0f,
                        scoreThreshold = 0.25f,
                        downshift = 0.0f,
                        downscaleMp = "2.0 MP",
                        segmentationModel = "yolo11s",
                        success = true
                    )
                )

                delegate.showCsvExportDialog(sampleResults)
            }
        }
    }

    @Test
    fun testInstagramExportDelegate_showExportDialog_showsOptionsDialog() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                var messageReceived: String? = null
                val exportManager = ImageExportManager(activity)
                val delegate = InstagramExportDelegate(activity, exportManager) { message ->
                    messageReceived = message
                }

                val original = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
                val inpainted = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
                try {
                    delegate.showExportDialog(original, inpainted)
                } finally {
                    original.recycle()
                    inpainted.recycle()
                }
            }
        }
    }
}
