package de.konradvoelkel.android.autokorrektur

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions.click
import androidx.test.espresso.action.ViewActions.scrollTo
import androidx.test.espresso.assertion.ViewAssertions.doesNotExist
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.isEnabled
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.espresso.matcher.ViewMatchers.withText
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import de.konradvoelkel.android.autokorrektur.shared.AndroidTestUtils
import de.konradvoelkel.android.autokorrektur.shared.PostInpaintingVehicleAssertionUtils
import de.konradvoelkel.android.autokorrektur.ui.model.MainUiState
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream

/**
 * Full End-to-End Emulated UI Test Routine:
 * Simulates complete user journey from selecting very_high_res_car.jpg, verifying UI states,
 * triggering on-device ML inpainting, and validating simultaneous Mask Preview and Before/After slider.
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class FullEmulatedUiInferenceE2ETest : AndroidInstrumentedBaseTest() {

    private val tempFiles = mutableListOf<File>()

    @Before
    fun setUp() {
        val prefs = appContext.getSharedPreferences("autokorrektur_prefs", Context.MODE_PRIVATE)
        prefs.edit().clear().commit()
    }

    @After
    fun tearDown() {
        tempFiles.forEach { it.delete() }
        System.gc()
    }

    @Test
    fun testFullUiWorkflow_withVeryHighResCar_showsMaskAndSliderAndRemovesCar() = runBlocking {
        // 1. Prepare asset
        val imageFile = AndroidTestUtils.copyAssetToCache(appContext, "very_high_res_car.jpg")
        tempFiles.add(imageFile)
        val imageUri = Uri.fromFile(imageFile)

        // 2. Launch MainActivity
        val scenario = ActivityScenario.launch(MainActivity::class.java)

        var viewModel: MainViewModel? = null
        scenario.onActivity { activity ->
            viewModel = ViewModelProvider(activity)[MainViewModel::class.java]
            viewModel?.setSelectedImageUri(imageUri)
        }

        assertNotNull("ViewModel must be available", viewModel)
        val vm = viewModel!!

        // 3. Verify Selected Image UI State
        onView(withId(R.id.fileSelect)).perform(scrollTo()).check(matches(isDisplayed()))
        onView(withId(R.id.startInference)).perform(scrollTo()).check(matches(isDisplayed()))
        onView(withId(R.id.startInference)).check(matches(isEnabled()))

        // Capture initial UI state screenshot
        saveScreenCapture("e2e_01_image_selected.png")

        // 4. Trigger Inpainting Inference
        AppLogger.info("FullEmulatedUiInferenceE2ETest: Clicking Start button...")
        onView(withId(R.id.startInference)).perform(scrollTo(), click())

        // 5. Await Inpainting Completion (timeout 60s)
        val timeoutMs = 60_000L
        val startMs = System.currentTimeMillis()
        var completedState: MainUiState? = null

        while (System.currentTimeMillis() - startMs < timeoutMs) {
            val current = vm.uiState.value
            if (current is MainUiState.Success || current is MainUiState.Error) {
                completedState = current
                break
            }
            delay(500)
        }

        assertNotNull("Inference should complete within $timeoutMs ms", completedState)
        if (completedState is MainUiState.Error) {
            fail("Inference failed with error: ${(completedState as MainUiState.Error).message}")
        }

        assertTrue("State must be Success", completedState is MainUiState.Success)
        val successResult = (completedState as MainUiState.Success).result

        // 6. Verify Visual UI Components Displayed
        // Assert Before/After Slider View is visible
        onView(withId(R.id.beforeAfterSliderView)).perform(scrollTo()).check(matches(isDisplayed()))

        // Assert Mask Preview Container is visible
        onView(withId(R.id.imagesContainer)).perform(scrollTo()).check(matches(isDisplayed()))
        onView(withText(R.string.label_mask)).perform(scrollTo()).check(matches(isDisplayed()))

        // Assert No Error Snackbars
        onView(withId(com.google.android.material.R.id.snackbar_text)).check(doesNotExist())

        // Capture final UI state screenshot showing both Mask and Slider
        saveScreenCapture("e2e_02_inference_success_with_mask_and_slider.png")

        // Save result bitmaps to /sdcard/Download for debugging
        try {
            val maskFile = File("/sdcard/Download/e2e_mask_bitmap.png")
            FileOutputStream(maskFile).use { successResult.maskBitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
            successResult.inpaintedBitmap?.let { bmp ->
                val inpaintFile = File("/sdcard/Download/e2e_inpainted_bitmap.png")
                FileOutputStream(inpaintFile).use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
            }
        } catch (_: Exception) {}

        // 7. Verify Inpainting Processing and Blending Fidelity
        assertNotNull("Inpainted bitmap must not be null", successResult.inpaintedBitmap)
        val origBmp = successResult.originalBitmap
        val inpaintBmp = successResult.inpaintedBitmap!!
        val maskBmp = successResult.maskBitmap

        assertTrue("Inpainted bitmap must have valid dimensions", inpaintBmp.width > 0 && inpaintBmp.height > 0)
        assertTrue("Mask bitmap must have valid dimensions", maskBmp.width > 0 && maskBmp.height > 0)

        // Sample pixels to verify that vehicle area was inpainted and background was preserved
        var maskedPixelsChanged = 0
        var totalMaskedPixels = 0
        val sampleStep = 8

        for (y in 0 until maskBmp.height step sampleStep) {
            for (x in 0 until maskBmp.width step sampleStep) {
                val maskPixel = maskBmp.getPixel(x, y) and 0xFF
                // In maskBitmap: 0 is masked vehicle hole, 255 is background
                if (maskPixel < 128) {
                    totalMaskedPixels++
                    val origPixel = origBmp.getPixel(x.coerceAtMost(origBmp.width - 1), y.coerceAtMost(origBmp.height - 1))
                    val inpaintPixel = inpaintBmp.getPixel(x.coerceAtMost(inpaintBmp.width - 1), y.coerceAtMost(inpaintBmp.height - 1))
                    if (origPixel != inpaintPixel) {
                        maskedPixelsChanged++
                    }
                }
            }
        }

        assertTrue("Mask must detect vehicle region", totalMaskedPixels > 100)
        val changeRatio = maskedPixelsChanged.toFloat() / totalMaskedPixels.toFloat()
        AppLogger.info("E2E Inpainting verification: maskedPixelsChanged=$maskedPixelsChanged, totalMaskedPixels=$totalMaskedPixels, changeRatio=$changeRatio")
        assertTrue("Inpainting must modify pixels in masked vehicle area (changeRatio=$changeRatio)", changeRatio > 0.80f)

        scenario.close()
    }

    private fun saveScreenCapture(fileName: String) {
        try {
            val outFile = File("/sdcard/Download", fileName)
            outFile.parentFile?.mkdirs()
            val uiDevice = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().uiAutomation
            val screenshot = uiDevice.takeScreenshot()
            if (screenshot != null) {
                FileOutputStream(outFile).use { fos ->
                    screenshot.compress(Bitmap.CompressFormat.PNG, 100, fos)
                }
                screenshot.recycle()
                AppLogger.info("Saved E2E screencap to: ${outFile.absolutePath}")
            }
        } catch (e: Exception) {
            AppLogger.warn("Failed to capture screenshot $fileName: ${e.message}")
        }
    }
}
