package de.konradvoelkel.android.autokorrektur

import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.pipeline.StaticImagePipeline
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Regression Test Suite specifically catching uninitialized YoloService & StaticImagePipeline usage.
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class UninitializedYoloServiceUsageTest : AndroidInstrumentedBaseTest() {

    @Test
    fun testUninitializedYoloServiceImplAutoInitializesOnInfer() {
        val yoloService = YoloServiceImpl(appContext)

        // 1. Assert isInitialized is false before calling initialize()
        assertFalse("YoloService.isInitialized must be false upon creation", yoloService.isInitialized)

        // 2. Calling inferDetailed without explicit initialize() MUST auto-initialize lazily and succeed
        val dummyMat = org.opencv.core.Mat(640, 640, org.opencv.core.CvType.CV_8UC3)
        try {
            val result = yoloService.inferDetailed(dummyMat, 1.0f, 1.0f, 1.0f)
            assertNotNull("Infer result must not be null post auto-initialization", result)
            assertTrue("YoloService must be initialized after inferDetailed", yoloService.isInitialized)
        } finally {
            dummyMat.release()
            yoloService.close()
        }
    }

    @Test
    fun testStaticPipelineAutoInitializationWhenUninitialized() {
        val pipeline = StaticImagePipeline(appContext)
        assertFalse("Pipeline should not be initialized immediately upon construction", pipeline.isInitialized)

        // Copy a test photo to cache
        val testFile = cacheAsset("sample_street_with_car.jpg")
        val uri = android.net.Uri.fromFile(testFile)

        // Process image directly without calling pipeline.initialize() first.
        // On un-fixed code, this WILL FAIL with "YoloService used before initialize()" because staticPipeline was not auto-initialized!
        val result = kotlinx.coroutines.runBlocking {
            pipeline.processImage(
                uri = uri,
                downscaleMp = 1.0f,
                maskUpscale = 1.05f,
                scoreThreshold = 0.5f,
                useServerSdxl = false
            )
        }

        assertNotNull("Pipeline result should not be null", result)
        assertTrue("Pipeline should be initialized after processing image", pipeline.isInitialized)
        testFile.delete()
    }

    @Test
    fun testStartInferenceInFirstFragmentDoesNotShowUninitializedError() {
        val scenario = ActivityScenario.launch(MainActivity::class.java)
        var testFile: java.io.File? = null

        scenario.onActivity { activity ->
            val navHost = activity.supportFragmentManager.findFragmentById(R.id.nav_host_fragment_content_main)
            val fragment = navHost?.childFragmentManager?.fragments?.firstOrNull() as? FirstFragment
            assertNotNull("FirstFragment must be attached", fragment)

            testFile = cacheAsset("sample_street_with_car.jpg")
            fragment?.javaClass?.getDeclaredField("selectedImageUri")?.apply {
                isAccessible = true
                set(fragment, android.net.Uri.fromFile(testFile))
            }

            val method = fragment?.javaClass?.getDeclaredMethod("performOnnxInference")
            method?.isAccessible = true
            method?.invoke(fragment)
        }

        // Verify that the Snackbar "ML components not initialized. Please restart the app." does NOT exist
        androidx.test.espresso.Espresso.onView(
            androidx.test.espresso.matcher.ViewMatchers.withText("ML components not initialized. Please restart the app.")
        ).check(androidx.test.espresso.assertion.ViewAssertions.doesNotExist())

        testFile?.delete()
        scenario.close()
    }
}
