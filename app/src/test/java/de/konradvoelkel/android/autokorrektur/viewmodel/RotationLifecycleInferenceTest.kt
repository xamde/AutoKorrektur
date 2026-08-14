package de.konradvoelkel.android.autokorrektur.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import de.konradvoelkel.android.autokorrektur.MainViewModel
import de.konradvoelkel.android.autokorrektur.pipeline.PipelineResult
import de.konradvoelkel.android.autokorrektur.pipeline.StaticImagePipeline
import de.konradvoelkel.android.autokorrektur.ui.model.MainUiState
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class RotationLifecycleInferenceTest {

    private val testDispatcher = StandardTestDispatcher()
    private lateinit var mockApplication: Application
    private lateinit var mockPipeline: StaticImagePipeline
    private lateinit var mockUri: Uri
    private lateinit var mockBitmap: Bitmap

    @Before
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
        mockkObject(AppLogger)
        every { AppLogger.info(any(), any()) } returns Unit
        every { AppLogger.error(any(), any()) } returns Unit
        every { AppLogger.debug(any(), any()) } returns Unit
        every { AppLogger.warn(any(), any()) } returns Unit

        mockApplication = mockk(relaxed = true)
        mockPipeline = mockk(relaxed = true)
        mockUri = mockk(relaxed = true)
        mockBitmap = mockk(relaxed = true)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun testInferenceStateContinuityAcrossConfigurationChange() = runTest(testDispatcher) {
        val dummyResult = PipelineResult(
            originalBitmap = mockBitmap,
            maskBitmap = mockBitmap,
            inpaintedBitmap = mockBitmap,
            isServerProcessed = false,
            errorMessage = null
        )

        coEvery {
            mockPipeline.processImage(
                uri = any(),
                downscaleMp = any(),
                maskUpscale = any(),
                scoreThreshold = any(),
                useServerSdxl = any(),
                onMaskGenerated = any(),
                onProgressUpdate = any()
            )
        } returns dummyResult

        val viewModel = MainViewModel(mockApplication, mockPipeline)

        // 1. Initial State
        assertEquals(MainUiState.Idle, viewModel.uiState.value)

        // 2. Select Image
        viewModel.setSelectedImageUri(mockUri)
        assertEquals(mockUri, viewModel.properties.value.selectedImageUri)

        // 3. Start Inference (User taps Inpaint)
        viewModel.startInference(
            downscaleMp = 2.0f,
            maskUpscale = 1.05f,
            scoreThreshold = 0.45f,
            useServerSdxl = false
        )

        // Advance coroutines to complete pipeline execution
        advanceUntilIdle()

        // 4. Verify Success State is produced
        val finalState = viewModel.uiState.value
        assertTrue("State should be Success after inference", finalState is MainUiState.Success)
        val successState = finalState as MainUiState.Success
        assertNotNull("Inpainted result bitmap must be present", successState.result.inpaintedBitmap)

        // 5. Simulate Fragment / Activity Rotation (Unbind old view, re-observe StateFlow in new instance)
        val stateAfterRotation = viewModel.uiState.value
        assertTrue("State must remain Success across screen rotation", stateAfterRotation is MainUiState.Success)
        assertEquals(mockBitmap, (stateAfterRotation as MainUiState.Success).result.inpaintedBitmap)

        // 6. Ensure pipeline was executed exactly once (no duplicate re-runs on rotation)
        coVerify(exactly = 1) {
            mockPipeline.processImage(any(), any(), any(), any(), any(), any(), any())
        }
    }

    @Test
    fun testClearStateResetsAllPropertiesAndStateFlow() = runTest(testDispatcher) {
        val viewModel = MainViewModel(mockApplication, mockPipeline)
        viewModel.setSelectedImageUri(mockUri)
        viewModel.setSliderPosition(0.75f)
        viewModel.setBatchMode(true)

        assertEquals(0.75f, viewModel.properties.value.sliderPosition)
        assertTrue(viewModel.properties.value.isBatchMode)

        viewModel.clearState()

        assertEquals(MainUiState.Idle, viewModel.uiState.value)
        assertEquals(0.5f, viewModel.properties.value.sliderPosition)
        assertEquals(false, viewModel.properties.value.isBatchMode)
    }
}
