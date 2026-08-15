package de.konradvoelkel.android.autokorrektur

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import de.konradvoelkel.android.autokorrektur.pipeline.PipelineResult
import de.konradvoelkel.android.autokorrektur.pipeline.StaticImagePipeline
import de.konradvoelkel.android.autokorrektur.ui.model.MainUiState
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.unmockkAll
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class MainViewModelTest {

    private val testDispatcher = StandardTestDispatcher()
    private lateinit var application: Application
    private lateinit var pipeline: StaticImagePipeline

    @Before
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
        mockkObject(AppLogger)
        every { AppLogger.info(any(), any()) } returns Unit
        every { AppLogger.error(any(), any()) } returns Unit
        every { AppLogger.debug(any(), any()) } returns Unit
        every { AppLogger.warn(any(), any()) } returns Unit

        application = mockk(relaxed = true)
        pipeline = mockk(relaxed = true)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
        unmockkAll()
    }

    @Test
    fun viewModel_initialState_isIdleWithCenterSlider() {
        val viewModel = MainViewModel(application, pipeline)

        assertEquals(MainUiState.Idle, viewModel.uiState.value)
        assertEquals(0.5f, viewModel.properties.value.sliderPosition, 0.001f)
        assertNull(viewModel.properties.value.selectedImageUri)
        assertTrue(viewModel.properties.value.selectedImageUris.isEmpty())
        assertFalse(viewModel.properties.value.isBatchMode)
    }

    @Test
    fun viewModel_setSelectedImageUri_updatesSingleUri() {
        val viewModel = MainViewModel(application, pipeline)
        val testUri = mockk<Uri>()

        viewModel.setSelectedImageUri(testUri)

        assertEquals(testUri, viewModel.properties.value.selectedImageUri)
        assertEquals(MainUiState.Idle, viewModel.uiState.value)
    }

    @Test
    fun viewModel_setSelectedImageUris_updatesBatchUris() {
        val viewModel = MainViewModel(application, pipeline)
        val uri1 = mockk<Uri>()
        val uri2 = mockk<Uri>()
        val uriList = listOf(uri1, uri2)

        viewModel.setSelectedImageUris(uriList)
        viewModel.setBatchMode(true)

        assertEquals(uriList, viewModel.properties.value.selectedImageUris)
        assertTrue(viewModel.properties.value.isBatchMode)
    }

    @Test
    fun viewModel_setSliderPosition_updatesSliderPosition() {
        val viewModel = MainViewModel(application, pipeline)

        viewModel.setSliderPosition(0.8f)
        assertEquals(0.8f, viewModel.properties.value.sliderPosition, 0.001f)

        viewModel.setSliderPosition(0.2f)
        assertEquals(0.2f, viewModel.properties.value.sliderPosition, 0.001f)
    }

    @Test
    fun viewModel_setBatchMode_updatesFlag() {
        val viewModel = MainViewModel(application, pipeline)

        viewModel.setBatchMode(true)
        assertTrue(viewModel.properties.value.isBatchMode)

        viewModel.setBatchMode(false)
        assertFalse(viewModel.properties.value.isBatchMode)
    }

    @Test
    fun viewModel_clearState_resetsPropertiesAndUiState() {
        val viewModel = MainViewModel(application, pipeline)
        val uri = mockk<Uri>()

        viewModel.setSelectedImageUri(uri)
        viewModel.setSliderPosition(0.9f)

        viewModel.clearState()

        assertEquals(MainUiState.Idle, viewModel.uiState.value)
        assertNull(viewModel.properties.value.selectedImageUri)
        assertEquals(0.5f, viewModel.properties.value.sliderPosition, 0.001f)
    }

    @Test
    fun viewModel_startInference_singleSuccess_transitionsToSuccessState() = runTest(testDispatcher) {
        val viewModel = MainViewModel(application, pipeline)
        val uri = mockk<Uri>()
        val dummyOriginal = mockk<Bitmap>(relaxed = true)
        val dummyMask = mockk<Bitmap>(relaxed = true)
        val dummyInpainted = mockk<Bitmap>(relaxed = true)

        val expectedResult = PipelineResult(
            originalBitmap = dummyOriginal,
            maskBitmap = dummyMask,
            inpaintedBitmap = dummyInpainted,
            isServerProcessed = false
        )

        coEvery {
            pipeline.processImage(
                uri = uri,
                downscaleMp = null,
                maskUpscale = 1.0f,
                scoreThreshold = 0.25f,
                useServerSdxl = false,
                qualityMode = any(),
                onProgressUpdate = any(),
                onIntermediateInpaintUpdate = any()
            )
        } returns expectedResult

        viewModel.setSelectedImageUri(uri)
        viewModel.startInference(
            downscaleMp = null,
            maskUpscale = 1.0f,
            scoreThreshold = 0.25f,
            useServerSdxl = false
        )

        advanceUntilIdle()

        val state = viewModel.uiState.value
        assertTrue("Expected Success state, got $state", state is MainUiState.Success)
        val successState = state as MainUiState.Success
        assertEquals(expectedResult, successState.result)
    }

    @Test
    fun viewModel_startInference_pipelineError_transitionsToErrorState() = runTest(testDispatcher) {
        val viewModel = MainViewModel(application, pipeline)
        val uri = mockk<Uri>()

        coEvery {
            pipeline.processImage(
                uri = uri,
                downscaleMp = null,
                maskUpscale = 1.0f,
                scoreThreshold = 0.25f,
                useServerSdxl = false,
                qualityMode = any(),
                onProgressUpdate = any(),
                onIntermediateInpaintUpdate = any()
            )
        } returns PipelineResult(
            originalBitmap = mockk(relaxed = true),
            maskBitmap = mockk(relaxed = true),
            inpaintedBitmap = null,
            errorMessage = "Model execution failed"
        )

        viewModel.setSelectedImageUri(uri)
        viewModel.startInference(
            downscaleMp = null,
            maskUpscale = 1.0f,
            scoreThreshold = 0.25f,
            useServerSdxl = false
        )

        advanceUntilIdle()

        val state = viewModel.uiState.value
        assertTrue("Expected Error state, got $state", state is MainUiState.Error)
        val errorState = state as MainUiState.Error
        assertEquals("Model execution failed", errorState.message)
    }
}
