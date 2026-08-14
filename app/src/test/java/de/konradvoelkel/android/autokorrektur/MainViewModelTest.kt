package de.konradvoelkel.android.autokorrektur

import android.app.Application
import android.net.Uri
import de.konradvoelkel.android.autokorrektur.pipeline.StaticImagePipeline
import de.konradvoelkel.android.autokorrektur.ui.model.MainUiState
import io.mockk.mockk
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MainViewModelTest {

    @Test
    fun viewModel_initialState_isIdleWithCenterSlider() {
        val application = mockk<Application>(relaxed = true)
        val pipeline = mockk<StaticImagePipeline>(relaxed = true)
        val viewModel = MainViewModel(application, pipeline)

        assertEquals(MainUiState.Idle, viewModel.uiState.value)
        assertEquals(0.5f, viewModel.properties.value.sliderPosition, 0.001f)
        assertNull(viewModel.properties.value.selectedImageUri)
        assertTrue(viewModel.properties.value.selectedImageUris.isEmpty())
        assertFalse(viewModel.properties.value.isBatchMode)
    }

    @Test
    fun viewModel_setSelectedImageUri_updatesSingleUri() {
        val application = mockk<Application>(relaxed = true)
        val pipeline = mockk<StaticImagePipeline>(relaxed = true)
        val viewModel = MainViewModel(application, pipeline)
        val testUri = mockk<Uri>()

        viewModel.setSelectedImageUri(testUri)

        assertEquals(testUri, viewModel.properties.value.selectedImageUri)
        assertEquals(MainUiState.Idle, viewModel.uiState.value)
    }

    @Test
    fun viewModel_setSelectedImageUris_updatesBatchUris() {
        val application = mockk<Application>(relaxed = true)
        val pipeline = mockk<StaticImagePipeline>(relaxed = true)
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
        val application = mockk<Application>(relaxed = true)
        val pipeline = mockk<StaticImagePipeline>(relaxed = true)
        val viewModel = MainViewModel(application, pipeline)

        viewModel.setSliderPosition(0.8f)
        assertEquals(0.8f, viewModel.properties.value.sliderPosition, 0.001f)

        viewModel.setSliderPosition(0.2f)
        assertEquals(0.2f, viewModel.properties.value.sliderPosition, 0.001f)
    }

    @Test
    fun viewModel_setBatchMode_updatesFlag() {
        val application = mockk<Application>(relaxed = true)
        val pipeline = mockk<StaticImagePipeline>(relaxed = true)
        val viewModel = MainViewModel(application, pipeline)

        viewModel.setBatchMode(true)
        assertTrue(viewModel.properties.value.isBatchMode)

        viewModel.setBatchMode(false)
        assertFalse(viewModel.properties.value.isBatchMode)
    }

    @Test
    fun viewModel_clearState_resetsPropertiesAndUiState() {
        val application = mockk<Application>(relaxed = true)
        val pipeline = mockk<StaticImagePipeline>(relaxed = true)
        val viewModel = MainViewModel(application, pipeline)
        val uri = mockk<Uri>()

        viewModel.setSelectedImageUri(uri)
        viewModel.setSliderPosition(0.9f)

        viewModel.clearState()

        assertEquals(MainUiState.Idle, viewModel.uiState.value)
        assertNull(viewModel.properties.value.selectedImageUri)
        assertEquals(0.5f, viewModel.properties.value.sliderPosition, 0.001f)
    }
}
