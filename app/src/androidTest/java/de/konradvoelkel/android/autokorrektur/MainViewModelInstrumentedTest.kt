package de.konradvoelkel.android.autokorrektur

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SmallTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
@SmallTest
class MainViewModelInstrumentedTest {

    @Test
    fun viewModel_initialState_isEmpty() {
        val viewModel = MainViewModel()
        assertNull(viewModel.selectedImageUri.value)
        assertNull(viewModel.processedImageUri.value)
        assertNull(viewModel.processedBitmap)
        assertEquals(0.5f, viewModel.sliderPosition, 0.001f)
    }

    @Test
    fun viewModel_clearState_resetsAllFields() {
        val viewModel = MainViewModel()
        viewModel.sliderPosition = 0.75f
        viewModel.clearState()

        assertNull(viewModel.selectedImageUri.value)
        assertNull(viewModel.processedImageUri.value)
        assertNull(viewModel.processedBitmap)
        assertEquals(0.5f, viewModel.sliderPosition, 0.001f)
    }
}
