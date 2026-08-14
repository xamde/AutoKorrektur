package de.konradvoelkel.android.autokorrektur

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SmallTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
@SmallTest
class MainViewModelInstrumentedTest {

    private val application = ApplicationProvider.getApplicationContext<Application>()

    @Test
    fun viewModel_initialState_isEmpty() {
        val viewModel = MainViewModel(application)
        try {
            val properties = viewModel.properties.value
            assertNull(properties.selectedImageUri)
            assertEquals(0.5f, properties.sliderPosition, 0.001f)
        } finally {
            closeViewModel(viewModel)
        }
    }

    @Test
    fun viewModel_clearState_resetsAllFields() {
        val viewModel = MainViewModel(application)
        try {
            viewModel.setSliderPosition(0.75f)
            viewModel.clearState()

            val properties = viewModel.properties.value
            assertNull(properties.selectedImageUri)
            assertEquals(0.5f, properties.sliderPosition, 0.001f)
        } finally {
            closeViewModel(viewModel)
        }
    }

    private fun closeViewModel(viewModel: MainViewModel) {
        try {
            val method = androidx.lifecycle.ViewModel::class.java.getDeclaredMethod("onCleared")
            method.isAccessible = true
            method.invoke(viewModel)
        } catch (_: Exception) {}
    }
}
