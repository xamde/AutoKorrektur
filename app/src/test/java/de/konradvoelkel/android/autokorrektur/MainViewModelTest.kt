package de.konradvoelkel.android.autokorrektur

import android.app.Application
import de.konradvoelkel.android.autokorrektur.pipeline.StaticImagePipeline
import io.mockk.mockk
import org.junit.Assert.assertEquals
import org.junit.Test

class MainViewModelTest {

    @Test
    fun viewModel_sliderPosition_defaultsToCenter() {
        val application = mockk<Application>(relaxed = true)
        val pipeline = mockk<StaticImagePipeline>(relaxed = true)
        val viewModel = MainViewModel(application, pipeline)
        assertEquals(0.5f, viewModel.properties.value.sliderPosition, 0.001f)
    }
}
