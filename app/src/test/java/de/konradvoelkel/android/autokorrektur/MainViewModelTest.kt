package de.konradvoelkel.android.autokorrektur

import org.junit.Assert.assertEquals
import org.junit.Test

class MainViewModelTest {

    @Test
    fun viewModel_sliderPosition_defaultsToCenter() {
        val viewModel = MainViewModel()
        assertEquals(0.5f, viewModel.sliderPosition, 0.001f)
    }
}
