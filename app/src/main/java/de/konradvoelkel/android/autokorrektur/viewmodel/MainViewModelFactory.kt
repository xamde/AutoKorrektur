package de.konradvoelkel.android.autokorrektur.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService

class MainViewModelFactory(
    private val imageProcessor: ImageProcessor,
    private val yoloInference: YoloService,
    private val miGanInference: MiGanInference
) : ViewModelProvider.Factory {

    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return MainViewModel(imageProcessor, yoloInference, miGanInference) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
