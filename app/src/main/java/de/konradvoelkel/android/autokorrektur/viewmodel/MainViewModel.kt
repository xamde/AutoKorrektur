package de.konradvoelkel.android.autokorrektur.viewmodel

import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Mat

class MainViewModel(
    private val imageProcessor: ImageProcessor,
    private val yoloInference: YoloService,
    private val miGanInference: MiGanInference
) : ViewModel() {

    private val _uiState = MutableLiveData<UiState>()
    val uiState: LiveData<UiState> = _uiState

    private val _processing = MutableLiveData<Boolean>()
    val processing: LiveData<Boolean> = _processing

    private val _batchUiState = MutableLiveData<BatchUiState>()
    val batchUiState: LiveData<BatchUiState> = _batchUiState

    fun performBatchProcessing(
        uris: List<Uri>,
        segModel: String,
        downscaleMp: Float?,
        maskUpscale: Float,
        scoreThreshold: Float,
    ) {
        viewModelScope.launch {
            _batchUiState.value = BatchUiState.Progress(0, uris.size)
            val results = mutableListOf<Bitmap>()
            try {
                for ((index, uri) in uris.withIndex()) {
                    val result = withContext(Dispatchers.IO) {
                        inference(
                            uri,
                            null,
                            false,
                            segModel,
                            downscaleMp,
                            maskUpscale,
                            scoreThreshold
                        )
                    }
                    if (result is UiState.Success) {
                        results.add(result.processedBitmap)
                    }
                    _batchUiState.value = BatchUiState.Progress(index + 1, uris.size)
                }
                _batchUiState.value = BatchUiState.Success(results)
            } catch (e: Exception) {
                _batchUiState.value = BatchUiState.Error(e.message ?: "An unknown error occurred")
            }
        }
    }

    fun performOnnxInference(
        selectedImageUri: Uri?,
        resultImageUri: Uri?,
        continueWithResult: Boolean,
        segModel: String,
        downscaleMp: Float?,
        maskUpscale: Float,
        scoreThreshold: Float,
    ) {
        viewModelScope.launch {
            _processing.value = true
            try {
                val result = withContext(Dispatchers.IO) {
                    inference(
                        selectedImageUri,
                        resultImageUri,
                        continueWithResult,
                        segModel,
                        downscaleMp,
                        maskUpscale,
                        scoreThreshold
                    )
                }
                _uiState.value = result
            } catch (e: Exception) {
                _uiState.value = UiState.Error(e.message ?: "An unknown error occurred")
            } finally {
                _processing.value = false
            }
        }
    }

    private suspend fun inference(
        selectedImageUri: Uri?,
        resultImageUri: Uri?,
        continueWithResult: Boolean,
        segModel: String,
        downscaleMp: Float?,
        maskUpscale: Float,
        scoreThreshold: Float,
    ): UiState {
        val processingUri = if (continueWithResult && resultImageUri != null) {
            resultImageUri
        } else {
            selectedImageUri ?: throw IllegalArgumentException("No image selected")
        }

        val useFP16 = segModel.contains("fp16")
        val modelName = when {
            segModel.contains("small") -> "yolo11s"
            segModel.contains("nano") -> "yolo11n"
            segModel.contains("medium") -> "yolo11m"
            else -> "yolo11s"
        }
        yoloInference.initialize(modelName = modelName, useFP16 = useFP16)
        miGanInference.initialize()

        val processedImage = imageProcessor.processInputImage(
            imageUri = processingUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = downscaleMp
        )

        val maskMat = yoloInference.inferDetailed(
            transformedMat = processedImage.transformedMat,
            xRatio = processedImage.xRatio,
            yRatio = processedImage.yRatio,
            upscaleFactor = maskUpscale,
            originalWidth = processedImage.originalMat.cols(),
            originalHeight = processedImage.originalMat.rows(),
            overrideConfig = YoloConfig(scoreThreshold = scoreThreshold)
        ).mask

        val resultMat = miGanInference.inferMiGan(
            imageMat = processedImage.originalMat,
            maskMat = maskMat
        )

        val processedBitmap = Bitmap.createBitmap(resultMat.cols(), resultMat.rows(), Bitmap.Config.ARGB_8888)
        org.opencv.android.Utils.matToBitmap(resultMat, processedBitmap)

        val maskBitmap = Bitmap.createBitmap(maskMat.cols(), maskMat.rows(), Bitmap.Config.ARGB_8888)
        org.opencv.android.Utils.matToBitmap(maskMat, maskBitmap)

        return UiState.Success(processedBitmap, maskBitmap)
    }
}

sealed class UiState {
    data class Success(val processedBitmap: Bitmap, val maskBitmap: Bitmap) : UiState()
    data class Error(val message: String) : UiState()
}

sealed class BatchUiState {
    data class Success(val results: List<Bitmap>) : BatchUiState()
    data class Error(val message: String) : BatchUiState()
    data class Progress(val progress: Int, val total: Int) : BatchUiState()
}
