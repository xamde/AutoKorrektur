package de.konradvoelkel.android.autokorrektur

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import de.konradvoelkel.android.autokorrektur.model.BatchProcessingResult
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import de.konradvoelkel.android.autokorrektur.ml.api.ServerSdxlApi
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine
import de.konradvoelkel.android.autokorrektur.pipeline.StaticImagePipeline
import de.konradvoelkel.android.autokorrektur.ui.model.MainUiProperties
import de.konradvoelkel.android.autokorrektur.ui.model.MainUiState
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * ViewModel to retain UI state and orchestrate ML inference.
 */
class MainViewModel(
    application: Application,
    private val pipeline: StaticImagePipeline
) : AndroidViewModel(application) {

    // Primary constructor for Android's ViewModelProvider
    constructor(application: Application) : this(
        application,
        StaticImagePipeline(
            ImageProcessor(application),
            YoloServiceImpl(YoloTFLiteEngine(application)),
            MiGanInference(application),
            ServerSdxlApi(application)
        )
    )

    private val _uiState = MutableStateFlow<MainUiState>(MainUiState.Idle)
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

    private val _properties = MutableStateFlow(MainUiProperties())
    val properties: StateFlow<MainUiProperties> = _properties.asStateFlow()

    private var inferenceJob: Job? = null

    fun setSelectedImageUri(uri: Uri?) {
        _properties.update { it.copy(selectedImageUri = uri) }
        _uiState.value = MainUiState.Idle
    }

    fun setSelectedImageUris(uris: List<Uri>) {
        _properties.update { it.copy(selectedImageUris = uris) }
    }

    fun setSliderPosition(position: Float) {
        _properties.update { it.copy(sliderPosition = position) }
    }

    fun setBatchMode(enabled: Boolean) {
        _properties.update { it.copy(isBatchMode = enabled) }
    }

    fun startInference(
        downscaleMp: Float?,
        maskUpscale: Float,
        scoreThreshold: Float,
        useServerSdxl: Boolean,
        downshift: Float = 0f,
        segModel: String = ""
    ) {
        val props = properties.value
        inferenceJob?.cancel()
        inferenceJob = viewModelScope.launch {
            try {
                if (props.isBatchMode) {
                    processBatch(
                        props.selectedImageUris,
                        downscaleMp,
                        maskUpscale,
                        scoreThreshold,
                        useServerSdxl,
                        downshift,
                        segModel
                    )
                } else {
                    processSingle(
                        props.selectedImageUri,
                        downscaleMp,
                        maskUpscale,
                        scoreThreshold,
                        useServerSdxl
                    )
                }
            } catch (e: Exception) {
                AppLogger.error("Inference failed", e)
                _uiState.value = MainUiState.Error(e.message ?: "Unknown error")
            }
        }
    }

    private suspend fun processSingle(
        uri: Uri?,
        downscaleMp: Float?,
        maskUpscale: Float,
        scoreThreshold: Float,
        useServerSdxl: Boolean
    ) {
        if (uri == null) return
        _uiState.value = MainUiState.Loading("Initializing", 0)

        val result = pipeline.processImage(
            uri = uri,
            downscaleMp = downscaleMp,
            maskUpscale = maskUpscale,
            scoreThreshold = scoreThreshold,
            useServerSdxl = useServerSdxl,
            onProgressUpdate = { stage, percent ->
                _uiState.value = MainUiState.Loading(stage, percent)
            }
        )

        if (result.errorMessage != null) {
            _uiState.value = MainUiState.Error(result.errorMessage)
        } else {
            _uiState.value = MainUiState.Success(result)
        }
    }

    private suspend fun processBatch(
        uris: List<Uri>,
        downscaleMp: Float?,
        maskUpscale: Float,
        scoreThreshold: Float,
        useServerSdxl: Boolean,
        downshift: Float,
        segModel: String
    ) {
        if (uris.isEmpty()) return
        val results = mutableListOf<BatchProcessingResult>()

        uris.forEachIndexed { index, uri ->
            val startTime = System.currentTimeMillis()
            val imageName = "Image_${index + 1}"
            _uiState.value = MainUiState.Loading("Batch (${index + 1}/${uris.size})", 0)

            try {
                val result = pipeline.processImage(
                    uri = uri,
                    downscaleMp = downscaleMp,
                    maskUpscale = maskUpscale,
                    scoreThreshold = scoreThreshold,
                    useServerSdxl = useServerSdxl,
                    onProgressUpdate = { stage, percent ->
                        _uiState.value = MainUiState.Loading("Batch ${index + 1}: $stage", percent)
                    }
                )

                val processingTime = System.currentTimeMillis() - startTime
                results.add(
                    BatchProcessingResult(
                        originalImageName = imageName,
                        processingTimeMs = processingTime,
                        maskUpscale = maskUpscale,
                        scoreThreshold = scoreThreshold,
                        downshift = downshift,
                        downscaleMp = downscaleMp?.toString() ?: "No Scaling",
                        segmentationModel = segModel,
                        success = result.errorMessage == null,
                        errorMessage = result.errorMessage
                    )
                )

                if (result.errorMessage == null) {
                    _uiState.value = MainUiState.Success(result)
                }
            } catch (e: Exception) {
                results.add(
                    BatchProcessingResult(
                        originalImageName = imageName,
                        processingTimeMs = System.currentTimeMillis() - startTime,
                        maskUpscale = maskUpscale,
                        scoreThreshold = scoreThreshold,
                        downshift = downshift,
                        downscaleMp = downscaleMp?.toString() ?: "No Scaling",
                        segmentationModel = segModel,
                        success = false,
                        errorMessage = e.message
                    )
                )
            }
        }
        _properties.update { it.copy(batchProcessingResults = results) }
    }

    fun clearState() {
        inferenceJob?.cancel()
        _properties.value = MainUiProperties()
        _uiState.value = MainUiState.Idle
    }

    override fun onCleared() {
        super.onCleared()
        pipeline.close()
        // B14: Recycle bitmaps if success state was held
        val currentState = uiState.value
        if (currentState is MainUiState.Success) {
            currentState.result.originalBitmap.recycle()
            currentState.result.maskBitmap.recycle()
            currentState.result.inpaintedBitmap?.recycle()
        }
    }
}
