package de.konradvoelkel.android.autokorrektur.ui.model

import android.graphics.Bitmap
import android.net.Uri
import de.konradvoelkel.android.autokorrektur.model.BatchProcessingResult
import de.konradvoelkel.android.autokorrektur.model.InpaintingQualityMode
import de.konradvoelkel.android.autokorrektur.pipeline.PipelineResult
import de.konradvoelkel.android.autokorrektur.pipeline.PipelineStage

/**
 * Sealed class representing the different UI states for the Main Fragment.
 */
sealed class MainUiState {
    data object Idle : MainUiState()

    /**
     * In-progress inference. [stage] is resolved to text in the UI's locale at display time;
     * [batchIndex]/[batchTotal] (1-based, 0 = single image) prefix it with the batch position.
     */
    data class Loading(
        val stage: PipelineStage,
        val percent: Int,
        val intermediateInpaintedBitmap: Bitmap? = null,
        val batchIndex: Int = 0,
        val batchTotal: Int = 0
    ) : MainUiState()

    data class Success(
        val result: PipelineResult
    ) : MainUiState()

    data class Error(
        val message: String
    ) : MainUiState()
}

/**
 * Data class for consistent UI state properties that aren't strictly mutually exclusive.
 */
data class MainUiProperties(
    val selectedImageUri: Uri? = null,
    val selectedImageUris: List<Uri> = emptyList(),
    val sliderPosition: Float = 0.5f,
    val isBatchMode: Boolean = false,
    val qualityMode: InpaintingQualityMode = InpaintingQualityMode.FAST_PREVIEW,
    val batchProcessingResults: List<BatchProcessingResult> = emptyList()
)
