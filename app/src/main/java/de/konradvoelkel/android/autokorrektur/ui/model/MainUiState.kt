package de.konradvoelkel.android.autokorrektur.ui.model

import android.net.Uri
import de.konradvoelkel.android.autokorrektur.model.BatchProcessingResult
import de.konradvoelkel.android.autokorrektur.pipeline.PipelineResult

/**
 * Sealed class representing the different UI states for the Main Fragment.
 */
sealed class MainUiState {
    object Idle : MainUiState()

    data class Loading(
        val stage: String,
        val percent: Int
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
    val batchProcessingResults: List<BatchProcessingResult> = emptyList()
)
