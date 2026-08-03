package de.konradvoelkel.android.autokorrektur.model

/**
 * Data class to store batch processing results for CSV export.
 */
data class BatchProcessingResult(
    val originalImageName: String,
    val processingTimeMs: Long,
    val maskUpscale: Float,
    val scoreThreshold: Float,
    val downshift: Float,
    val downscaleMp: String,
    val segmentationModel: String,
    val success: Boolean,
    val errorMessage: String? = null
)
