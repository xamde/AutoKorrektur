package de.konradvoelkel.android.autokorrektur.model

/**
 * Inpainting quality and computation modes for AutoKorrektur.
 */
enum class InpaintingQualityMode {
    /**
     * Fast single-pass on-device inpainting at 1-2 MP resolution (~200-500ms).
     */
    FAST_PREVIEW,

    /**
     * High-resolution multi-pass progressive tile inpainting at native sensor resolution (~1.5-3.5s).
     */
    HIGH_RES_PROGRESSIVE,

    /**
     * Premium cloud SDXL inpainting with memory-only Frankfurt, Germany server (~4-8s).
     */
    CLOUD_SDXL
}
