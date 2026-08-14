package de.konradvoelkel.android.autokorrektur.ml.factory

/**
 * Enumeration of supported inpainting model backends.
 */
enum class InpaintingModelType(val id: String, val displayName: String) {
    MIGAN("migan", "MI-GAN (Ultra-Fast)"),
    LAMA("lama", "LaMa (High-Fidelity)"),
    SDXL_CLOUD("sdxl", "Stable Diffusion XL (Cloud)");

    companion object {
        fun fromString(key: String?): InpaintingModelType {
            return when (key?.lowercase()?.trim()) {
                "lama" -> LAMA
                "sdxl", "cloud", "sdxl_cloud" -> SDXL_CLOUD
                else -> MIGAN
            }
        }
    }
}
