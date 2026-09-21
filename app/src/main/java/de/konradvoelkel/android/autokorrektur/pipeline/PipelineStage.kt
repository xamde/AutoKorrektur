package de.konradvoelkel.android.autokorrektur.pipeline

import android.content.Context
import androidx.annotation.StringRes
import de.konradvoelkel.android.autokorrektur.R

/**
 * A localizable progress step reported by the image, progressive-tile and video pipelines.
 *
 * The pipelines used to hand hard-coded English strings to their progress callbacks, which the UI
 * showed verbatim — so a German device saw "Running YOLO Segmentation" next to "Verarbeitung…".
 * Every step is now a string resource; counted steps ("region 2/5") carry [current] and [total]
 * and their resource takes `%1$d`/`%2$d`. [label] resolves the text at display time, so the same
 * stage object also serves as the (locale-independent) `stage` key in diagnostics events.
 */
data class PipelineStage(
    @field:StringRes val labelRes: Int,
    val current: Int = 0,
    val total: Int = 0,
) {
    /** Resolves the user-visible text in the current locale. */
    fun label(context: Context): String =
        if (total > 0) context.getString(labelRes, current, total) else context.getString(labelRes)

    companion object {
        val INITIALIZING = PipelineStage(R.string.stage_initializing)
        val INITIALIZING_ENGINES = PipelineStage(R.string.stage_initializing_engines)
        val LOADING_IMAGE = PipelineStage(R.string.stage_loading_image)
        val SEGMENTATION = PipelineStage(R.string.stage_segmentation)
        val INPAINTING_CLOUD = PipelineStage(R.string.stage_inpainting_cloud)
        val INPAINTING_HIGH_RES = PipelineStage(R.string.stage_inpainting_high_res)
        val INPAINTING_ON_DEVICE = PipelineStage(R.string.stage_inpainting_on_device)
        val COMPLETED = PipelineStage(R.string.stage_completed)

        // Progressive tile inpainter
        val EXTRACTING_REGIONS = PipelineStage(R.string.stage_extracting_regions)
        fun inpaintingRegion(current: Int, total: Int) =
            PipelineStage(R.string.stage_inpainting_region, current, total)
        fun refiningTextures(current: Int, total: Int) =
            PipelineStage(R.string.stage_refining_textures, current, total)
        val FINALIZING_HIGH_RES = PipelineStage(R.string.stage_finalizing_high_res)

        // Video
        val PREPARING_ENCODER = PipelineStage(R.string.stage_preparing_encoder)
        fun inpaintingFrame(current: Int, total: Int) =
            PipelineStage(R.string.stage_inpainting_frame, current, total)
        val FINALIZING_VIDEO = PipelineStage(R.string.stage_finalizing_video)
        val VIDEO_COMPLETED = PipelineStage(R.string.stage_video_completed)
    }
}
