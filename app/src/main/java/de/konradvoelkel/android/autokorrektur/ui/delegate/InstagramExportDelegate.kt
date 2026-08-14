package de.konradvoelkel.android.autokorrektur.ui.delegate

import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import de.konradvoelkel.android.autokorrektur.R
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import de.konradvoelkel.android.autokorrektur.utils.ImageExportManager
import de.konradvoelkel.android.autokorrektur.utils.InstagramExportUtils

/**
 * Delegate handling the Instagram Export & Before/After social graphic generation.
 */
class InstagramExportDelegate(
    private val context: Context,
    private val exportManager: ImageExportManager,
    private val onMessage: (String) -> Unit
) {

    /**
     * Shows the dialog to choose aspect ratio and share/save the graphic.
     */
    fun showExportDialog(
        originalBitmap: Bitmap,
        inpaintedBitmap: Bitmap
    ) {
        try {
            val options = context.resources.getStringArray(R.array.instagram_formats)
            AlertDialog.Builder(context)
                .setTitle(R.string.dialog_instagram_format_title)
                .setItems(options) { _, selectedPosition ->
                    val (ratio, layout) = when (selectedPosition) {
                        0 -> Pair(InstagramExportUtils.AspectRatio.SQUARE_1_1, InstagramExportUtils.LayoutStyle.SIDE_BY_SIDE)
                        1 -> Pair(InstagramExportUtils.AspectRatio.PORTRAIT_4_5, InstagramExportUtils.LayoutStyle.STACKED)
                        else -> Pair(InstagramExportUtils.AspectRatio.STORY_9_16, InstagramExportUtils.LayoutStyle.SIDE_BY_SIDE)
                    }

                    val graphic = InstagramExportUtils.createComparisonBitmap(
                        beforeBitmap = originalBitmap,
                        afterBitmap = inpaintedBitmap,
                        ratio = ratio,
                        layout = layout
                    )

                    val actionOptions = arrayOf(
                        context.getString(R.string.share_to_instagram),
                        context.getString(R.string.save_graphic)
                    )
                    AlertDialog.Builder(context)
                        .setTitle(R.string.dialog_instagram_ready_title)
                        .setItems(actionOptions) { _, actionWhich ->
                            when (actionWhich) {
                                0 -> {
                                    val shareUri = InstagramExportUtils.saveBitmapForSharing(context, graphic)
                                    val shareIntent = Intent(Intent.ACTION_SEND).apply {
                                        type = "image/jpeg"
                                        putExtra(Intent.EXTRA_STREAM, shareUri)
                                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                    }
                                    val chooser = Intent.createChooser(shareIntent, context.getString(R.string.share_chooser_title)).apply {
                                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                                    }
                                    context.startActivity(chooser)
                                }
                                1 -> {
                                    if (exportManager.saveImageToGallery(graphic) != null) {
                                        onMessage(context.getString(R.string.msg_instagram_graphic_saved))
                                    }
                                }
                            }
                        }
                        .show()
                }
                .show()
        } catch (e: Exception) {
            AppLogger.error("Failed to generate Instagram comparison graphic", e)
            onMessage(context.getString(R.string.error_export_message, e.message))
        }
    }
}
