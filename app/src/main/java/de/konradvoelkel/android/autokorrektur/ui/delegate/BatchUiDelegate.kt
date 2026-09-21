package de.konradvoelkel.android.autokorrektur.ui.delegate

import android.app.AlertDialog
import android.content.Context
import de.konradvoelkel.android.autokorrektur.R
import de.konradvoelkel.android.autokorrektur.model.BatchProcessingResult
import de.konradvoelkel.android.autokorrektur.utils.ImageExportManager

/**
 * Delegate handling UI presentation and CSV export for batch processing runs.
 */
class BatchUiDelegate(
    private val context: Context,
    private val exportManager: ImageExportManager,
    private val onMessage: (String) -> Unit
) {

    /**
     * Displays a confirmation dialog to export the batch statistics as a CSV file.
     */
    fun showCsvExportDialog(results: List<BatchProcessingResult>) {
        if (results.isEmpty()) {
            onMessage(context.getString(R.string.csv_export_none))
            return
        }

        AlertDialog.Builder(context)
            .setTitle(R.string.csv_export_dialog_title)
            .setMessage(context.getString(R.string.csv_export_dialog_message, results.size))
            .setPositiveButton(R.string.btn_export_csv) { _, _ ->
                val csvUri = exportManager.exportBatchResultsToCSV(results)
                if (csvUri != null) {
                    onMessage(context.getString(R.string.csv_export_done))
                } else {
                    onMessage(context.getString(R.string.csv_export_failed))
                }
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }
}
