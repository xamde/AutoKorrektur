package de.konradvoelkel.android.autokorrektur.ui.delegate

import android.app.AlertDialog
import android.content.Context
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
            onMessage("No batch results to export")
            return
        }

        AlertDialog.Builder(context)
            .setTitle("Export CSV Report")
            .setMessage("Save batch processing metrics for ${results.size} images to Documents/AutoKorrektur?")
            .setPositiveButton("Export") { _, _ ->
                val csvUri = exportManager.exportBatchResultsToCSV(results)
                if (csvUri != null) {
                    onMessage("Batch report exported to CSV")
                } else {
                    onMessage("Failed to export CSV report")
                }
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }
}
