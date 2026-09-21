package de.konradvoelkel.android.autokorrektur.ui.diagnostics

import android.app.Activity
import android.content.Intent
import android.view.LayoutInflater
import android.widget.Toast
import androidx.core.content.FileProvider
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import de.konradvoelkel.android.autokorrektur.R
import de.konradvoelkel.android.autokorrektur.databinding.DialogDiagnosticsBinding
import de.konradvoelkel.android.autokorrektur.telemetry.Telemetry
import de.konradvoelkel.android.autokorrektur.utils.AppLogger

/**
 * The "Diagnostics" menu entry: the opt-in switch for [Telemetry], a summary of what has been
 * recorded, and Export (share sheet, via the FileProvider `cache` path) / Delete actions.
 *
 * Kept as a plain dialog rather than a settings screen because it is the app's only setting and
 * the `core` tier has no settings surface at all (docs/MVP_FEATURE_FLAG_PLAN.md §1). Available on
 * every flavor — field testers on `core` are exactly who the export is for.
 */
object DiagnosticsDialog {

    fun show(activity: Activity) {
        val binding = DialogDiagnosticsBinding.inflate(LayoutInflater.from(activity))
        binding.switchDiagnostics.isChecked = Telemetry.isEnabled
        binding.switchDiagnostics.setOnCheckedChangeListener { _, checked ->
            Telemetry.setEnabled(checked)
            binding.tvDiagnosticsSummary.text = summary(activity)
        }
        binding.tvDiagnosticsSummary.text = summary(activity)

        MaterialAlertDialogBuilder(activity)
            .setTitle(R.string.diagnostics_title)
            .setView(binding.root)
            .setPositiveButton(R.string.diagnostics_export) { _, _ -> export(activity) }
            .setNeutralButton(R.string.diagnostics_delete) { _, _ -> confirmDelete(activity) }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun summary(activity: Activity): String {
        val count = Telemetry.eventCount()
        val kb = Telemetry.sizeBytes() / 1024.0
        val id = Telemetry.installId
        return buildString {
            append(activity.resources.getQuantityString(R.plurals.diagnostics_event_count, count, count))
            append(" · ").append(String.format(java.util.Locale.ROOT, "%.1f KB", kb))
            if (id != null) append('\n').append(activity.getString(R.string.diagnostics_install_id_label, id.take(8)))
        }
    }

    private fun export(activity: Activity) {
        val file = try {
            Telemetry.buildExportFile(activity)
        } catch (e: Exception) {
            AppLogger.error("Diagnostics export failed", e)
            null
        }
        if (file == null) {
            Toast.makeText(activity, R.string.diagnostics_export_empty, Toast.LENGTH_SHORT).show()
            return
        }
        val uri = FileProvider.getUriForFile(activity, "${activity.packageName}.fileprovider", file)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "application/json"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, activity.getString(R.string.diagnostics_export_subject))
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        activity.startActivity(Intent.createChooser(send, activity.getString(R.string.diagnostics_export_chooser)))
    }

    private fun confirmDelete(activity: Activity) {
        MaterialAlertDialogBuilder(activity)
            .setTitle(R.string.diagnostics_delete)
            .setMessage(R.string.diagnostics_delete_confirm)
            .setPositiveButton(R.string.btn_delete) { _, _ ->
                Telemetry.clear()
                Toast.makeText(activity, R.string.diagnostics_deleted, Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }
}
