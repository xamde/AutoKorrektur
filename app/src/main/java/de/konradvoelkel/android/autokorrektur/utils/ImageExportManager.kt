package de.konradvoelkel.android.autokorrektur.utils

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import de.konradvoelkel.android.autokorrektur.model.BatchProcessingResult
import java.io.File
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Manages exporting images to the gallery and batch results to CSV.
 */
class ImageExportManager(private val context: Context) {

    /**
     * Saves the given [bitmap] to the system gallery.
     */
    fun saveImageToGallery(bitmap: Bitmap): Uri? {
        return try {
            val filename = "AutoKorrektur_${System.currentTimeMillis()}.jpg"
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)
            }

            val contentResolver = context.contentResolver
            val imageUri = contentResolver.insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues
            )
            val fos: OutputStream? = imageUri?.let { contentResolver.openOutputStream(it) }

            fos?.use {
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, it)
                AppLogger.info("Image saved to gallery successfully: $imageUri")
            }

            imageUri
        } catch (e: Exception) {
            AppLogger.error("Error saving image to gallery", e)
            null
        }
    }

    /**
     * Exports a list of [BatchProcessingResult] to a CSV file in the Documents directory.
     */
    fun exportBatchResultsToCSV(results: List<BatchProcessingResult>): File? {
        if (results.isEmpty()) return null

        return try {
            val csvContent = StringBuilder()
            csvContent.append("Image Name,Processing Time (ms),Mask Upscale,Score Threshold,Downshift,Downscale MP,Segmentation Model,Success,Error Message\n")

            results.forEach { result ->
                csvContent.append("${result.originalImageName},")
                csvContent.append("${result.processingTimeMs},")
                csvContent.append("${result.maskUpscale},")
                csvContent.append("${result.scoreThreshold},")
                csvContent.append("${result.downshift},")
                csvContent.append("${result.downscaleMp},")
                csvContent.append("${result.segmentationModel},")
                csvContent.append("${result.success},")
                csvContent.append("${result.errorMessage ?: ""}\n")
            }

            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "autokorrektur_batch_results_$timestamp.csv"

            val csvFile = File(
                context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS),
                fileName
            )
            csvFile.writeText(csvContent.toString())

            AppLogger.info("CSV exported to: ${csvFile.absolutePath}")
            csvFile
        } catch (e: Exception) {
            AppLogger.error("Failed to export CSV", e)
            null
        }
    }
}
