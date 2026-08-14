package de.konradvoelkel.android.autokorrektur.ml.asset

import android.content.Context
import java.io.File
import java.io.InputStream

/**
 * Unified model asset provider with transparent support for both standard Android Assets
 * and Play Asset Delivery (PAD) install-time asset packs.
 */
object ModelAssetProvider {

    /**
     * Opens an InputStream for the specified model file relative to assets.
     */
    fun openModelAsset(context: Context, relativePath: String): InputStream {
        val padFile = File(context.filesDir, relativePath)
        if (padFile.exists()) {
            return padFile.inputStream()
        }
        return context.assets.open(relativePath)
    }

    /**
     * Returns or extracts the model file to local storage for native engines requiring File paths.
     */
    @Synchronized
    fun getOrExtractModelFile(context: Context, relativePath: String): File {
        val fileName = File(relativePath).name
        val destinationFile = File(context.filesDir, fileName)
        if (!destinationFile.exists() || destinationFile.length() == 0L) {
            val tempFile = File(context.filesDir, "$fileName.tmp_${System.currentTimeMillis()}")
            try {
                openModelAsset(context, relativePath).use { input ->
                    tempFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                if (!tempFile.renameTo(destinationFile)) {
                    // Fallback copy if renameTo across mount fails
                    tempFile.copyTo(destinationFile, overwrite = true)
                    tempFile.delete()
                }
            } catch (e: Exception) {
                tempFile.delete()
                throw e
            }
        }
        return destinationFile
    }
}
