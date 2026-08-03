package de.konradvoelkel.android.autokorrektur.ml.api

import android.graphics.Bitmap
import java.io.IOException

/**
 * Interface for server-side high-quality inpainting.
 */
interface ServerInpainter {
    /**
     * Sends the original image, mask, and structural preview to the server for processing.
     * @return The inpainted result bitmap.
     * @throws IOException if network or server error occurs.
     */
    @Throws(IOException::class)
    suspend fun processWithSdxl(
        originalBitmap: Bitmap,
        maskBitmap: Bitmap,
        previewBitmap: Bitmap
    ): Bitmap
}
