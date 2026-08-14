package de.konradvoelkel.android.autokorrektur.ml.api

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import de.konradvoelkel.android.autokorrektur.manager.QuotaManager
import de.konradvoelkel.android.autokorrektur.ml.errors.CloudInferenceException
import de.konradvoelkel.android.autokorrektur.ml.errors.QuotaExceededException
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

class ServerSdxlApi(
    private val context: Context,
    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build(),
    private val quotaManager: QuotaManager = QuotaManager(context)
) : ServerInpainter {

    private var serverUrl = de.konradvoelkel.android.autokorrektur.BuildConfig.BACKEND_URL

    override suspend fun processWithSdxl(
        originalBitmap: Bitmap, 
        maskBitmap: Bitmap, 
        previewBitmap: Bitmap
    ): Bitmap = withContext(Dispatchers.IO) {
        if (!quotaManager.hasAvailableQuota()) {
            throw QuotaExceededException("Daily free SDXL limit reached (${QuotaManager.DEFAULT_DAILY_LIMIT} edits/day). Please try again tomorrow.")
        }

        AppLogger.info("ServerSdxlApi: Sending images to server for SDXL inpainting (Remaining: ${quotaManager.getRemainingDailyQuota()})...")
        
        val origBytes = bitmapToByteArray(originalBitmap)
        val maskBytes = bitmapToByteArray(maskBitmap)
        val previewBytes = bitmapToByteArray(previewBitmap)
        
        val deviceUuid = quotaManager.getDeviceUuid()
        
        val playIntegrityToken = "mock-valid-token" // Placeholder for actual API integration
        
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("device_uuid", deviceUuid)
            .addFormDataPart("play_integrity_token", playIntegrityToken)
            .addFormDataPart("image", "image.jpg", origBytes.toRequestBody("image/jpeg".toMediaTypeOrNull()))
            .addFormDataPart("mask", "mask.jpg", maskBytes.toRequestBody("image/jpeg".toMediaTypeOrNull()))
            .addFormDataPart("preview", "preview.jpg", previewBytes.toRequestBody("image/jpeg".toMediaTypeOrNull()))
            .build()
            
        val request = Request.Builder()
            .url(serverUrl)
            .post(requestBody)
            .build()
            
        try {
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    val errorBody = response.body.string()
                    AppLogger.error("Server returned ${response.code}: $errorBody")
                    throw CloudInferenceException("Server inpainting failed (${response.code}): $errorBody")
                }

                val responseBytes = response.body.bytes()
                val resultBitmap = BitmapFactory.decodeByteArray(responseBytes, 0, responseBytes.size)
                    ?: throw CloudInferenceException("Failed to decode image from server")
                quotaManager.consumeQuota()
                AppLogger.info("ServerSdxlApi: Successfully received inpainted image (Remaining today: ${quotaManager.getRemainingDailyQuota()})")
                return@withContext resultBitmap
            }
        } catch (e: Exception) {
            AppLogger.error("ServerSdxlApi Error", e)
            if (e is CloudInferenceException || e is QuotaExceededException) throw e
            throw CloudInferenceException("Cloud communication error: ${e.message}", e)
        }
    }
    
    private fun bitmapToByteArray(bitmap: Bitmap): ByteArray {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream)
        return stream.toByteArray()
    }
}
