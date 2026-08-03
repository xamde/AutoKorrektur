package de.konradvoelkel.android.autokorrektur.ml.api

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
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

class ServerSdxlApi(private val context: Context) {
    
    // In production, configure to use HTTPS and proper domain
    // For local emulator testing against local server, we use 10.0.2.2
    private val serverUrl = "http://10.0.2.2:8000/v1/inpaint"

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    suspend fun processWithSdxl(
        originalBitmap: Bitmap, 
        maskBitmap: Bitmap, 
        previewBitmap: Bitmap
    ): Bitmap = withContext(Dispatchers.IO) {
        AppLogger.info("ServerSdxlApi: Sending images to server for SDXL inpainting...")
        
        val origBytes = bitmapToByteArray(originalBitmap)
        val maskBytes = bitmapToByteArray(maskBitmap)
        val previewBytes = bitmapToByteArray(previewBitmap)
        
        val sharedPrefs = context.getSharedPreferences("autokorrektur_prefs", Context.MODE_PRIVATE)
        var deviceUuid = sharedPrefs.getString("device_uuid", null)
        if (deviceUuid == null) {
            deviceUuid = java.util.UUID.randomUUID().toString()
            sharedPrefs.edit().putString("device_uuid", deviceUuid).apply()
        }
        
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
            val response = client.newCall(request).execute()
            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: "Unknown error"
                AppLogger.error("Server returned ${response.code}: $errorBody")
                throw Exception("Server inpainting failed: ${response.code}")
            }
            
            val responseBytes = response.body?.bytes() ?: throw Exception("Empty response from server")
            val resultBitmap = BitmapFactory.decodeByteArray(responseBytes, 0, responseBytes.size) 
                ?: throw Exception("Failed to decode image from server")
                
            AppLogger.info("ServerSdxlApi: Successfully received inpainted image")
            return@withContext resultBitmap
            
        } catch (e: Exception) {
            AppLogger.error("ServerSdxlApi Error", e)
            throw e
        }
    }
    
    private fun bitmapToByteArray(bitmap: Bitmap): ByteArray {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream)
        return stream.toByteArray()
    }
}
