package de.konradvoelkel.android.autokorrektur

import android.graphics.Bitmap
import android.net.Uri
import android.os.Looper
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

/**
 * ViewModel to retain UI state across configuration changes (device rotation, theme toggles).
 */
class MainViewModel : ViewModel() {

    private val _selectedImageUri = MutableLiveData<Uri?>(null)
    val selectedImageUri: LiveData<Uri?> = _selectedImageUri

    private val _processedImageUri = MutableLiveData<Uri?>(null)
    val processedImageUri: LiveData<Uri?> = _processedImageUri

    var processedBitmap: Bitmap? = null
        private set

    var originalBitmap: Bitmap? = null
        private set

    var sliderPosition: Float = 0.5f

    fun setSelectedImageUri(uri: Uri?) {
        updateLiveData(_selectedImageUri, uri)
    }

    fun setProcessedResult(uri: Uri?, original: Bitmap?, processed: Bitmap?) {
        this.originalBitmap = original
        this.processedBitmap = processed
        updateLiveData(_processedImageUri, uri)
    }

    fun clearState() {
        updateLiveData(_selectedImageUri, null)
        updateLiveData(_processedImageUri, null)
        originalBitmap = null
        processedBitmap = null
        sliderPosition = 0.5f
    }

    private fun <T> updateLiveData(liveData: MutableLiveData<T>, newValue: T) {
        try {
            if (Looper.myLooper() == Looper.getMainLooper()) {
                liveData.value = newValue
            } else {
                liveData.postValue(newValue)
            }
        } catch (_: Exception) {
            // JVM unit test fallback when Looper is unmocked
            liveData.postValue(newValue)
        }
    }
}
