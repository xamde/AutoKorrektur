package com.autokorrektur.android

import android.app.Application
import dagger.hilt.android.HiltAndroidApp
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoaderCallback
import org.opencv.android.OpenCVManagerService
import timber.log.Timber

@HiltAndroidApp
class AutoKorrekturApplication : Application() {

    private val openCVLoaderCallback = object : OpenCVLoaderCallback() {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Timber.d("OpenCV loaded successfully")
                    // OpenCV is ready to use
                }
                else -> {
                    super.onManagerConnected(status)
                    Timber.e("OpenCV initialization failed")
                }
            }
        }
    }

    override fun onCreate() {
        super.onCreate()

        // Initialize Timber for logging
        if (BuildConfig.DEBUG) {
            Timber.plant(Timber.DebugTree())
        }

        Timber.d("AutoKorrektur Application started")

        // Initialize OpenCV
        initializeOpenCV()
    }

    private fun initializeOpenCV() {
        if (!OpenCVManagerService.connectManagerService(this, openCVLoaderCallback)) {
            Timber.e("Cannot connect to OpenCV Manager")
        }
    }
}
