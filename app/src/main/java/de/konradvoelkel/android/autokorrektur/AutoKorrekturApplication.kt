package de.konradvoelkel.android.autokorrektur

import android.app.Application
import de.konradvoelkel.android.autokorrektur.telemetry.Telemetry
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import org.opencv.android.OpenCVLoader

/**
 * Custom Application class to handle global initialization.
 */
class AutoKorrekturApplication : Application() {

    override fun onCreate() {
        super.onCreate()

        // Initialize the logger with application context
        AppLogger.initialize(this)

        // Opt-in on-device diagnostics (no-op until the user enables it in the Diagnostics dialog)
        Telemetry.initialize(this)

        // Initialize OpenCV native binaries for the entire process lifetime
        if (!OpenCVLoader.initLocal()) {
            AppLogger.error("OpenCV initialization failed via initLocal()")
        } else {
            AppLogger.info("OpenCV initialized successfully.")
        }

        AppLogger.info("AutoKorrekturApplication created and logger initialized.")
    }
}

