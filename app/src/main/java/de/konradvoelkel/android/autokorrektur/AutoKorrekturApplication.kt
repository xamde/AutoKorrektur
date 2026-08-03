package de.konradvoelkel.android.autokorrektur

import android.app.Application
import de.konradvoelkel.android.autokorrektur.utils.AppLogger

/**
 * Custom Application class to handle global initialization.
 */
class AutoKorrekturApplication : Application() {

    override fun onCreate() {
        super.onCreate()

        // Initialize the logger with application context
        AppLogger.initialize(this)

        AppLogger.info("AutoKorrekturApplication created and logger initialized.")
    }
}
