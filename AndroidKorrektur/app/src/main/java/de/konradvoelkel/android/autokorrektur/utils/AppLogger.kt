package de.konradvoelkel.android.autokorrektur.utils

import android.content.Context
import android.os.Environment
import android.util.Log
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * A comprehensive logging utility that writes to both Android Log and a persistent log file.
 * This replaces the problematic Toast messages and println statements with proper logging.
 */
object AppLogger {
    private const val TAG = "AutoKorrektur"
    private const val LOG_FILE_NAME = "autokorrektur_debug.log"
    private const val MAX_LOG_FILE_SIZE = 5 * 1024 * 1024 // 5MB

    private var logFile: File? = null
    private var isInitialized = false
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.getDefault())

    enum class LogLevel(val priority: Int, val tag: String) {
        DEBUG(Log.DEBUG, "DEBUG"),
        INFO(Log.INFO, "INFO"),
        WARN(Log.WARN, "WARN"),
        ERROR(Log.ERROR, "ERROR")
    }

    /**
     * Initialize the logger with application context.
     * This should be called once when the app starts.
     */
    fun initialize(context: Context) {
        try {
            // Try to use external storage first, fall back to internal if not available
            val logDir = if (Environment.getExternalStorageState() == Environment.MEDIA_MOUNTED) {
                File(context.getExternalFilesDir(null), "logs")
            } else {
                File(context.filesDir, "logs")
            }

            if (!logDir.exists()) {
                logDir.mkdirs()
            }

            logFile = File(logDir, LOG_FILE_NAME)

            // Rotate log file if it's too large
            if (logFile?.exists() == true && logFile?.length()!! > MAX_LOG_FILE_SIZE) {
                rotateLogFile()
            }

            isInitialized = true

            // Log initialization success
            info("AppLogger initialized successfully")
            info("Log file location: ${logFile?.absolutePath}")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize AppLogger", e)
            isInitialized = false
        }
    }

    /**
     * Log a debug message
     */
    fun debug(message: String, throwable: Throwable? = null) {
        log(LogLevel.DEBUG, message, throwable)
    }

    /**
     * Log an info message
     */
    fun info(message: String, throwable: Throwable? = null) {
        log(LogLevel.INFO, message, throwable)
    }

    /**
     * Log a warning message
     */
    fun warn(message: String, throwable: Throwable? = null) {
        log(LogLevel.WARN, message, throwable)
    }

    /**
     * Log an error message
     */
    fun error(message: String, throwable: Throwable? = null) {
        log(LogLevel.ERROR, message, throwable)
    }

    /**
     * Log a message with specified level
     */
    private fun log(level: LogLevel, message: String, throwable: Throwable? = null) {
        val timestamp = dateFormat.format(Date())
        val logMessage = "[$timestamp] [${level.tag}] $message"

        // Always log to Android Log
        when (level) {
            LogLevel.DEBUG -> Log.d(TAG, message, throwable)
            LogLevel.INFO -> Log.i(TAG, message, throwable)
            LogLevel.WARN -> Log.w(TAG, message, throwable)
            LogLevel.ERROR -> Log.e(TAG, message, throwable)
        }

        // Write to file if initialized
        if (isInitialized && logFile != null) {
            try {
                FileWriter(logFile, true).use { writer ->
                    writer.appendLine(logMessage)
                    if (throwable != null) {
                        writer.appendLine("Exception: ${throwable.javaClass.simpleName}: ${throwable.message}")
                        throwable.stackTrace.forEach { element ->
                            writer.appendLine("  at $element")
                        }
                    }
                    writer.flush()
                }
            } catch (e: IOException) {
                Log.e(TAG, "Failed to write to log file", e)
            }
        }
    }

    /**
     * Rotate the log file when it gets too large
     */
    private fun rotateLogFile() {
        try {
            val backupFile = File(logFile?.parent, "${LOG_FILE_NAME}.old")
            if (backupFile.exists()) {
                backupFile.delete()
            }
            logFile?.renameTo(backupFile)
            logFile = File(logFile?.parent, LOG_FILE_NAME)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to rotate log file", e)
        }
    }

    /**
     * Get the current log file for sharing or viewing
     */
    fun getLogFile(): File? = logFile

    /**
     * Clear the log file
     */
    fun clearLog() {
        try {
            logFile?.writeText("")
            info("Log file cleared")
        } catch (e: Exception) {
            error("Failed to clear log file", e)
        }
    }

    /**
     * Get log file content as string (for debugging or sharing)
     */
    fun getLogContent(): String? {
        return try {
            logFile?.readText()
        } catch (e: Exception) {
            error("Failed to read log file", e)
            null
        }
    }
}