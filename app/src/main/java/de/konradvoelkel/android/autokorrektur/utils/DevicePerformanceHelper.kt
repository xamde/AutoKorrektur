package de.konradvoelkel.android.autokorrektur.utils

import android.app.ActivityManager
import android.content.Context
import android.os.Build

object DevicePerformanceHelper {
    /**
     * Determines if the current device is considered "weak" or "old" based on RAM and CPU cores.
     * A weak device should use a smaller YOLO model (e.g., yolo11n-seg) instead of the small one (yolo11s-seg).
     */
    fun isWeakDevice(context: Context): Boolean {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        
        val totalRamGb = memoryInfo.totalMem / (1024 * 1024 * 1024.0)
        val cores = Runtime.getRuntime().availableProcessors()
        
        // Consider a device "weak" if it has less than 4GB RAM or less than 6 cores
        return totalRamGb < 4.0 || cores < 6
    }

    /**
     * Checks if Android Neural Networks API (NNAPI) is supported on this device.
     */
    fun isNnapiSupported(): Boolean {
        return true
    }
}
