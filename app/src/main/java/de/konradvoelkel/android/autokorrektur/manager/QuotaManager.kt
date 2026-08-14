package de.konradvoelkel.android.autokorrektur.manager

import android.content.Context
import android.content.SharedPreferences
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

/**
 * Manages daily free quota and device identification for Cloud SDXL Premium Inpainting.
 */
class QuotaManager(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val dateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.US)

    /**
     * Returns the unique device UUID, creating one if not yet initialized.
     */
    fun getDeviceUuid(): String {
        var uuid = prefs.getString(KEY_DEVICE_UUID, null)
        if (uuid == null) {
            uuid = UUID.randomUUID().toString()
            prefs.edit().putString(KEY_DEVICE_UUID, uuid).apply()
        }
        return uuid
    }

    /**
     * Returns remaining quota for today (out of DEFAULT_DAILY_LIMIT).
     */
    @Synchronized
    fun getRemainingDailyQuota(): Int {
        val today = getTodayKey()
        val lastDate = prefs.getString(KEY_QUOTA_DATE, "")
        if (lastDate != today) {
            // Reset quota for new day
            prefs.edit()
                .putString(KEY_QUOTA_DATE, today)
                .putInt(KEY_USED_COUNT, 0)
                .apply()
            return DEFAULT_DAILY_LIMIT
        }
        val used = prefs.getInt(KEY_USED_COUNT, 0)
        return (DEFAULT_DAILY_LIMIT - used).coerceAtLeast(0)
    }

    /**
     * Checks if the device has available quota today.
     */
    fun hasAvailableQuota(): Boolean {
        return getRemainingDailyQuota() > 0
    }

    /**
     * Consumes one quota credit if available.
     * @return true if quota was successfully consumed, false if quota exceeded.
     */
    @Synchronized
    fun consumeQuota(): Boolean {
        val remaining = getRemainingDailyQuota()
        if (remaining <= 0) return false

        val used = prefs.getInt(KEY_USED_COUNT, 0)
        prefs.edit().putInt(KEY_USED_COUNT, used + 1).apply()
        return true
    }

    private fun getTodayKey(): String = dateFormat.format(Date())

    companion object {
        const val DEFAULT_DAILY_LIMIT = 5
        private const val PREFS_NAME = "autokorrektur_prefs"
        private const val KEY_DEVICE_UUID = "device_uuid"
        private const val KEY_QUOTA_DATE = "sdxl_quota_date"
        private const val KEY_USED_COUNT = "sdxl_quota_used"
    }
}
