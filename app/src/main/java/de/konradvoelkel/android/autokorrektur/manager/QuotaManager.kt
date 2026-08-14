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
        context.getSharedPreferences(PreferencesConstants.PREFS_NAME, Context.MODE_PRIVATE)

    /**
     * Returns the unique device UUID, creating one if not yet initialized.
     */
    fun getDeviceUuid(): String {
        var uuid = prefs.getString(PreferencesConstants.KEY_DEVICE_UUID, null)
        if (uuid == null) {
            uuid = UUID.randomUUID().toString()
            prefs.edit().putString(PreferencesConstants.KEY_DEVICE_UUID, uuid).apply()
        }
        return uuid
    }

    /**
     * Returns remaining quota for today (out of DEFAULT_DAILY_LIMIT).
     */
    @Synchronized
    fun getRemainingDailyQuota(): Int {
        val today = getTodayKey()
        val lastDate = prefs.getString(PreferencesConstants.KEY_QUOTA_DATE, "")
        if (lastDate != today) {
            // Reset quota for new day
            prefs.edit()
                .putString(PreferencesConstants.KEY_QUOTA_DATE, today)
                .putInt(PreferencesConstants.KEY_USED_COUNT, 0)
                .apply()
            return DEFAULT_DAILY_LIMIT
        }

        val used = prefs.getInt(PreferencesConstants.KEY_USED_COUNT, 0)
        return (DEFAULT_DAILY_LIMIT - used).coerceAtLeast(0)
    }

    /**
     * Checks if the device has available quota today.
     */
    fun hasAvailableQuota(): Boolean {
        return getRemainingDailyQuota() > 0
    }

    /**
     * Attempts to consume one quota point for today.
     * @return true if quota was successfully consumed, false if quota exceeded.
     */
    @Synchronized
    fun consumeQuota(): Boolean {
        val remaining = getRemainingDailyQuota()
        if (remaining <= 0) return false

        val used = prefs.getInt(PreferencesConstants.KEY_USED_COUNT, 0)
        prefs.edit().putInt(PreferencesConstants.KEY_USED_COUNT, used + 1).apply()
        return true
    }

    private fun getTodayKey(): String {
        return try {
            java.time.LocalDate.now().toString()
        } catch (_: Throwable) {
            SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())
        }
    }

    companion object {
        const val DEFAULT_DAILY_LIMIT = 5
    }
}
