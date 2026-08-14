package de.konradvoelkel.android.autokorrektur.manager

import android.content.Context
import android.content.SharedPreferences

/**
 * Manages GDPR user consent for optional cloud-based image processing.
 */
class ConsentManager(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun isConsentGranted(): Boolean {
        return prefs.getBoolean(KEY_GDPR_CONSENT, false)
    }

    fun setConsentGranted(granted: Boolean) {
        prefs.edit().putBoolean(KEY_GDPR_CONSENT, granted).apply()
    }

    companion object {
        private const val PREFS_NAME = "autokorrektur_prefs"
        private const val KEY_GDPR_CONSENT = "gdpr_sdxl_consent"
    }
}
