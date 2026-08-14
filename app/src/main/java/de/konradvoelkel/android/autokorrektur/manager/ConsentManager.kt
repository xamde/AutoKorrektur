package de.konradvoelkel.android.autokorrektur.manager

import android.content.Context
import android.content.SharedPreferences

/**
 * Manages GDPR user consent for optional cloud-based image processing.
 */
class ConsentManager(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PreferencesConstants.PREFS_NAME, Context.MODE_PRIVATE)

    fun isConsentGranted(): Boolean {
        return prefs.getBoolean(PreferencesConstants.KEY_GDPR_CONSENT, false)
    }

    fun setConsentGranted(granted: Boolean) {
        prefs.edit().putBoolean(PreferencesConstants.KEY_GDPR_CONSENT, granted).apply()
    }
}
