package de.konradvoelkel.android.autokorrektur.manager

import android.content.Context
import android.content.SharedPreferences
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class ConsentManagerTest {

    private lateinit var mockContext: Context
    private lateinit var mockPrefs: SharedPreferences
    private lateinit var mockEditor: SharedPreferences.Editor

    @Before
    fun setUp() {
        mockContext = mockk(relaxed = true)
        mockPrefs = mockk(relaxed = true)
        mockEditor = mockk(relaxed = true)

        every { mockContext.getSharedPreferences(PreferencesConstants.PREFS_NAME, Context.MODE_PRIVATE) } returns mockPrefs
        every { mockPrefs.edit() } returns mockEditor
        every { mockEditor.putBoolean(any(), any()) } returns mockEditor
    }

    @Test
    fun testIsConsentGranted_defaultIsFalse() {
        every { mockPrefs.getBoolean(PreferencesConstants.KEY_GDPR_CONSENT, false) } returns false
        val consentManager = ConsentManager(mockContext)
        assertFalse(consentManager.isConsentGranted())
    }

    @Test
    fun testSetConsentGranted_persistsTrue() {
        val consentManager = ConsentManager(mockContext)
        consentManager.setConsentGranted(true)

        verify { mockEditor.putBoolean(PreferencesConstants.KEY_GDPR_CONSENT, true) }
        verify { mockEditor.apply() }
    }

    @Test
    fun testSetConsentGranted_persistsFalse() {
        val consentManager = ConsentManager(mockContext)
        consentManager.setConsentGranted(false)

        verify { mockEditor.putBoolean(PreferencesConstants.KEY_GDPR_CONSENT, false) }
        verify { mockEditor.apply() }
    }
}
