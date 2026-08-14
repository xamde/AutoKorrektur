package de.konradvoelkel.android.autokorrektur.manager

import android.content.Context
import android.content.SharedPreferences
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class QuotaManagerTest {

    private lateinit var context: Context
    private lateinit var prefs: SharedPreferences
    private lateinit var editor: SharedPreferences.Editor
    private val memoryPrefs = mutableMapOf<String, Any>()

    @Before
    fun setUp() {
        context = mockk(relaxed = true)
        prefs = mockk(relaxed = true)
        editor = mockk(relaxed = true)

        every { context.getSharedPreferences(any(), any()) } returns prefs
        every { prefs.edit() } returns editor

        every { prefs.getString(any(), any()) } answers {
            val key = firstArg<String>()
            val def = secondArg<String?>()
            (memoryPrefs[key] as? String) ?: def
        }
        every { prefs.getInt(any(), any()) } answers {
            val key = firstArg<String>()
            val def = secondArg<Int>()
            (memoryPrefs[key] as? Int) ?: def
        }
        every { prefs.getBoolean(any(), any()) } answers {
            val key = firstArg<String>()
            val def = secondArg<Boolean>()
            (memoryPrefs[key] as? Boolean) ?: def
        }

        every { editor.putString(any(), any()) } answers {
            memoryPrefs[firstArg()] = secondArg<String>()
            editor
        }
        every { editor.putInt(any(), any()) } answers {
            memoryPrefs[firstArg()] = secondArg<Int>()
            editor
        }
        every { editor.putBoolean(any(), any()) } answers {
            memoryPrefs[firstArg()] = secondArg<Boolean>()
            editor
        }
        every { editor.apply() } returns Unit
    }

    @Test
    fun testGetDeviceUuidGeneratesPersistentId() {
        val quotaManager = QuotaManager(context)
        val uuid1 = quotaManager.getDeviceUuid()
        assertNotNull(uuid1)
        assertTrue(uuid1.isNotEmpty())

        val uuid2 = quotaManager.getDeviceUuid()
        assertEquals(uuid1, uuid2)
    }

    @Test
    fun testDailyQuotaInitialState() {
        val quotaManager = QuotaManager(context)
        assertEquals(5, quotaManager.getRemainingDailyQuota())
        assertTrue(quotaManager.hasAvailableQuota())
    }

    @Test
    fun testConsumeQuotaDecrementsRemaining() {
        val quotaManager = QuotaManager(context)
        assertEquals(5, quotaManager.getRemainingDailyQuota())

        assertTrue(quotaManager.consumeQuota())
        assertEquals(4, quotaManager.getRemainingDailyQuota())

        assertTrue(quotaManager.consumeQuota())
        assertEquals(3, quotaManager.getRemainingDailyQuota())
    }

    @Test
    fun testConsumeQuotaExhaustion() {
        val quotaManager = QuotaManager(context)
        repeat(5) {
            assertTrue(quotaManager.consumeQuota())
        }
        assertEquals(0, quotaManager.getRemainingDailyQuota())
        assertFalse(quotaManager.hasAvailableQuota())
        assertFalse(quotaManager.consumeQuota())
    }

    @Test
    fun testDailyQuotaResetsOnNextDay() {
        var currentDate = "2026-08-14"
        val quotaManager = QuotaManager(context, dateProvider = { currentDate })

        // Exhaust all 5 quota points on Day 1
        repeat(5) {
            assertTrue(quotaManager.consumeQuota())
        }
        assertEquals(0, quotaManager.getRemainingDailyQuota())
        assertFalse(quotaManager.hasAvailableQuota())

        // Advance to Day 2
        currentDate = "2026-08-15"
        assertEquals(5, quotaManager.getRemainingDailyQuota())
        assertTrue(quotaManager.hasAvailableQuota())
        assertTrue(quotaManager.consumeQuota())
        assertEquals(4, quotaManager.getRemainingDailyQuota())
    }
}
