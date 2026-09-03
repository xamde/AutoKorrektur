package de.konradvoelkel.android.autokorrektur

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ApplicationContextTest {
    @Test
    fun useAppContext() {
        // Context of the app under test. Compare against BuildConfig.APPLICATION_ID rather than
        // a hardcoded literal — product flavors other than `core` carry an applicationIdSuffix
        // (see docs/MVP_FEATURE_FLAG_PLAN.md §4), so the actual package name varies per flavor.
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals(BuildConfig.APPLICATION_ID, appContext.packageName)
    }
}
