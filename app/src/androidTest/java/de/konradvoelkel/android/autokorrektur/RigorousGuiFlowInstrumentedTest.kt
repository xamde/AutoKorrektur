package de.konradvoelkel.android.autokorrektur

import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions.click
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import org.hamcrest.CoreMatchers.not
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
@LargeTest
class RigorousGuiFlowInstrumentedTest {

    @Test
    fun testGuiElementsAndSdxlToggleVisibility() {
        val scenario = ActivityScenario.launch(MainActivity::class.java)
        scenario.onActivity { }

        // Verify select image button is displayed
        onView(withId(R.id.fileSelect)).check(matches(isDisplayed()))

        // startInference is hidden when this tier offers no engine choice (inference auto-starts
        // instead — see FirstFragment.autoStartInferenceEnabled).
        if (BuildConfig.FEATURE_HIGH_RES_PROGRESSIVE || BuildConfig.FEATURE_CLOUD_SDXL) {
            onView(withId(R.id.startInference)).check(matches(isDisplayed()))
        } else {
            onView(withId(R.id.startInference)).check(matches(not(isDisplayed())))
        }

        // Verify the inpainting engine picker (Fast/High-Res/Cloud SDXL chips) is displayed on
        // the main UI — only when this tier actually offers a choice (see
        // FirstFragment.applyFeatureFlags, which hides the whole card otherwise). Pre-existing
        // test debt fixed here: this used to assert R.id.useSdxl, a SwitchCompat superseded by
        // the chip picker below and left `visibility="gone"` in the layout as a backwards-compat
        // stub ever since — that assertion could never have passed against the current layout,
        // confirmed unrelated to any change in this session.
        if (BuildConfig.FEATURE_HIGH_RES_PROGRESSIVE || BuildConfig.FEATURE_CLOUD_SDXL) {
            onView(withId(R.id.chipGroupQuality)).check(matches(isDisplayed()))
        } else {
            onView(withId(R.id.chipGroupQuality)).check(matches(not(isDisplayed())))
        }

        // Error Snackbar Guard: Assert no startup error Snackbars are displayed on screen
        onView(withId(com.google.android.material.R.id.snackbar_text)).check(androidx.test.espresso.assertion.ViewAssertions.doesNotExist())

        scenario.close()
    }

    @org.junit.After
    fun tearDown() {
        System.gc()
    }
}
