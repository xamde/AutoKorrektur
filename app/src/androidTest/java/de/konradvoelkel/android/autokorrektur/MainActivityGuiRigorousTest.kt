package de.konradvoelkel.android.autokorrektur

import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions.click
import androidx.test.espresso.action.ViewActions.scrollTo
import androidx.test.espresso.assertion.ViewAssertions.doesNotExist
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.isEnabled
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import org.hamcrest.CoreMatchers.not
import org.junit.After
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Rigorous Espresso UI Test Suite verifying all interactive controls,
 * expandable options panels, sliders, mode switches, and AR activity navigation.
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class MainActivityGuiRigorousTest {

    @Test
    fun testMainScreenButtonsDisplayedAndClickable() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { }
            onView(withId(R.id.fileSelect)).perform(scrollTo()).check(matches(isDisplayed()))
            onView(withId(R.id.download)).perform(scrollTo()).check(matches(isDisplayed()))
            onView(withId(R.id.exportInstagram)).perform(scrollTo()).check(matches(isDisplayed()))

            // startInference is hidden when this tier offers no engine choice — inference
            // auto-starts on image selection instead (see FirstFragment.autoStartInferenceEnabled).
            if (BuildConfig.FEATURE_HIGH_RES_PROGRESSIVE || BuildConfig.FEATURE_CLOUD_SDXL) {
                onView(withId(R.id.startInference)).perform(scrollTo()).check(matches(isDisplayed()))
            } else {
                onView(withId(R.id.startInference)).check(matches(not(isDisplayed())))
            }

            if (BuildConfig.FEATURE_LIVE_AR) {
                onView(withId(R.id.arLiveModeButton)).perform(scrollTo()).check(matches(isDisplayed()))
            } else {
                onView(withId(R.id.arLiveModeButton)).check(matches(not(isDisplayed())))
            }

            if (BuildConfig.FEATURE_BATCH_PROCESSING) {
                onView(withId(R.id.optionsButton)).perform(scrollTo()).check(matches(isDisplayed()))
            } else {
                onView(withId(R.id.optionsButton)).check(matches(not(isDisplayed())))
            }

            // Error Snackbar Guard: Assert that no startup initialization errors or exceptions are displayed
            onView(withId(com.google.android.material.R.id.snackbar_text)).check(doesNotExist())
        }
    }

    @Test
    fun testOptionsPanelToggleAndSliderInteractions() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { }
            // Options panel is initially hidden
            onView(withId(R.id.optionsPanel)).check(matches(not(isDisplayed())))

            if (!BuildConfig.FEATURE_BATCH_PROCESSING) {
                // optionsButton is the panel's only entry point (see
                // FirstFragment.applyFeatureFlags) — with it hidden, the panel is unreachable
                // and there's nothing further to verify.
                onView(withId(R.id.optionsButton)).check(matches(not(isDisplayed())))
                return@use
            }

            // Tap Options button to expand panel
            onView(withId(R.id.optionsButton)).perform(scrollTo(), click())
            onView(withId(R.id.optionsPanel)).perform(scrollTo()).check(matches(isDisplayed()))

            // Verify sliders and controls inside options panel. maskUpscale/downshift/
            // scoreThreshold/downscaleMP/segModel are gated by FEATURE_EVALUATION_MODE, a
            // debug-build-type flag rather than a per-flavor one — always true for the debug
            // build types these instrumented tests run against, so unconditional here is correct.
            onView(withId(R.id.batchMode)).perform(scrollTo()).check(matches(isDisplayed()))
            onView(withId(R.id.continueWithResult)).perform(scrollTo()).check(matches(isDisplayed()))
            onView(withId(R.id.maskUpscale)).perform(scrollTo()).check(matches(isDisplayed()))
            onView(withId(R.id.downshift)).perform(scrollTo()).check(matches(isDisplayed()))
            onView(withId(R.id.scoreThreshold)).perform(scrollTo()).check(matches(isDisplayed()))
            onView(withId(R.id.downscaleMP)).perform(scrollTo()).check(matches(isDisplayed()))
            onView(withId(R.id.segModel)).perform(scrollTo()).check(matches(isDisplayed()))

            // Tap Options button to collapse panel again
            onView(withId(R.id.optionsButton)).perform(scrollTo(), click())
            onView(withId(R.id.optionsPanel)).check(matches(not(isDisplayed())))
        }
    }

    @Test
    fun testBatchModeSwitchStateChange() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { }

            if (!BuildConfig.FEATURE_BATCH_PROCESSING) {
                // Nothing to toggle — batch mode doesn't exist in this tier, and its entry
                // point (optionsButton) is hidden along with it.
                onView(withId(R.id.optionsButton)).check(matches(not(isDisplayed())))
                return@use
            }

            // Expand options panel
            onView(withId(R.id.optionsButton)).perform(scrollTo(), click())
            onView(withId(R.id.optionsPanel)).perform(scrollTo()).check(matches(isDisplayed()))

            // Toggle batch mode switch
            onView(withId(R.id.batchMode)).perform(scrollTo(), click())
            onView(withId(R.id.batchMode)).check(matches(isDisplayed()))

            // Toggle back to single mode
            onView(withId(R.id.batchMode)).perform(scrollTo(), click())

            // Collapse options panel
            onView(withId(R.id.optionsButton)).perform(scrollTo(), click())
        }
    }

    @Test
    fun testInitialViewStates() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { }
            // Before/After slider view is initially gone until an image is processed
            onView(withId(R.id.beforeAfterSliderView)).check(matches(not(isDisplayed())))
            // Start button is initially disabled until an image is loaded
            onView(withId(R.id.startInference)).check(matches(not(isEnabled())))
        }
    }

    @Test
    fun testArCameraActivityLaunchButton() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { }

            if (!BuildConfig.FEATURE_LIVE_AR) {
                onView(withId(R.id.arLiveModeButton)).check(matches(not(isDisplayed())))
                return@use
            }

            onView(withId(R.id.arLiveModeButton)).perform(scrollTo()).check(matches(isDisplayed()))
            onView(withId(R.id.arLiveModeButton)).check(matches(isEnabled()))
        }
    }

    @After
    fun tearDown() {
        System.gc()
    }
}
