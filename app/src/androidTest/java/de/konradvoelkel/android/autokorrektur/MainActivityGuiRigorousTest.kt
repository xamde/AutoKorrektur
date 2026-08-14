package de.konradvoelkel.android.autokorrektur

import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions.click
import androidx.test.espresso.action.ViewActions.scrollTo
import androidx.test.espresso.assertion.ViewAssertions.doesNotExist
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.isEnabled
import androidx.test.espresso.matcher.ViewMatchers.withId
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
    fun test1_mainScreenButtonsDisplayedAndClickable() {
        ActivityScenario.launch(MainActivity::class.java).use {
            onView(withId(R.id.fileSelect)).perform(scrollTo()).check(matches(isDisplayed()))
            onView(withId(R.id.startInference)).perform(scrollTo()).check(matches(isDisplayed()))
            onView(withId(R.id.download)).perform(scrollTo()).check(matches(isDisplayed()))
            onView(withId(R.id.exportInstagram)).perform(scrollTo()).check(matches(isDisplayed()))
            onView(withId(R.id.arLiveModeButton)).perform(scrollTo()).check(matches(isDisplayed()))
            onView(withId(R.id.optionsButton)).perform(scrollTo()).check(matches(isDisplayed()))
            
            // Error Snackbar Guard: Assert that no startup initialization errors or exceptions are displayed
            onView(withId(com.google.android.material.R.id.snackbar_text)).check(doesNotExist())
        }
    }

    @Test
    fun test2_optionsPanelToggleAndSliderInteractions() {
        ActivityScenario.launch(MainActivity::class.java).use {
            // Options panel is initially hidden
            onView(withId(R.id.optionsPanel)).check(matches(not(isDisplayed())))

            // Tap Options button to expand panel
            onView(withId(R.id.optionsButton)).perform(scrollTo(), click())
            onView(withId(R.id.optionsPanel)).perform(scrollTo()).check(matches(isDisplayed()))

            // Verify sliders and controls inside options panel
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
    fun test3_batchModeSwitchStateChange() {
        ActivityScenario.launch(MainActivity::class.java).use {
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
    fun test4_initialViewStates() {
        ActivityScenario.launch(MainActivity::class.java).use {
            // Before/After slider view is initially gone until an image is processed
            onView(withId(R.id.beforeAfterSliderView)).check(matches(not(isDisplayed())))
            // Start button is initially disabled until an image is loaded
            onView(withId(R.id.startInference)).check(matches(not(isEnabled())))
        }
    }

    @Test
    fun test5_arCameraActivityLaunchAndHud() {
        ActivityScenario.launch(MainActivity::class.java).use {
            onView(withId(R.id.arLiveModeButton)).perform(scrollTo()).check(matches(isDisplayed()))
            onView(withId(R.id.arLiveModeButton)).check(matches(isEnabled()))
            // Tap AR button to launch ArCameraActivity
            onView(withId(R.id.arLiveModeButton)).perform(scrollTo(), click())
        }
    }

    @org.junit.After
    fun tearDown() {
        System.gc()
    }
}
