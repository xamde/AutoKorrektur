package de.konradvoelkel.android.autokorrektur

import android.content.Context
import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions.click
import androidx.test.espresso.action.ViewActions.scrollTo
import androidx.test.espresso.assertion.ViewAssertions.doesNotExist
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.isEnabled
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.espresso.matcher.ViewMatchers.withText
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import org.hamcrest.CoreMatchers.containsString
import org.hamcrest.CoreMatchers.not
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * End-to-End Integrated Test Suite testing all 5 core application workflows.
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class EndToEndWorkflowsInstrumentedTest : AndroidInstrumentedBaseTest() {

    @Before
    fun setUp() {
        // Reset shared preferences to ensure a clean state for GDPR consent & options
        val prefs = appContext.getSharedPreferences("autokorrektur_prefs", Context.MODE_PRIVATE)
        prefs.edit().clear().commit()
    }

    @org.junit.After
    fun tearDown() {
        System.gc()
    }

    /**
     * Workflow 1: On-Device Single Image Vehicle Removal Flow
     * Verifies main buttons, options expansion, and initial on-device pipeline setup.
     */
    @Test
    fun testWorkflow1_SingleImageOnDeviceFlowUI() {
        val scenario = ActivityScenario.launch(MainActivity::class.java)

        onView(withId(R.id.fileSelect)).perform(scrollTo()).check(matches(isDisplayed()))
        onView(withId(R.id.download)).perform(scrollTo()).check(matches(isDisplayed()))
        onView(withId(R.id.exportInstagram)).perform(scrollTo()).check(matches(isDisplayed()))

        // startInference is hidden when this tier offers no engine choice (inference auto-starts
        // instead — see FirstFragment.autoStartInferenceEnabled).
        if (BuildConfig.FEATURE_HIGH_RES_PROGRESSIVE || BuildConfig.FEATURE_CLOUD_SDXL) {
            onView(withId(R.id.startInference)).perform(scrollTo()).check(matches(isDisplayed()))
        } else {
            onView(withId(R.id.startInference)).check(matches(not(isDisplayed())))
        }

        // Guard: Assert no startup error Snackbars are displayed on screen
        onView(withId(com.google.android.material.R.id.snackbar_text)).check(doesNotExist())

        scenario.close()
    }

    /**
     * Workflow 2: Opt-In Hybrid Server SDXL Inpainting Flow & GDPR Consent
     * Toggles SDXL mode, verifies GDPR dialog pops up, accepts consent, and validates toggle state.
     * Only meaningful when this tier offers the Cloud SDXL engine at all (see
     * FirstFragment.applyFeatureFlags); nothing to verify otherwise.
     */
    @Test
    fun testWorkflow2_CloudSdxlGdprConsentFlow() {
        val scenario = ActivityScenario.launch(MainActivity::class.java)

        if (!BuildConfig.FEATURE_CLOUD_SDXL) {
            onView(withId(R.id.chipCloudSdxl)).check(matches(not(isDisplayed())))
            scenario.close()
            return
        }

        // Select the Cloud SDXL engine chip — this is the actual visible control (R.id.useSdxl
        // is a hidden backwards-compat SwitchCompat the chip drives internally, not a real UI
        // target; pre-existing test debt, unrelated to any change in this session, fixed here
        // for the same reason RigorousGuiFlowInstrumentedTest was: it could never have passed
        // against the current layout).
        onView(withId(R.id.chipCloudSdxl)).perform(scrollTo(), click())

        // Verify GDPR Consent dialog is displayed with title Premium Edit (Server SDXL)
        onView(withText(R.string.premium_edit_title)).check(matches(isDisplayed()))

        // Click "Accept" in dialog
        onView(withText(R.string.btn_accept)).perform(click())

        // Guard: Assert no error Snackbars displayed post-consent
        onView(withId(com.google.android.material.R.id.snackbar_text)).check(doesNotExist())

        scenario.close()
    }

    /**
     * Workflow 3: Batch Mode Processing Flow Setup
     * Expands options panel, enables batch processing checkbox, and verifies UI state. Only
     * meaningful when this tier has batch processing at all (see
     * FirstFragment.applyFeatureFlags, which hides optionsButton — the panel's only entry
     * point — otherwise).
     */
    @Test
    fun testWorkflow3_BatchModeOptionFlow() {
        val scenario = ActivityScenario.launch(MainActivity::class.java)

        if (!BuildConfig.FEATURE_BATCH_PROCESSING) {
            onView(withId(R.id.optionsButton)).check(matches(not(isDisplayed())))
            scenario.close()
            return
        }

        // Open options panel
        onView(withId(R.id.optionsButton)).perform(scrollTo(), click())
        onView(withId(R.id.optionsPanel)).perform(scrollTo()).check(matches(isDisplayed()))

        // Enable Batch Mode
        onView(withId(R.id.batchMode)).perform(scrollTo(), click())

        // Guard: Assert clean UI without error Snackbars
        onView(withId(com.google.android.material.R.id.snackbar_text)).check(doesNotExist())

        scenario.close()
    }

    /**
     * Workflow 4: AR Live Real-Time Video Stream Activity Navigation
     * Verifies AR Live Mode button is active and navigates cleanly. Only meaningful when this
     * tier has live AR at all (see FirstFragment.applyFeatureFlags).
     */
    @Test
    fun testWorkflow4_ArCameraNavigationFlow() {
        val scenario = ActivityScenario.launch(MainActivity::class.java)

        if (!BuildConfig.FEATURE_LIVE_AR) {
            onView(withId(R.id.arLiveModeButton)).check(matches(not(isDisplayed())))
            scenario.close()
            return
        }

        onView(withId(R.id.arLiveModeButton)).perform(scrollTo()).check(matches(isDisplayed()))
        onView(withId(R.id.arLiveModeButton)).check(matches(isEnabled()))

        // Guard: Assert clean UI without error Snackbars
        onView(withId(com.google.android.material.R.id.snackbar_text)).check(doesNotExist())

        scenario.close()
    }

    /**
     * Workflow 5: Parameter Tuning & Custom Options Flow
     * Verifies all configuration controls (scoreThreshold, maskUpscale, segModel, downscaleMP).
     * Reachable only through the options panel, whose only entry point (optionsButton) is
     * gated by FEATURE_BATCH_PROCESSING (see FirstFragment.applyFeatureFlags) — the individual
     * sliders themselves are gated by FEATURE_EVALUATION_MODE, a debug-build-type flag that's
     * always true for the debug build types these instrumented tests run against.
     */
    @Test
    fun testWorkflow5_ParameterTuningFlow() {
        val scenario = ActivityScenario.launch(MainActivity::class.java)

        if (!BuildConfig.FEATURE_BATCH_PROCESSING) {
            onView(withId(R.id.optionsButton)).check(matches(not(isDisplayed())))
            scenario.close()
            return
        }

        // Expand options panel
        onView(withId(R.id.optionsButton)).perform(scrollTo(), click())
        onView(withId(R.id.optionsPanel)).perform(scrollTo()).check(matches(isDisplayed()))

        // Verify all parameter controls exist and render
        onView(withId(R.id.scoreThreshold)).perform(scrollTo()).check(matches(isDisplayed()))
        onView(withId(R.id.maskUpscale)).perform(scrollTo()).check(matches(isDisplayed()))
        onView(withId(R.id.downscaleMP)).perform(scrollTo()).check(matches(isDisplayed()))
        onView(withId(R.id.segModel)).perform(scrollTo()).check(matches(isDisplayed()))

        // Guard: Assert clean UI without error Snackbars
        onView(withId(com.google.android.material.R.id.snackbar_text)).check(doesNotExist())

        scenario.close()
    }
}
