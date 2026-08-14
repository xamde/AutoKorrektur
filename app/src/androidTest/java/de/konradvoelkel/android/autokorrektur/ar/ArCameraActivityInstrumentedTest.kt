package de.konradvoelkel.android.autokorrektur.ar

import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions.click
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.ext.junit.runners.AndroidJUnit4
import de.konradvoelkel.android.autokorrektur.R
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ArCameraActivityInstrumentedTest : AndroidInstrumentedBaseTest() {

    @Test
    fun testArCameraActivity_viewsDisplayedAndInteractions() {
        ActivityScenario.launch(ArCameraActivity::class.java).use { scenario ->
            // Verify all key AR HUD controls are displayed
            onView(withId(R.id.cameraPreview)).check(matches(isDisplayed()))
            onView(withId(R.id.arOverlayView)).check(matches(isDisplayed()))
            onView(withId(R.id.resetArButton)).check(matches(isDisplayed()))
            onView(withId(R.id.captureArButton)).check(matches(isDisplayed()))
            onView(withId(R.id.arFpsBadge)).check(matches(isDisplayed()))

            // Click reset AR button
            onView(withId(R.id.resetArButton)).perform(click())

            // Click capture AR button
            onView(withId(R.id.captureArButton)).perform(click())
        }
    }
}
