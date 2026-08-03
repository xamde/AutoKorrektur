package de.konradvoelkel.android.autokorrektur

import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions.click
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
@LargeTest
class RigorousGuiFlowInstrumentedTest {

    @Test
    fun testGuiElementsAndSdxlToggleVisibility() {
        val scenario = ActivityScenario.launch(MainActivity::class.java)

        // Verify select image button is displayed
        onView(withId(R.id.fileSelect)).check(matches(isDisplayed()))

        // Verify start inference button is displayed
        onView(withId(R.id.startInference)).check(matches(isDisplayed()))

        // Verify Premium Edit (Server SDXL) toggle is displayed on main UI
        onView(withId(R.id.useSdxl)).check(matches(isDisplayed()))

        // Toggle SDXL switch
        onView(withId(R.id.useSdxl)).perform(click())

        // Error Snackbar Guard: Assert no startup error Snackbars are displayed on screen
        onView(withId(com.google.android.material.R.id.snackbar_text)).check(androidx.test.espresso.assertion.ViewAssertions.doesNotExist())

        scenario.close()
    }
}
