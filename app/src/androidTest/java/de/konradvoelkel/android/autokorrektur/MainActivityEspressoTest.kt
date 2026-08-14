package de.konradvoelkel.android.autokorrektur

import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import org.junit.Assert.assertNotNull
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
@LargeTest
class MainActivityEspressoTest {

    @get:Rule
    val activityRule = ActivityScenarioRule(MainActivity::class.java)

    @Test
    fun mainActivity_launchesSuccessfully() {
        activityRule.scenario.onActivity { activity ->
            assertNotNull("MainActivity should launch successfully", activity)
        }
    }

    @org.junit.After
    fun tearDown() {
        System.gc()
    }
}
