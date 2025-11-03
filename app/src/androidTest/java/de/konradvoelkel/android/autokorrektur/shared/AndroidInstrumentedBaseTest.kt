package de.konradvoelkel.android.autokorrektur.shared

import android.content.Context
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.BeforeClass

/**
 * Base class for instrumented tests. Provides common setup and convenient accessors.
 */
open class AndroidInstrumentedBaseTest {

    protected val appContext: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    companion object {
        @BeforeClass
        @JvmStatic
        fun beforeAllBase() {
            // Ensure OpenCV is initialized once per test class
            AndroidTestUtils.initOpenCV()
        }
    }
}