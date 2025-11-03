package de.konradvoelkel.android.autokorrektur.shared

import android.content.Context
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.BeforeClass
import java.io.File

/**
 * Base class for instrumented tests. Provides common setup and convenient accessors.
 */
open class AndroidInstrumentedBaseTest {

    protected val appContext: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    /**
     * Convenience helper to copy an asset into the app's cache directory.
     * This delegates to the canonical implementation in AndroidTestUtils and can optionally
     * track the created temp file in a provided [sink] list for later cleanup in @After.
     */
    protected fun cacheAsset(assetFileName: String, sink: MutableList<File>? = null): File {
        val f = AndroidTestUtils.copyAssetToCache(appContext, assetFileName)
        sink?.add(f)
        return f
    }

    companion object {
        @BeforeClass
        @JvmStatic
        fun beforeAllBase() {
            // Ensure OpenCV is initialized once per test class
            AndroidTestUtils.initOpenCV()
        }
    }
}