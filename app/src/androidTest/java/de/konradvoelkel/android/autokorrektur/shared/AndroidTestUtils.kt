package de.konradvoelkel.android.autokorrektur.shared

import android.content.Context
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.fail
import org.opencv.android.OpenCVLoader
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * Android-specific test utilities for instrumented tests.
 * These intentionally depend on Instrumentation APIs and should not be used in JVM tests.
 */
object AndroidTestUtils {

    fun initOpenCV() {
        try {
            if (!OpenCVLoader.initLocal()) {
                fail("OpenCV initialization failed")
            }
        } catch (e: Exception) {
            fail("OpenCV initialization check failed: ${e.message}")
        }
    }

    @Throws(IOException::class)
    fun copyAssetToCache(context: Context, assetFileName: String): File {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val assetManager = testContext.assets
        val inputStream = assetManager.open(assetFileName)
        val tempFile = File(context.cacheDir, assetFileName)
        val outputStream = FileOutputStream(tempFile)
        inputStream.use { input ->
            outputStream.use { output ->
                input.copyTo(output)
            }
        }
        return tempFile
    }
}