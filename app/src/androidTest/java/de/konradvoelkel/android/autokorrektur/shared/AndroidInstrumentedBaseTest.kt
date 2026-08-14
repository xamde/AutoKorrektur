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

    protected val testContext: Context
        get() = InstrumentationRegistry.getInstrumentation().context

    protected val baseTempFiles = mutableListOf<File>()

    /**
     * Convenience helper to copy an asset into the app's cache directory.
     * This delegates to the canonical implementation in AndroidTestUtils and automatically
     * tracks the created temp file in [baseTempFiles] and [sink] for cleanup in @After.
     */
    protected fun cacheAsset(assetFileName: String, sink: MutableList<File>? = null): File {
        val f = AndroidTestUtils.copyAssetToCache(appContext, assetFileName)
        baseTempFiles.add(f)
        sink?.add(f)
        return f
    }

    /**
     * Determines whether a subtractive binary mask (where 0/black = detected vehicle)
     * contains vehicle pixels exceeding the threshold ratio.
     */
    protected fun hasCarDetection(mask: org.opencv.core.Mat, minRatio: Double = 0.01): Boolean {
        val total = mask.rows().toDouble() * mask.cols().toDouble()
        if (total <= 0) return false
        val blackMat = org.opencv.core.Mat()
        try {
            org.opencv.core.Core.inRange(
                mask,
                org.opencv.core.Scalar(0.0),
                org.opencv.core.Scalar(10.0),
                blackMat
            )
            val blackCount = org.opencv.core.Core.countNonZero(blackMat)
            return (blackCount.toDouble() / total) > minRatio
        } finally {
            blackMat.release()
        }
    }

    /**
     * Runs YOLO re-detection on an image Bitmap to verify whether vehicles remain (RF-70).
     */
    protected suspend fun hasVehicleInImage(
        yolo: de.konradvoelkel.android.autokorrektur.ml.api.YoloService,
        image: android.graphics.Bitmap,
        scoreThreshold: Float = 0.35f
    ): Boolean {
        val imageProcessor = PipelineTestFixtures.imageProcessor()
        val tempFile = File(appContext.cacheDir, "yolo_recheck_${System.nanoTime()}.jpg")
        baseTempFiles.add(tempFile)
        java.io.FileOutputStream(tempFile).use { image.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, it) }
        val processed = imageProcessor.processInputImage(
            imageUri = android.net.Uri.fromFile(tempFile),
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )
        val config = de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig(scoreThreshold = scoreThreshold)
        val result = yolo.inferDetailed(
            transformedMat = processed.transformedMat,
            xRatio = processed.xRatio,
            yRatio = processed.yRatio,
            upscaleFactor = 1.0f,
            overrideConfig = config
        )
        val hasDetections = result.detections.isNotEmpty()
        result.mask.release()
        return hasDetections
    }

    @org.junit.After
    fun baseTearDown() {
        baseTempFiles.forEach {
            try {
                if (it.exists()) it.delete()
            } catch (_: Exception) {}
        }
        baseTempFiles.clear()
        try {
            InstrumentationRegistry.getInstrumentation().waitForIdleSync()
        } catch (_: Exception) {}
        System.gc()
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