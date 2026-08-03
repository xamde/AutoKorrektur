package de.konradvoelkel.android.autokorrektur.pipeline

import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import de.konradvoelkel.android.autokorrektur.shared.PipelineTestFixtures
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.Mat
import java.io.File

@RunWith(AndroidJUnit4::class)
@LargeTest
class PortraitImageInstrumentedTest : AndroidInstrumentedBaseTest() {

    private lateinit var yoloInference: YoloService
    private lateinit var imageProcessor: ImageProcessor
    private val tempFiles = mutableListOf<File>()

    companion object {
        @org.junit.AfterClass
        @JvmStatic
        fun tearDownFixtures() {
            PipelineTestFixtures.closeAll()
        }
    }

    @Before
    fun setUp() {
        yoloInference = PipelineTestFixtures.yolo()
        imageProcessor = PipelineTestFixtures.imageProcessor()
    }

    @After
    fun tearDown() {
        tempFiles.forEach { it.delete() }
    }

    @Test
    fun testPortraitImage_loadsAndGeneratesValidMask() = kotlinx.coroutines.runBlocking {
        val inputFile = cacheAsset("portraitcar.jpg", tempFiles)
        val inputUri = Uri.fromFile(inputFile)

        val processedIn = imageProcessor.processInputImage(
            imageUri = inputUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = 4.0f
        )

        assertNotNull("Processed portrait image should not be null", processedIn)
        assertTrue("Height should be greater than width for portrait image", processedIn.originalBitmap.height > processedIn.originalBitmap.width)

        val mask = yoloInference.infer(
            transformedMat = processedIn.transformedMat,
            xRatio = processedIn.xRatio,
            yRatio = processedIn.yRatio,
            upscaleFactor = 1.2f,
            originalWidth = processedIn.originalMat.cols(),
            originalHeight = processedIn.originalMat.rows()
        )

        assertEquals("Mask height should match original image height", processedIn.originalMat.rows(), mask.rows())
        assertEquals("Mask width should match original image width", processedIn.originalMat.cols(), mask.cols())
        assertTrue("Car mask should be detected in portrait image", hasCarDetection(mask))

        mask.release()
        processedIn.release()
    }

    private fun hasCarDetection(mask: Mat): Boolean {
        val totalPixels = mask.rows() * mask.cols()
        if (totalPixels == 0) return false
        val blackMask = Mat()
        org.opencv.core.Core.inRange(
            mask,
            org.opencv.core.Scalar(0.0),
            org.opencv.core.Scalar(10.0),
            blackMask
        )
        val blackPixels = org.opencv.core.Core.countNonZero(blackMask)
        blackMask.release()
        val blackPixelRatio = blackPixels.toDouble() / totalPixels.toDouble()
        return blackPixelRatio > 0.01
    }
}
