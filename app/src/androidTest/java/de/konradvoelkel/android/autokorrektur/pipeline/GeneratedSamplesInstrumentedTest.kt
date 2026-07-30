package de.konradvoelkel.android.autokorrektur.pipeline

import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import de.konradvoelkel.android.autokorrektur.shared.OpenCvTestUtils
import de.konradvoelkel.android.autokorrektur.shared.PipelineTestFixtures
import org.junit.After
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import java.io.File

@RunWith(AndroidJUnit4::class)
@LargeTest
class GeneratedSamplesInstrumentedTest : AndroidInstrumentedBaseTest() {

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
    fun testStreetSample_detectsCarAndValidatesReferenceOutsideMask() {
        val inputFile = cacheAsset("sample_street_with_car.jpg", tempFiles)
        val inputUri = Uri.fromFile(inputFile)

        val processedIn = imageProcessor.processInputImage(
            imageUri = inputUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )

        val mask = yoloInference.infer(
            transformedMat = processedIn.transformedMat,
            xRatio = processedIn.xRatio,
            yRatio = processedIn.yRatio,
            upscaleFactor = 1.2f,
            originalWidth = processedIn.originalMat.cols(),
            originalHeight = processedIn.originalMat.rows()
        )

        assertTrue("YOLO should detect car mask in street sample", hasCarDetection(mask))

        val refFile = cacheAsset("sample_street_without_car.jpg", tempFiles)
        val refBgr = Imgcodecs.imread(refFile.absolutePath, Imgcodecs.IMREAD_COLOR)
        assertNotNull("Reference image should load", refBgr)
        assertFalse("Reference image should not be empty", refBgr.empty())

        val refRgb = OpenCvTestUtils.matLoadedFromFileBgrToRgb(refBgr)
        refBgr.release()

        val inRgb8 = Mat()
        val refRgb8 = Mat()
        processedIn.originalMat.convertTo(inRgb8, org.opencv.core.CvType.CV_8UC3)
        refRgb.convertTo(refRgb8, org.opencv.core.CvType.CV_8UC3)

        val meanAbs = OpenCvTestUtils.meanAbsDiffOnMaskRgb8u3(mask, inRgb8, refRgb8)
        assertTrue("Background outside mask should match reference image (street meanAbs=$meanAbs)", meanAbs <= 35.0)

        inRgb8.release(); refRgb8.release(); refRgb.release(); mask.release()
    }

    @Test
    fun testSuburbSample_detectsCarAndValidatesReferenceOutsideMask() {
        val inputFile = cacheAsset("sample_suburb_with_car.jpg", tempFiles)
        val inputUri = Uri.fromFile(inputFile)

        val processedIn = imageProcessor.processInputImage(
            imageUri = inputUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )

        val mask = yoloInference.infer(
            transformedMat = processedIn.transformedMat,
            xRatio = processedIn.xRatio,
            yRatio = processedIn.yRatio,
            upscaleFactor = 1.2f,
            originalWidth = processedIn.originalMat.cols(),
            originalHeight = processedIn.originalMat.rows()
        )

        assertTrue("YOLO should detect car mask in suburb sample", hasCarDetection(mask))

        val refFile = cacheAsset("sample_suburb_without_car.jpg", tempFiles)
        val refBgr = Imgcodecs.imread(refFile.absolutePath, Imgcodecs.IMREAD_COLOR)
        assertNotNull("Reference image should load", refBgr)
        assertFalse("Reference image should not be empty", refBgr.empty())

        val refRgb = OpenCvTestUtils.matLoadedFromFileBgrToRgb(refBgr)
        refBgr.release()

        val inRgb8 = Mat()
        val refRgb8 = Mat()
        processedIn.originalMat.convertTo(inRgb8, org.opencv.core.CvType.CV_8UC3)
        refRgb.convertTo(refRgb8, org.opencv.core.CvType.CV_8UC3)

        val meanAbs = OpenCvTestUtils.meanAbsDiffOnMaskRgb8u3(mask, inRgb8, refRgb8)
        assertTrue("Background outside mask should match reference image (suburb meanAbs=$meanAbs)", meanAbs <= 35.0)

        inRgb8.release(); refRgb8.release(); refRgb.release(); mask.release()
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
