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
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import java.io.File

/**
 * Reference comparisons for masks and inpainted outputs.
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class ReferenceComparisonInstrumentedTest : AndroidInstrumentedBaseTest() {

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
    fun testCarMaskMatchesReference() {
        val photoFile = cacheAsset("photo_with_car_1.png", tempFiles)
        val photoUri = Uri.fromFile(photoFile)

        val processedImage = imageProcessor.processInputImage(
            imageUri = photoUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )

        val resultMask = yoloInference.infer(
            transformedMat = processedImage.transformedMat,
            xRatio = processedImage.xRatio,
            yRatio = processedImage.yRatio,
            upscaleFactor = 1.2f,
            originalWidth = processedImage.originalMat.cols(),
            originalHeight = processedImage.originalMat.rows()
        )

        val refMaskFile = cacheAsset("photo_with_car_1_mask.png", tempFiles)
        val referenceMask = Imgcodecs.imread(refMaskFile.absolutePath, Imgcodecs.IMREAD_GRAYSCALE)

        assertNotNull("Reference mask should load", referenceMask)
        assertTrue("Reference mask should not be empty", !referenceMask.empty())

        assertEquals(processedImage.originalMat.rows(), resultMask.rows())
        assertEquals(processedImage.originalMat.cols(), resultMask.cols())
        assertEquals(resultMask.rows(), referenceMask.rows())
        assertEquals(resultMask.cols(), referenceMask.cols())

        // Binarize: treat near-black as car (255), others 0
        val binResult = Mat()
        val binReference = Mat()
        org.opencv.core.Core.inRange(
            resultMask,
            org.opencv.core.Scalar(0.0),
            org.opencv.core.Scalar(10.0),
            binResult
        )
        org.opencv.core.Core.inRange(
            referenceMask,
            org.opencv.core.Scalar(0.0),
            org.opencv.core.Scalar(10.0),
            binReference
        )

        // Compute XOR to find mismatches
        val diff = Mat()
        org.opencv.core.Core.bitwise_xor(binResult, binReference, diff)
        val mismatches = org.opencv.core.Core.countNonZero(diff)
        val totalPixels = diff.rows() * diff.cols()
        val agreement = 1.0 - (mismatches.toDouble() / totalPixels.toDouble())

        assertTrue("Generated mask should agree with reference mask >= 95%", agreement >= 0.95)

        // Cleanup
        diff.release(); binResult.release(); binReference.release(); referenceMask.release(); resultMask.release()
    }

    @Test
    fun testEndToEndMiGanOnExample2MatchesReferenceOutsideMask() {
        // 1) Load input and process
        val inputFile = cacheAsset("example2.png", tempFiles)
        val inputUri = Uri.fromFile(inputFile)
        val processedIn = imageProcessor.processInputImage(
            imageUri = inputUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )

        // 2) Build YOLO mask at original size (black = car), will serve as "allowed difference" mask
        val mask = yoloInference.infer(
            transformedMat = processedIn.transformedMat,
            xRatio = processedIn.xRatio,
            yRatio = processedIn.yRatio,
            upscaleFactor = 1.2f,
            originalWidth = processedIn.originalMat.cols(),
            originalHeight = processedIn.originalMat.rows()
        )

        // 3) Load reference result and convert BGR->RGB for fair comparison
        val refFile = cacheAsset("example2Result.png", tempFiles)
        val refBgr = Imgcodecs.imread(refFile.absolutePath, Imgcodecs.IMREAD_COLOR)
        assertTrue("Reference image should load", !refBgr.empty())
        val refRgb = OpenCvTestUtils.matLoadedFromFileBgrToRgb(refBgr)
        refBgr.release()

        // 4) Ensure sizes match
        assertEquals(processedIn.originalMat.rows(), refRgb.rows())
        assertEquals(processedIn.originalMat.cols(), refRgb.cols())

        // 5) Compute similarity over white regions of mask (non-car)
        val inRgb8 = Mat()
        val refRgb8 = Mat()
        processedIn.originalMat.convertTo(inRgb8, org.opencv.core.CvType.CV_8UC3)
        refRgb.convertTo(refRgb8, org.opencv.core.CvType.CV_8UC3)
        val meanAbs = OpenCvTestUtils.meanAbsDiffOnMaskRgb8u3(mask, inRgb8, refRgb8)

        val tolerancePerChannel = 12.0
        assertTrue(
            "Inpainted output should match reference outside masked regions (<= $tolerancePerChannel per-channel)",
            meanAbs <= tolerancePerChannel
        )

        // 6) Optional: verify no car remains after inpainting by running YOLO on the provided reference result too
        val tempOutFile = File(appContext.cacheDir, "example2_ref_out.png")
        OpenCvTestUtils.saveDebugRgbMatAsPngBgr(refRgb, tempOutFile)
        tempFiles.add(tempOutFile)
        val outUri = Uri.fromFile(tempOutFile)
        val processedOut = imageProcessor.processInputImage(
            imageUri = outUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )
        val maskAfter = yoloInference.infer(
            transformedMat = processedOut.transformedMat,
            xRatio = processedOut.xRatio,
            yRatio = processedOut.yRatio,
            upscaleFactor = 1.2f,
            originalWidth = processedOut.originalMat.cols(),
            originalHeight = processedOut.originalMat.rows()
        )
        assertFalse(
            "After inpainting, there should be no car detected (example2)",
            hasCarDetection(maskAfter)
        )

        // Cleanup
        inRgb8.release(); refRgb8.release(); refRgb.release(); maskAfter.release(); mask.release()
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
