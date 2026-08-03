package de.konradvoelkel.android.autokorrektur.pipeline

import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.MediumTest
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
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
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * Mi-GAN inpainting behavior tests.
 */
@RunWith(AndroidJUnit4::class)
@MediumTest
class MiGanInpaintingInstrumentedTest : AndroidInstrumentedBaseTest() {

    private lateinit var yolo: YoloService
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var migan: MiGanInference
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
        yolo = PipelineTestFixtures.yolo()
        imageProcessor = PipelineTestFixtures.imageProcessor()
        migan = PipelineTestFixtures.migan()
    }

    @After
    fun tearDown() {
        tempFiles.forEach { it.delete() }
    }

    @Test
    fun testMiGanRemovesCarAndKeepsBackground() = kotlinx.coroutines.runBlocking {
        // Load input photo and reference mask
        val photoFile = cacheAsset("photo_with_car_1.png", tempFiles)
        val photoUri = Uri.fromFile(photoFile)
        val refMaskFile = cacheAsset("photo_with_car_1_mask.png", tempFiles)
        val referenceMask = Imgcodecs.imread(refMaskFile.absolutePath, Imgcodecs.IMREAD_GRAYSCALE)
        assertNotNull("Reference mask should load", referenceMask)
        assertTrue("Reference mask should not be empty", !referenceMask.empty())

        // Prepare original image as Mat via ImageProcessor (to reuse decoding logic)
        val processed = imageProcessor.processInputImage(
            imageUri = photoUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )

        // Run Mi-GAN inference using original image size and the reference mask
        val inpainted = migan.inferMiGan(processed.originalMat, referenceMask)

        // Save RGB->BGR debug output if enabled
        if (OpenCvTestUtils.shouldWriteDebugArtifacts(appContext)) {
            val outDir = appContext.getExternalFilesDir(null)
            if (outDir != null) {
                val inpaintedPath = File(outDir, "debug_migan_inpainted.png").absolutePath
                val bgrDebug = Mat()
                Imgproc.cvtColor(inpainted, bgrDebug, Imgproc.COLOR_RGB2BGR)
                Imgcodecs.imwrite(inpaintedPath, bgrDebug)
                bgrDebug.release()
            }
        }

        // 1) Verify no car remains in the inpainted image via YOLO mask
        val tempOutFile = File(appContext.cacheDir, "migan_inpainted_tmp.png")
        val bgrTmp = Mat()
        Imgproc.cvtColor(inpainted, bgrTmp, Imgproc.COLOR_RGB2BGR)
        Imgcodecs.imwrite(tempOutFile.absolutePath, bgrTmp)
        bgrTmp.release()
        tempFiles.add(tempOutFile)
        val outUri = Uri.fromFile(tempOutFile)

        val processedOut = imageProcessor.processInputImage(
            imageUri = outUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )

        val yoloMaskOnOutput = yolo.infer(
            transformedMat = processedOut.transformedMat,
            xRatio = processedOut.xRatio,
            yRatio = processedOut.yRatio,
            upscaleFactor = 1.02f,
            originalWidth = processedOut.originalMat.cols(),
            originalHeight = processedOut.originalMat.rows()
        )

        assertFalse(
            "After Mi-GAN, there should be no car detected",
            hasCarDetection(yoloMaskOnOutput)
        )

        // 2) Verify in non-car (white) regions, output roughly agrees with input
        assertEquals(processed.originalMat.rows(), inpainted.rows())
        assertEquals(processed.originalMat.cols(), inpainted.cols())
        assertEquals(referenceMask.rows(), inpainted.rows())
        assertEquals(referenceMask.cols(), inpainted.cols())

        val inRgb8 = Mat()
        val outRgb8 = Mat()
        processed.originalMat.convertTo(inRgb8, CvType.CV_8UC3)
        inpainted.convertTo(outRgb8, CvType.CV_8UC3)
        val meanAbs = OpenCvTestUtils.meanAbsDiffOnMaskRgb8u3(referenceMask, inRgb8, outRgb8)
        val tolerancePerChannel = 10.0
        assertTrue(
            "Background should be preserved (mean abs diff per channel <= $tolerancePerChannel)",
            meanAbs <= tolerancePerChannel
        )

        // cleanup
        inRgb8.release(); outRgb8.release(); yoloMaskOnOutput.release(); inpainted.release(); referenceMask.release()
    }

    private fun hasCarDetection(mask: Mat): Boolean {
        val totalPixels = mask.rows() * mask.cols()
        if (totalPixels == 0) return false
        val carMask = Mat()
        org.opencv.core.Core.inRange(
            mask,
            org.opencv.core.Scalar(0.0),
            org.opencv.core.Scalar(10.0),
            carMask
        )
        val carPixels = org.opencv.core.Core.countNonZero(carMask)
        carMask.release()
        val carPixelRatio = carPixels.toDouble() / totalPixels.toDouble()
        android.util.Log.d("MiGanTest", "Car pixel ratio after inpainting: $carPixelRatio ($carPixels / $totalPixels)")
        return carPixelRatio > 0.20
    }
}
