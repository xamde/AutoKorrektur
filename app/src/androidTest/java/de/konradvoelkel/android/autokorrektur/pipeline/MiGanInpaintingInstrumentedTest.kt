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
import org.opencv.core.Core
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

        // Convert reference mask (255=car) to pipeline standard format (0=car, 255=background)
        val standardMask = Mat()
        Core.bitwise_not(referenceMask, standardMask)

        // Run Mi-GAN inference using original image size and the standard mask
        val inpainted = migan.inferMiGan(processed.originalMat, standardMask)
        standardMask.release()

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

        // 1) Verify car pixels were modified (inpainted) in the car hole
        val inRgb8 = Mat()
        val outRgb8 = Mat()
        processed.originalMat.convertTo(inRgb8, CvType.CV_8UC3)
        inpainted.convertTo(outRgb8, CvType.CV_8UC3)

        // 1) Verify car pixels were modified (inpainted) in the car hole (referenceMask > 0)
        val carDiff = OpenCvTestUtils.meanAbsDiffOnMaskRgb8u3(referenceMask, inRgb8, outRgb8)
        assertTrue("Car region must be inpainted (mean diff >= 10.0, actual: $carDiff)", carDiff >= 10.0)

        // 2) Verify in non-car (background) regions, output agrees with input
        val bgMask = Mat()
        org.opencv.core.Core.bitwise_not(referenceMask, bgMask)
        val bgDiff = OpenCvTestUtils.meanAbsDiffOnMaskRgb8u3(bgMask, inRgb8, outRgb8)
        bgMask.release()

        val tolerancePerChannel = 2.0
        assertTrue(
            "Background should be preserved (mean abs diff per channel <= $tolerancePerChannel, actual: $bgDiff)",
            bgDiff <= tolerancePerChannel
        )

        // cleanup
        inRgb8.release()
        outRgb8.release()
        inpainted.release()
        referenceMask.release()
    }
}
