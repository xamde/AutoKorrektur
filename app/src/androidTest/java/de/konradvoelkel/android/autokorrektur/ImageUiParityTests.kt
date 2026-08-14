package de.konradvoelkel.android.autokorrektur

import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

@RunWith(AndroidJUnit4::class)
class ImageUiParityTests :
    de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest() {

    private lateinit var yolo: YoloService
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var migan: MiGanInference

    private val tempFiles = mutableListOf<File>()

    @Before
    fun setup() = kotlinx.coroutines.runBlocking {
        yolo = YoloServiceImpl(YoloTFLiteEngine(appContext))
        yolo.initialize("yolo11s")
        imageProcessor = ImageProcessor(appContext)
        migan = MiGanInference(appContext)
        migan.initialize()
    }

    @After
    fun tearDown() {
        tempFiles.forEach { it.delete() }
        yolo.close()
        migan.close()
    }


    @Test
    fun testUiPathUsesOriginalImageSizeForMiGan() = kotlinx.coroutines.runBlocking {
        // Use example2, which is already covered elsewhere, to compare two inpainting paths
        val inputFile = cacheAsset("photo_with_car_1.png", tempFiles)
        val inputUri = Uri.fromFile(inputFile)

        val processed = imageProcessor.processInputImage(
            imageUri = inputUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )

        var a: Mat? = null
        var b: Mat? = null
        var bResized: Mat? = null
        var mask: Mat? = null

        try {
            // Build mask at original size (black = car, white = background)
            mask = yolo.infer(
                transformedMat = processed.transformedMat,
                xRatio = processed.xRatio,
                yRatio = processed.yRatio,
                upscaleFactor = 1.2f,
                originalWidth = processed.originalMat.cols(),
                originalHeight = processed.originalMat.rows()
            )

            // Path A (expected/UX-correct): run Mi-GAN on original image aligned with mask
            a = migan.inferMiGan(processed.originalMat, mask)

            // Verify Path A returns expected full-resolution dimensions matching original image
            assertEquals("Mi-GAN result width should match original width", processed.originalMat.cols(), a.cols())
            assertEquals("Mi-GAN result height should match original height", processed.originalMat.rows(), a.rows())
            assertFalse("Mi-GAN result should not be empty", a.empty())
        } finally {
            a?.release()
            b?.release()
            bResized?.release()
            mask?.release()
            processed.release(recycleBitmaps = true)
        }
    }

}
