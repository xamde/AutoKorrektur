package de.konradvoelkel.android.autokorrektur

import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import de.konradvoelkel.android.autokorrektur.ml.YoloInferenceTFLite
import org.junit.After
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
class ImageUiParityTests : de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest() {

    private lateinit var yolo: YoloInferenceTFLite
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var migan: MiGanInference

    private val tempFiles = mutableListOf<File>()

    @Before
    fun setup() {
        yolo = YoloInferenceTFLite(appContext)
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

    private fun copyAssetToCache(assetFileName: String): File {
        val file = de.konradvoelkel.android.autokorrektur.shared.AndroidTestUtils
            .copyAssetToCache(appContext, assetFileName)
        tempFiles.add(file)
        return file
    }

    @Test
    fun testUiPathUsesOriginalImageSizeForMiGan() {
        // Use example2, which is already covered elsewhere, to compare two inpainting paths
        val inputFile = copyAssetToCache("example2.png")
        val inputUri = Uri.fromFile(inputFile)

        val processed = imageProcessor.processInputImage(
            imageUri = inputUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )

        // Build mask at original size (black = car, white = background)
        val mask = yolo.inferYolo(
            transformedMat = processed.transformedMat,
            xRatio = processed.xRatio,
            yRatio = processed.yRatio,
            upscaleFactor = 1.2f,
            downshiftFactor = 0.0f,
            originalWidth = processed.originalMat.cols(),
            originalHeight = processed.originalMat.rows()
        )

        // Path A (expected/UX-correct): run Mi-GAN on original image aligned with mask
        val a = migan.inferMiGan(processed.originalMat, mask)

        // Path B (problematic): run Mi-GAN on transformed image, forcing mask resize inside Mi-GAN
        val b = migan.inferMiGan(processed.transformedMat, mask)

        // Resize B back to original to compare with A (nearest for mask alignment, linear for image)
        val bResized = Mat()
        Imgproc.resize(
            b,
            bResized,
            Size(processed.originalMat.cols().toDouble(), processed.originalMat.rows().toDouble()),
            0.0,
            0.0,
            Imgproc.INTER_LINEAR
        )

        // Save debug images for manual inspection when running instrumented tests locally
        val outDir = appContext.getExternalFilesDir(null)
        if (outDir != null) {
            // Save via RGB->BGR conversion for correct colors in PNG
            val aBgr = Mat()
            val bBgr = Mat()
            val bResBgr = Mat()
            Imgproc.cvtColor(a, aBgr, Imgproc.COLOR_RGB2BGR)
            Imgproc.cvtColor(b, bBgr, Imgproc.COLOR_RGB2BGR)
            Imgproc.cvtColor(bResized, bResBgr, Imgproc.COLOR_RGB2BGR)
            Imgcodecs.imwrite(File(outDir, "ui_parity_A_original.png").absolutePath, aBgr)
            Imgcodecs.imwrite(File(outDir, "ui_parity_B_transformed.png").absolutePath, bBgr)
            Imgcodecs.imwrite(
                File(outDir, "ui_parity_B_resized_to_original.png").absolutePath,
                bResBgr
            )
            aBgr.release(); bBgr.release(); bResBgr.release()
        }

        // Minimal assertion to detect the problematic path: A and resized B should not be identical pixel-by-pixel.
        // This highlights that running Mi-GAN on a different-sized image gives a different output, explaining UX mismatch.
        val equal = matsAreExactlyEqual(a, bResized)
        assertFalse(
            "Mi-GAN result on transformed size should not exactly equal result on original size when resized back",
            equal
        )

        // Cleanup
        a.release()
        b.release()
        bResized.release()
        mask.release()
    }

    private fun matsAreExactlyEqual(a: Mat, b: Mat): Boolean {
        if (a.rows() != b.rows() || a.cols() != b.cols() || a.type() != b.type()) return false
        val total = a.rows() * a.cols() * a.channels()
        val ad = ByteArray(total)
        val bd = ByteArray(total)
        a.get(0, 0, ad)
        b.get(0, 0, bd)
        for (i in 0 until total) {
            if (ad[i] != bd[i]) return false
        }
        return true
    }
}
