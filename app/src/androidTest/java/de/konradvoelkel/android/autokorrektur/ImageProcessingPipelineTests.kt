package de.konradvoelkel.android.autokorrektur

import android.content.pm.ApplicationInfo
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import org.junit.After
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

@org.junit.Ignore("Split into pipeline/* tests")
@RunWith(AndroidJUnit4::class)
class ImageProcessingPipelineTests :
    de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest() {

    // --- Helpers to avoid code duplication in tests ---


    companion object {
        private lateinit var sharedYolo: YoloService
        private lateinit var sharedImageProcessor: ImageProcessor
        private lateinit var sharedMiGan: de.konradvoelkel.android.autokorrektur.ml.MiGanInference

        @BeforeClass
        @JvmStatic
        fun beforeAll() {
            de.konradvoelkel.android.autokorrektur.shared.AndroidTestUtils.initOpenCV()
            val ctx = InstrumentationRegistry.getInstrumentation().targetContext
            sharedYolo = YoloServiceImpl(ctx)
            sharedYolo.initialize("yolo11s")
            sharedImageProcessor = ImageProcessor(ctx)
            sharedMiGan = de.konradvoelkel.android.autokorrektur.ml.MiGanInference(ctx)
            sharedMiGan.initialize()
        }

        @AfterClass
        @JvmStatic
        fun afterAll() {
            if (this::sharedYolo.isInitialized) {
                sharedYolo.close()
            }
            if (this::sharedMiGan.isInitialized) {
                sharedMiGan.close()
            }
        }
    }

    private lateinit var yoloInference: YoloService
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var miGanInference: de.konradvoelkel.android.autokorrektur.ml.MiGanInference
    private val tempFiles = mutableListOf<File>()

    private fun isDebug(): Boolean {
        val debuggable = (appContext.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0
        return debuggable && de.konradvoelkel.android.autokorrektur.shared.OpenCvTestUtils.shouldWriteDebugArtifacts(
            appContext
        )
    }

    @Before
    fun setUp() {
        // Reuse shared instances to avoid per-test initialization cost
        yoloInference = sharedYolo
        imageProcessor = sharedImageProcessor
        miGanInference = sharedMiGan
    }

    @After
    fun tearDown() {
        // Do not close shared instances here; only clean up temp files
        tempFiles.forEach { it.delete() }
    }


    private fun hasCarDetection(mask: Mat): Boolean {
        val totalPixels = mask.rows() * mask.cols()
        if (totalPixels == 0) return false
        val blackMask = Mat()
        Core.inRange(mask, Scalar(0.0), Scalar(10.0), blackMask)
        val blackPixels = Core.countNonZero(blackMask)
        blackMask.release()
        val blackPixelRatio = blackPixels.toDouble() / totalPixels.toDouble()
        println(
            "[DEBUG_LOG] hasCarDetection: Black pixels: $blackPixels / $totalPixels (${
                String.format(
                    "%.4f",
                    blackPixelRatio * 100
                )
            }%)"
        )
        return blackPixelRatio > 0.01
    }

    @Test
    fun testImageSelectionSimulation() {
        val mockupImageUri =
            de.konradvoelkel.android.autokorrektur.shared.AndroidTestUtils.copyAssetToCache(
                appContext,
                "image_1_with_car_640x640.png"
            )
                .let { Uri.fromFile(it) }
        assertNotNull("Mockup image URI should not be null", mockupImageUri)

        val processedImage = imageProcessor.processInputImage(
            imageUri = mockupImageUri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = 2.0f
        )

        assertNotNull("Processed image should not be null", processedImage)
        assertEquals(640, processedImage.transformedBitmap.width)
        assertEquals(640, processedImage.transformedBitmap.height)
        assertTrue("X ratio should be positive", processedImage.xRatio > 0)
        assertTrue("Y ratio should be positive", processedImage.yRatio > 0)
    }

    @Test
    fun testCarDetectionOnExampleImage() {
        val tempFile = cacheAsset("image_1_with_car_640x640.png", tempFiles)
        val fileUri = Uri.fromFile(tempFile)

        val processedImage = imageProcessor.processInputImage(
            imageUri = fileUri,
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

        // for debugging, how does the no-car-but-still-some-mask look like?
        if (isDebug()) {
            val outputFileName = "debug_mask_car.png"
            val outputFile = File(appContext.getExternalFilesDir(null), outputFileName)
            Imgcodecs.imwrite(outputFile.absolutePath, resultMask)
            println("[DEBUG_LOG] Saved debug mask to: ${outputFile.absolutePath}")
        }

        assertTrue(
            "Car should be detected in image_1_with_car_640x640.png",
            hasCarDetection(resultMask)
        )
        resultMask.release()
    }

    @Test
    fun testNoCarDetectionOnResultImage() {
        val tempFile = cacheAsset("image_1_without_car_640x640.png", tempFiles)
        val fileUri = Uri.fromFile(tempFile)

        val processedImage = imageProcessor.processInputImage(
            imageUri = fileUri,
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

        // for debugging, how does the no-car-but-still-some-mask look like?
        if (isDebug()) {
            val outputFileName = "debug_mask_no_car.png"
            val outputFile = File(appContext.getExternalFilesDir(null), outputFileName)
            Imgcodecs.imwrite(outputFile.absolutePath, resultMask)
            println("[DEBUG_LOG] Saved debug mask to: ${outputFile.absolutePath}")
        }

        assertFalse(
            "Car should NOT be detected in image_1_without_car_640x640.png",
            hasCarDetection(resultMask)
        )
        resultMask.release()
    }

    @Test
    fun testYoloMaskCreationProperties() {
        val tempFile = cacheAsset("image_1_with_car_640x640.png", tempFiles)
        val fileUri = Uri.fromFile(tempFile)

        val processedImage = imageProcessor.processInputImage(
            imageUri = fileUri,
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

        assertNotNull("Result mask should not be null", resultMask)
        assertTrue("Result mask should not be empty", !resultMask.empty())
        assertEquals(processedImage.originalMat.rows(), resultMask.rows())
        assertEquals(processedImage.originalMat.cols(), resultMask.cols())

        resultMask.release()
    }

    @Test
    fun testCarDetectionMaskIsSensible() {
        val tempFile = cacheAsset("image_1_with_car_640x640.png", tempFiles)
        val fileUri = Uri.fromFile(tempFile)

        val processedImage = imageProcessor.processInputImage(
            imageUri = fileUri,
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

        val totalPixelsInMask = resultMask.rows() * resultMask.cols()
        val blackMask = Mat()
        Core.inRange(resultMask, Scalar(0.0), Scalar(10.0), blackMask)
        val detectedPixels = Core.countNonZero(blackMask)
        blackMask.release()

        assertTrue("A car should be detected", detectedPixels > 0)

        val detectedRatio = detectedPixels.toDouble() / totalPixelsInMask.toDouble()
        assertTrue("Detected mask should be smaller than the whole image", detectedRatio < 0.95)
        assertTrue("Detected mask should be of a sensible size", detectedRatio > 0.001)

        resultMask.release()
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

        assertEquals(
            "Mask height must match original image height",
            processedImage.originalMat.rows(), resultMask.rows()
        )
        assertEquals(
            "Mask width must match original image width",
            processedImage.originalMat.cols(), resultMask.cols()
        )
        assertEquals(
            "Reference mask height must match result height",
            resultMask.rows(), referenceMask.rows()
        )
        assertEquals(
            "Reference mask width must match result width",
            resultMask.cols(), referenceMask.cols()
        )

        // Binarize: treat near-black as car (255), others 0
        val binResult = Mat()
        val binReference = Mat()
        Core.inRange(resultMask, Scalar(0.0), Scalar(10.0), binResult)
        Core.inRange(referenceMask, Scalar(0.0), Scalar(10.0), binReference)

        // Compute XOR to find mismatches
        val diff = Mat()
        Core.bitwise_xor(binResult, binReference, diff)
        val mismatches = Core.countNonZero(diff)
        val totalPixels = diff.rows() * diff.cols()
        val agreement = 1.0 - (mismatches.toDouble() / totalPixels.toDouble())

        println(
            "[DEBUG_LOG] Mask agreement: ${
                String.format(
                    "%.4f",
                    agreement * 100
                )
            }% (mismatches=$mismatches / total=$totalPixels)"
        )

        // Optionally save debug images
        if (isDebug()) {
            val outDebugDir = appContext.getExternalFilesDir(null)
            if (outDebugDir != null) {
                val resultPath = File(outDebugDir, "debug_mask_result.png").absolutePath
                val referencePath = File(outDebugDir, "debug_mask_reference.png").absolutePath
                val diffPath = File(outDebugDir, "debug_mask_diff.png").absolutePath
                Imgcodecs.imwrite(resultPath, resultMask)
                Imgcodecs.imwrite(referencePath, referenceMask)
                Imgcodecs.imwrite(diffPath, diff)
                println("[DEBUG_LOG] Saved debug images to: ${outDebugDir.absolutePath}")
                println("[DEBUG_LOG] - result: $resultPath")
                println("[DEBUG_LOG] - reference: $referencePath")
                println("[DEBUG_LOG] - diff: $diffPath")
            }
        }

        assertTrue("Generated mask should agree with reference mask >= 95%", agreement >= 0.95)

        // Cleanup
        diff.release()
        binResult.release()
        binReference.release()
        referenceMask.release()
        resultMask.release()
    }

    @Test
    fun testMiGanRemovesCarAndKeepsBackground() {
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
        val inpainted = miGanInference.inferMiGan(processed.originalMat, referenceMask)

        // Optionally save debug outputs (convert RGB->BGR before OpenCV imwrite to avoid swapped colors)
        if (isDebug()) {
            val outDir = appContext.getExternalFilesDir(null)
            if (outDir != null) {
                val inpaintedPath = File(outDir, "debug_migan_inpainted.png").absolutePath
                val bgrDebug = Mat()
                Imgproc.cvtColor(inpainted, bgrDebug, Imgproc.COLOR_RGB2BGR)
                Imgcodecs.imwrite(inpaintedPath, bgrDebug)
                bgrDebug.release()
                println("[DEBUG_LOG] Saved Mi-GAN output to: $inpaintedPath (saved with RGB->BGR conversion)")
            }
        }

        // 1) Verify no car remains in the inpainted image via YOLO mask
        // Save inpainted image temporarily to pass through ImageProcessor
        val tempOutFile = File(appContext.cacheDir, "migan_inpainted_tmp.png")
        // Save with RGB->BGR conversion to ensure correct colors when read back by Android Bitmap pipeline
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

        val yoloMaskOnOutput = yoloInference.infer(
            transformedMat = processedOut.transformedMat,
            xRatio = processedOut.xRatio,
            yRatio = processedOut.yRatio,
            upscaleFactor = 1.2f,
            originalWidth = processedOut.originalMat.cols(),
            originalHeight = processedOut.originalMat.rows()
        )

        if (isDebug()) {
            val outputFileName = "debug_mask_after_migan.png"
            val outputFile = File(appContext.getExternalFilesDir(null), outputFileName)
            Imgcodecs.imwrite(outputFile.absolutePath, yoloMaskOnOutput)
            println("[DEBUG_LOG] Saved YOLO mask (after Mi-GAN) to: ${outputFile.absolutePath}")
        }

        assertFalse(
            "After Mi-GAN, there should be no car detected",
            hasCarDetection(yoloMaskOnOutput)
        )

        // 2) Verify in non-car (white) regions, output agrees with input roughly
        // Ensure sizes match
        assertEquals(processed.originalMat.rows(), inpainted.rows())
        assertEquals(processed.originalMat.cols(), inpainted.cols())
        assertEquals(referenceMask.rows(), inpainted.rows())
        assertEquals(referenceMask.cols(), inpainted.cols())

        // Compute mean absolute difference over white mask (>= 245)
        val inputBGR = Mat()
        val outputBGR = Mat()
        // Ensure type is 8UC3
        processed.originalMat.convertTo(inputBGR, org.opencv.core.CvType.CV_8UC3)
        inpainted.convertTo(outputBGR, org.opencv.core.CvType.CV_8UC3)

        val inputData = ByteArray(inputBGR.rows() * inputBGR.cols() * inputBGR.channels())
        val outputData = ByteArray(outputBGR.rows() * outputBGR.cols() * outputBGR.channels())
        val maskData = ByteArray(referenceMask.rows() * referenceMask.cols())
        inputBGR.get(0, 0, inputData)
        outputBGR.get(0, 0, outputData)
        referenceMask.get(0, 0, maskData)

        var sumAbsDiff: Long = 0
        var count: Long = 0
        //val width = inputBGR.cols()
        val channels = inputBGR.channels()
        for (i in maskData.indices) {
            val m = maskData[i].toInt() and 0xFF
            if (m >= 245) { // white region (non-car)
                val base = i * channels
                // B, G, R channels
                val d0 =
                    kotlin.math.abs((inputData[base].toInt() and 0xFF) - (outputData[base].toInt() and 0xFF))
                val d1 =
                    kotlin.math.abs((inputData[base + 1].toInt() and 0xFF) - (outputData[base + 1].toInt() and 0xFF))
                val d2 =
                    kotlin.math.abs((inputData[base + 2].toInt() and 0xFF) - (outputData[base + 2].toInt() and 0xFF))
                sumAbsDiff += (d0 + d1 + d2).toLong()
                count += 3
            }
        }

        // To avoid divide-by-zero if mask is wrong
        assertTrue("Reference mask must contain white pixels", count > 0)
        val meanAbsDiff = sumAbsDiff.toDouble() / count.toDouble()

        println(
            "[DEBUG_LOG] Mean absolute difference on white mask pixels: ${
                String.format(
                    "%.2f",
                    meanAbsDiff
                )
            }"
        )

        // Allow small differences due to model/inference; threshold in 0..255 scale per channel
        val tolerancePerChannel = 10.0
        assertTrue(
            "Background should be preserved (mean abs diff per channel <= $tolerancePerChannel)",
            meanAbsDiff <= tolerancePerChannel
        )

        // Cleanup mats
        inputBGR.release()
        outputBGR.release()
        yoloMaskOnOutput.release()
        inpainted.release()
        referenceMask.release()
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

        // 3) Inpaint with Mi-GAN using that mask
        val inpainted = miGanInference.inferMiGan(processedIn.originalMat, mask)

        // 4) Load reference result and convert BGR->RGB for fair comparison
        val refFile = cacheAsset("example2Result.png", tempFiles)
        val refBgr = Imgcodecs.imread(refFile.absolutePath, Imgcodecs.IMREAD_COLOR)
        assertTrue("Reference image should load", !refBgr.empty())
        val refRgb =
            de.konradvoelkel.android.autokorrektur.shared.OpenCvTestUtils.matLoadedFromFileBgrToRgb(
                refBgr
            )
        refBgr.release()

        // 5) Ensure sizes match
        assertEquals(processedIn.originalMat.rows(), inpainted.rows())
        assertEquals(processedIn.originalMat.cols(), inpainted.cols())
        assertEquals(processedIn.originalMat.rows(), refRgb.rows())
        assertEquals(processedIn.originalMat.cols(), refRgb.cols())

        // 6) Compute similarity over white regions of mask (non-car)
        val inRgb8 = Mat()
        val refRgb8 = Mat()
        inpainted.convertTo(inRgb8, org.opencv.core.CvType.CV_8UC3)
        refRgb.convertTo(refRgb8, org.opencv.core.CvType.CV_8UC3)
        val meanAbs =
            de.konradvoelkel.android.autokorrektur.shared.OpenCvTestUtils.meanAbsDiffOnMaskRgb8u3(
                mask,
                inRgb8,
                refRgb8
            )

        println(
            "[DEBUG_LOG] example2 mean abs diff over white mask: ${
                String.format(
                    "%.2f",
                    meanAbs
                )
            }"
        )

        val tolerancePerChannel = 12.0
        assertTrue(
            "Inpainted output should match reference outside masked regions (<= $tolerancePerChannel per-channel)",
            meanAbs <= tolerancePerChannel
        )

        // 7) Optionally check that no car remains after inpainting
        val tempOutFile = File(appContext.cacheDir, "example2_migan_out.png")
        de.konradvoelkel.android.autokorrektur.shared.OpenCvTestUtils.saveDebugRgbMatAsPngBgr(
            inpainted,
            tempOutFile
        ) // write via RGB->BGR
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
            "After Mi-GAN, there should be no car detected (example2)",
            hasCarDetection(maskAfter)
        )

        if (isDebug()) {
            val outDir = appContext.getExternalFilesDir(null)
            if (outDir != null) {
                de.konradvoelkel.android.autokorrektur.shared.OpenCvTestUtils.saveDebugRgbMatAsPngBgr(
                    inpainted,
                    File(outDir, "debug_example2_migan_inpainted.png")
                )
                Imgcodecs.imwrite(File(outDir, "debug_example2_mask.png").absolutePath, mask)
                Imgcodecs.imwrite(
                    File(outDir, "debug_example2_mask_after.png").absolutePath,
                    maskAfter
                )
            }
        }

        // Cleanup
        inRgb8.release()
        refRgb8.release()
        refRgb.release()
        maskAfter.release()
        mask.release()
        inpainted.release()
    }
}
