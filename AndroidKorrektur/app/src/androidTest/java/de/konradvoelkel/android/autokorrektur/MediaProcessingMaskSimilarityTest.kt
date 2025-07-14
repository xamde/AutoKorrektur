package de.konradvoelkel.android.autokorrektur

import android.net.Uri
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor // Your ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.YoloInferenceTFLite // Your YoloInference
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

@RunWith(AndroidJUnit4::class)
class MediaProcessingMaskSimilarityTest { // Renamed for clarity

    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    private val testContext = InstrumentationRegistry.getInstrumentation().context

    private lateinit var yoloInference: YoloInferenceTFLite
    private lateinit var imageProcessor: ImageProcessor

    // Files to be cleaned up
    private val tempFiles = mutableListOf<File>()

    @Before
    fun setUp() {
        // Initialize OpenCV
        try {
            println("[DEBUG_LOG] Initializing OpenCV for mask similarity test")
            if (!OpenCVLoader.initDebug()) {
                fail("OpenCV initialization failed - required for mask similarity test")
            }
            println("[DEBUG_LOG] OpenCV initialized successfully")
        } catch (e: Exception) {
            fail("OpenCV initialization check failed: ${e.message}")
        }

        println("[DEBUG_LOG] Creating YoloInference and ImageProcessor")
        yoloInference = YoloInferenceTFLite(appContext)
        imageProcessor = ImageProcessor(appContext)

        println("[DEBUG_LOG] Initializing YOLO model")
        try {
            yoloInference.initialize("yolo11s") // Replace with your actual model name if different
            println("[DEBUG_LOG] YOLO model initialized successfully")
        } catch (e: Exception) {
            println("[DEBUG_LOG] YOLO initialization failed: ${e.message}")
            e.printStackTrace()
            fail("YOLO initialization failed: ${e.message}")
        }
    }

    @After
    fun tearDown() {
        println("[DEBUG_LOG] Cleaning up resources...")
        if (::yoloInference.isInitialized) {
            yoloInference.close()
        }
        tempFiles.forEach { it.delete() }
        tempFiles.clear()
        println("[DEBUG_LOG] Teardown complete.")
    }

    private fun copyAssetToCache(assetFileName: String): File {
        val assetManager = testContext.assets
        val inputStream: InputStream = assetManager.open(assetFileName)
        val tempFile = File(appContext.cacheDir, "test_${assetFileName}")
        FileOutputStream(tempFile).use { output -> inputStream.copyTo(output) }
        inputStream.close()
        tempFiles.add(tempFile) // Add to list for cleanup
        return tempFile
    }

    /**
     * Loads a ground truth mask image, converts it to grayscale, ensures it's binary
     * (0 for object/car, 255 for background), and resizes it.
     */
    private fun loadAndPrepareGroundTruthMask(
        assetFileName: String,
        targetWidth: Int,
        targetHeight: Int
    ): Mat {
        val gtFile = copyAssetToCache(assetFileName)
        var groundTruthMat = Imgcodecs.imread(gtFile.absolutePath, Imgcodecs.IMREAD_GRAYSCALE)

        assertNotNull("Failed to load ground truth mask: $assetFileName", groundTruthMat)
        assertTrue("Ground truth mask is empty: $assetFileName", !groundTruthMat.empty())

        // Ensure it's binary (0 for car, 255 for background)
        // If your example1mask.jpeg is already perfectly binary (only 0 and 255 values),
        // this thresholding might not be strictly necessary but is good for robustness.
        // It handles cases where JPEG compression might have introduced intermediate values.
        val binaryGroundTruth = Mat()
        Imgproc.threshold(groundTruthMat, binaryGroundTruth, 127.0, 255.0, Imgproc.THRESH_BINARY)
        // We expect cars (black) to be 0 and background (white) to be 255.
        // If your example1mask.jpeg has cars as white and background as black, use THRESH_BINARY_INV
        groundTruthMat.release() // Release the original loaded mat

        // Resize to match the YOLO output mask dimensions
        val resizedGtMask = Mat()
        Imgproc.resize(
            binaryGroundTruth,
            resizedGtMask,
            Size(targetWidth.toDouble(), targetHeight.toDouble()),
            0.0,
            0.0,
            Imgproc.INTER_NEAREST // Use INTER_NEAREST for masks to avoid interpolating new pixel values
        )
        binaryGroundTruth.release()

        println("[DEBUG_LOG] Loaded and prepared ground truth mask $assetFileName: ${resizedGtMask.rows()}x${resizedGtMask.cols()}, type: ${resizedGtMask.type()}, channels: ${resizedGtMask.channels()}")
        // Verify it's single channel after grayscale and thresholding
        assertEquals(
            "Ground truth mask should be single channel (CV_8UC1)",
            1,
            resizedGtMask.channels()
        )
        assertEquals("Ground truth mask type should be CV_8U", CvType.CV_8U, resizedGtMask.type())


        return resizedGtMask
    }


    @Test
    fun testYoloMaskSimilarityWithGroundTruth() {
        println("[DEBUG_LOG] Starting YOLO mask similarity test")

        val inputImageFile = "example1.jpeg" // Image containing car(s)
        val groundTruthMaskFile = "example1mask.jpeg" // Your pre-made mask

        val modelWidth = 640  // Adjust to your YOLO model's input width
        val modelHeight = 640 // Adjust to your YOLO model's input height
        val scoreThreshold = 0.25f // Adjust as needed
        val minAcceptablePixelAgreementRatio = 0.80 // 80% agreement

        var processedImage: ImageProcessor.ProcessedImage? = null
        var yoloResultMask: Mat? = null
        var groundTruthMask: Mat? = null

        try {
            // 1. Load and Process Input Image
            println("[DEBUG_LOG] Processing input image: $inputImageFile")
            val inputFile = copyAssetToCache(inputImageFile)
            val fileUri = Uri.fromFile(inputFile)

            processedImage = imageProcessor.processInputImage(
                uri = fileUri,
                modelWidth = modelWidth,
                modelHeight = modelHeight,
                downscaleMp = null
            )
            assertNotNull("Processed image should not be null", processedImage)
            assertNotNull("Transformed mat should not be null", processedImage!!.transformedMat)

            // 2. Run YOLO Inference to get the predicted mask
            println("[DEBUG_LOG] Running YOLO inference for $inputImageFile")
            yoloResultMask = yoloInference.inferYolo(
                transformedMat = processedImage.transformedMat,
                xRatio = processedImage.xRatio,
                yRatio = processedImage.yRatio,
                //modelWidth = modelWidth,
                //modelHeight = modelHeight,
                upscaleFactor = 1.2f,
                //scoreThreshold = scoreThreshold,
                downshiftFactor = 0.0f
            )
            assertNotNull("YOLO result mask should not be null", yoloResultMask)
            assertTrue("YOLO result mask should not be empty", !yoloResultMask!!.empty())

            // Ensure YOLO mask is CV_8UC1 and binary (0 for car, 255 for background)
            // Your inferYolo should ideally already return it in this format.
            // If not, add processing here.
            assertEquals(
                "YOLO mask should be single channel (CV_8UC1)",
                1,
                yoloResultMask.channels()
            )
            assertEquals("YOLO mask type should be CV_8U", CvType.CV_8U, yoloResultMask.type())
            // You might want to explicitly threshold it if inferYolo doesn't guarantee binary output:
            // Imgproc.threshold(yoloResultMask, yoloResultMask, 127.0, 255.0, Imgproc.THRESH_BINARY);


            println("[DEBUG_LOG] YOLO mask generated: ${yoloResultMask.rows()}x${yoloResultMask.cols()}, type: ${yoloResultMask.type()}")


            // 3. Load and Prepare Ground Truth Mask
            println("[DEBUG_LOG] Loading ground truth mask: $groundTruthMaskFile")
            groundTruthMask = loadAndPrepareGroundTruthMask(
                groundTruthMaskFile,
                yoloResultMask.cols(), // Ensure GT mask is resized to match YOLO output
                yoloResultMask.rows()
            )

            // 4. Compare Masks: Pixel-wise Agreement
            val totalPixels = yoloResultMask.rows() * yoloResultMask.cols().toDouble()
            if (totalPixels == 0.0) {
                fail("YOLO result mask has zero pixels.")
            }

            val agreementMask = Mat()
            // Compares pixels element by element. agreementMask will have 255 where pixels are equal, 0 otherwise.
            Core.compare(yoloResultMask, groundTruthMask, agreementMask, Core.CMP_EQ)

            val matchingPixels = Core.countNonZero(agreementMask).toDouble()
            val pixelAgreementRatio = matchingPixels / totalPixels

            println("[DEBUG_LOG] Matching Pixels: $matchingPixels / Total Pixels: $totalPixels")
            println(
                "[DEBUG_LOG] Pixel Agreement Ratio for $inputImageFile: ${
                    String.format(
                        "%.4f",
                        pixelAgreementRatio
                    )
                }"
            )

            assertTrue(
                "Pixel agreement ratio ($pixelAgreementRatio) for $inputImageFile is below threshold ($minAcceptablePixelAgreementRatio). " +
                        "YOLO mask is not similar enough to the ground truth.",
                pixelAgreementRatio >= minAcceptablePixelAgreementRatio
            )

            // Optional: Check if any car was detected by YOLO if ground truth expects it
            val gtCarPixels = countObjectPixels(groundTruthMask, 0.0) // Count black pixels in GT
            if (gtCarPixels > 0) {
                val yoloCarPixels =
                    countObjectPixels(yoloResultMask, 0.0) // Count black pixels in YOLO
                assertTrue(
                    "Ground truth has car pixels, but YOLO mask has none.",
                    yoloCarPixels > 0
                )
                println("[DEBUG_LOG] GT car pixels: $gtCarPixels, YOLO car pixels: $yoloCarPixels")
            }


            println("[DEBUG_LOG] YOLO mask similarity test for $inputImageFile PASSED")

        } catch (e: Exception) {
            println("[DEBUG_LOG] Error during YOLO mask similarity test: ${e.message}")
            e.printStackTrace()
            // Save masks for debugging if test fails
            if (yoloResultMask != null && !yoloResultMask.empty()) {
                val yoloPath =
                    File(appContext.cacheDir, "debug_yolo_mask_${inputImageFile}.png").absolutePath
                Imgcodecs.imwrite(yoloPath, yoloResultMask)
                println("[DEBUG_LOG] Saved failing YOLO mask to $yoloPath")
            }
            if (groundTruthMask != null && !groundTruthMask.empty()) {
                val gtPath =
                    File(appContext.cacheDir, "debug_gt_mask_${inputImageFile}.png").absolutePath
                Imgcodecs.imwrite(gtPath, groundTruthMask)
                println("[DEBUG_LOG] Saved failing GT mask to $gtPath")
            }
            fail("YOLO mask similarity test failed: ${e.message}")
        } finally {
            // Release OpenCV Mats
            processedImage?.originalMat?.release()
            processedImage?.transformedMat?.release()
            yoloResultMask?.release()
            groundTruthMask?.release()
            // Temp files are cleaned up in @After
        }
    }

    /**
     * Helper to count object pixels (assuming object is a specific value, e.g., 0 for black).
     */
    private fun countObjectPixels(mask: Mat, objectPixelValue: Double): Int {
        val objectOnlyMask = Mat()
        Core.compare(mask, Scalar(objectPixelValue), objectOnlyMask, Core.CMP_EQ)
        val count = Core.countNonZero(objectOnlyMask)
        objectOnlyMask.release()
        return count
    }
}