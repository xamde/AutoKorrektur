package de.konradvoelkel.android.autokorrektur

import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.YoloInferenceTFLite
import org.junit.Assert.assertFalse // For asserting false
import org.junit.Assert.assertTrue // For asserting true
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Scalar
import java.io.File
import java.io.FileOutputStream

@RunWith(AndroidJUnit4::class)
class CarDetectionDebugTest {

    @Test
    fun debugCarDetectionForExample1() {
        println("[DEBUG_LOG] Starting car detection debug for example1.jpeg")

        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val testContext = InstrumentationRegistry.getInstrumentation().context

        // Initialize OpenCV
        if (!org.opencv.android.OpenCVLoader.initDebug()) {
            fail("OpenCV initialization failed")
        }

        var yoloInference: YoloInferenceTFLite? = null
        var tempFile: File? = null
        var processedImage: ImageProcessor.ProcessedImage? = null
        var resultMask: Mat? = null
        val blackMask = Mat()
        val whiteMask = Mat()

        try {
            // Initialize YOLO inference
            yoloInference = YoloInferenceTFLite(appContext)
            yoloInference.initialize("yolo11s") // Ensure this matches your model name

            // Initialize ImageProcessor
            val imageProcessor = ImageProcessor(appContext)

            // Process example1.jpeg
            val mediaFile = "example1.jpeg"
            println("[DEBUG_LOG] Processing $mediaFile")

            // Copy media file from test assets
            val mediaInputStream = testContext.assets.open(mediaFile)
            tempFile = File(appContext.cacheDir, mediaFile)
            val outputStream = FileOutputStream(tempFile)

            mediaInputStream.use { input ->
                outputStream.use { output ->
                    input.copyTo(output)
                }
            }
            mediaInputStream.close()
            outputStream.close()


            val fileUri = Uri.fromFile(tempFile)
            val modelWidth = 640 // Assuming YOLOv11s default, adjust if different
            val modelHeight = 640 // Assuming YOLOv11s default, adjust if different

            // Process the image
            processedImage = imageProcessor.processInputImage(
                imageUri = fileUri,
                modelWidth = modelWidth,
                modelHeight = modelHeight,
                downscaleMp = null // Or your desired downscale setting
            )

            println("[DEBUG_LOG] Image processed: ${processedImage.originalMat.rows()}x${processedImage.originalMat.cols()}")
            println("[DEBUG_LOG] Transformed: ${processedImage.transformedMat.rows()}x${processedImage.transformedMat.cols()}")

            // Run YOLO inference
            resultMask = yoloInference.inferYolo(
                transformedMat = processedImage.transformedMat,
                xRatio = processedImage.xRatio,
                yRatio = processedImage.yRatio,
                upscaleFactor = 1.2f,    // Use relevant default or test values
                downshiftFactor = 0.0f   // Use relevant default or test values
            )

            println("[DEBUG_LOG] Result mask: ${resultMask.rows()}x${resultMask.cols()}")
            assertTrue("Result mask should not be empty", !resultMask.empty())


            // Analyze the mask in detail
            val totalPixels = resultMask.rows() * resultMask.cols()
            var blackPixels = 0
            // ... (pixel counting logic as in your original test) ...
            Core.inRange(
                resultMask,
                Scalar(0.0),
                Scalar(10.0),
                blackMask
            ) // Assuming black pixels (0-10) are detections
            blackPixels = Core.countNonZero(blackMask)

            Core.inRange(resultMask, Scalar(245.0), Scalar(255.0), whiteMask)
            val whitePixels = Core.countNonZero(whiteMask)

            val grayPixels = totalPixels - blackPixels - whitePixels


            println("[DEBUG_LOG] Pixel analysis for $mediaFile:")
            println("[DEBUG_LOG] - Total pixels: $totalPixels")
            println(
                "[DEBUG_LOG] - Black pixels (0-10): $blackPixels (${
                    String.format(
                        "%.4f",
                        blackPixels.toDouble() / totalPixels * 100
                    )
                }%)"
            )
            println(
                "[DEBUG_LOG] - White pixels (245-255): $whitePixels (${
                    String.format(
                        "%.4f",
                        whitePixels.toDouble() / totalPixels * 100
                    )
                }%)"
            )
            println(
                "[DEBUG_LOG] - Gray pixels (11-244): $grayPixels (${
                    String.format(
                        "%.4f",
                        grayPixels.toDouble() / totalPixels * 100
                    )
                }%)"
            )


            // Check car detection using the same logic as the test
            val blackPixelRatio =
                if (totalPixels > 0) blackPixels.toDouble() / totalPixels.toDouble() else 0.0
            val detectionThreshold = 0.0001 // Your defined threshold for detecting a car (0.01%)
            val carsDetected = blackPixelRatio > detectionThreshold

            println(
                "[DEBUG_LOG] Car detection result for $mediaFile: $carsDetected (ratio: ${
                    String.format(
                        "%.6f",
                        blackPixelRatio
                    )
                }, threshold: $detectionThreshold)"
            )

            // Assert that a car IS detected for example1.jpeg
            assertTrue("Car should be detected in $mediaFile", carsDetected)


            println("[DEBUG_LOG] Debug test for $mediaFile completed")

        } catch (e: Exception) {
            println("[DEBUG_LOG] Error in debug test for example1.jpeg: ${e.message}")
            e.printStackTrace()
            fail("Debug test for example1.jpeg failed: ${e.message}")
        } finally {
            // Clean up
            blackMask.release()
            whiteMask.release()
            processedImage?.originalMat?.release()
            processedImage?.transformedMat?.release()
            resultMask?.release()
            tempFile?.delete()
            yoloInference?.close()
        }
    }

    @Test
    fun debugNoCarDetectionForExample1Result() {
        println("[DEBUG_LOG] Starting NO car detection debug for example1Result.jpeg")

        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val testContext = InstrumentationRegistry.getInstrumentation().context

        // Initialize OpenCV
        if (!org.opencv.android.OpenCVLoader.initDebug()) {
            fail("OpenCV initialization failed")
        }

        var yoloInference: YoloInferenceTFLite? = null
        var tempFile: File? = null
        var processedImage: ImageProcessor.ProcessedImage? = null
        var resultMask: Mat? = null
        val blackMask = Mat() // For counting detected pixels
        val whiteMask =
            Mat() // For counting background pixels (optional, but good for full analysis)

        // Process example1Result.jpeg
        val mediaFile = "example1Result.jpeg"
        println("[DEBUG_LOG] Processing $mediaFile")

        try {
            // Initialize YOLO inference
            yoloInference = YoloInferenceTFLite(appContext)
            // Ensure you use the same model name as your other test or the one you are debugging
            yoloInference.initialize("yolo11s")

            // Initialize ImageProcessor
            val imageProcessor = ImageProcessor(appContext)

            // Copy media file from test assets
            val mediaInputStream = testContext.assets.open(mediaFile)
            tempFile = File(appContext.cacheDir, mediaFile)
            val outputStream = FileOutputStream(tempFile)

            mediaInputStream.use { input ->
                outputStream.use { output ->
                    input.copyTo(output)
                }
            }
            mediaInputStream.close()
            outputStream.close()

            val fileUri = Uri.fromFile(tempFile)
            // Use the same model dimensions as your YOLO model expects
            val modelWidth = 640
            val modelHeight = 640

            // Process the image
            processedImage = imageProcessor.processInputImage(
                imageUri = fileUri,
                modelWidth = modelWidth,
                modelHeight = modelHeight,
                downscaleMp = null // Or your desired downscale setting for this test
            )

            println("[DEBUG_LOG] Image processed: ${processedImage.originalMat.rows()}x${processedImage.originalMat.cols()}")
            println("[DEBUG_LOG] Transformed: ${processedImage.transformedMat.rows()}x${processedImage.transformedMat.cols()}")

            // Run YOLO inference
            // Use parameters that you expect to yield no detection, or typical parameters
            // if the image itself should inherently result in no car detection.
            resultMask = yoloInference.inferYolo(
                transformedMat = processedImage.transformedMat,
                xRatio = processedImage.xRatio,
                yRatio = processedImage.yRatio,
                upscaleFactor = 1.2f,    // Consistent with the other test
                downshiftFactor = 0.0f   // Consistent with the other test
            )

            println("[DEBUG_LOG] Result mask: ${resultMask.rows()}x${resultMask.cols()}")
            assertTrue("Result mask should not be empty", !resultMask.empty())


            // Analyze the mask in detail
            val totalPixels = resultMask.rows() * resultMask.cols()
            var blackPixels = 0
            // Assuming your mask uses black pixels (0-10 range) to indicate detected objects (cars)
            // and white (245-255 range) for background.
            Core.inRange(resultMask, Scalar(0.0), Scalar(10.0), blackMask)
            blackPixels = Core.countNonZero(blackMask)

            Core.inRange(resultMask, Scalar(245.0), Scalar(255.0), whiteMask)
            val whitePixels = Core.countNonZero(whiteMask)

            val grayPixels = totalPixels - blackPixels - whitePixels


            println("[DEBUG_LOG] Pixel analysis for $mediaFile:")
            println("[DEBUG_LOG] - Total pixels: $totalPixels")
            println(
                "[DEBUG_LOG] - Black pixels (0-10): $blackPixels (${
                    String.format(
                        "%.4f",
                        blackPixels.toDouble() / totalPixels * 100
                    )
                }%)"
            )
            println(
                "[DEBUG_LOG] - White pixels (245-255): $whitePixels (${
                    String.format(
                        "%.4f",
                        whitePixels.toDouble() / totalPixels * 100
                    )
                }%)"
            )
            println(
                "[DEBUG_LOG] - Gray pixels (11-244): $grayPixels (${
                    String.format(
                        "%.4f",
                        grayPixels.toDouble() / totalPixels * 100
                    )
                }%)"
            )

            // Check car detection logic
            // The black pixel ratio should be below the threshold if no car is detected
            val blackPixelRatio =
                if (totalPixels > 0) blackPixels.toDouble() / totalPixels.toDouble() else 0.0
            val detectionThreshold = 0.0001 // Using the same threshold as your other test (0.01%)
            // This means if more than 0.01% of pixels are black, it's a "detection"
            val carsDetected = blackPixelRatio > detectionThreshold

            println(
                "[DEBUG_LOG] Car detection result for $mediaFile: $carsDetected (ratio: ${
                    String.format(
                        "%.6f",
                        blackPixelRatio
                    )
                }, threshold: $detectionThreshold)"
            )

            // Assert that NO car is detected for example1Result.jpeg
            assertFalse(
                "Car should NOT be detected in $mediaFile. Black pixel ratio was $blackPixelRatio",
                carsDetected
            )

            println("[DEBUG_LOG] Debug NO car detection test for $mediaFile completed")

        } catch (e: Exception) {
            println("[DEBUG_LOG] Error in NO car detection debug test for $mediaFile: ${e.message}")
            e.printStackTrace()
            fail("Debug NO car detection test for $mediaFile failed: ${e.message}")
        } finally {
            // Clean up resources
            blackMask.release()
            whiteMask.release()
            processedImage?.originalMat?.release()
            processedImage?.transformedMat?.release()
            resultMask?.release()
            tempFile?.delete()
            yoloInference?.close()
        }
    }
}
