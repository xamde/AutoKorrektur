package de.konradvoelkel.android.autokorrektur

import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.YoloInferenceTFLite
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

        try {
            // Initialize YOLO inference
            val yoloInference = YoloInferenceTFLite(appContext)
            yoloInference.initialize("yolo11s")

            // Initialize ImageProcessor
            val imageProcessor = ImageProcessor(appContext)

            // Process example1.jpeg
            val mediaFile = "example1.jpeg"
            println("[DEBUG_LOG] Processing $mediaFile")

            // Copy media file from test assets
            val mediaInputStream = testContext.assets.open(mediaFile)
            val tempFile = File(appContext.cacheDir, mediaFile)
            val outputStream = FileOutputStream(tempFile)

            mediaInputStream.use { input ->
                outputStream.use { output ->
                    input.copyTo(output)
                }
            }

            val fileUri = Uri.fromFile(tempFile)
            val modelWidth = 640
            val modelHeight = 640

            // Process the image
            val processedImage = imageProcessor.processInputImage(
                imageUri = fileUri,
                modelWidth = modelWidth,
                modelHeight = modelHeight,
                downscaleMp = null
            )

            println("[DEBUG_LOG] Image processed: ${processedImage.originalMat.rows()}x${processedImage.originalMat.cols()}")
            println("[DEBUG_LOG] Transformed: ${processedImage.transformedMat.rows()}x${processedImage.transformedMat.cols()}")

            // Run YOLO inference with detailed logging
            val resultMask = yoloInference.inferYolo(
                transformedMat = processedImage.transformedMat,
                xRatio = processedImage.xRatio,
                yRatio = processedImage.yRatio,
                //modelWidth = modelWidth,
                //modelHeight = modelHeight,
                upscaleFactor = 1.2f,
                //scoreThreshold = 0.1f,
                downshiftFactor = 0.0f
            )

            println("[DEBUG_LOG] Result mask: ${resultMask.rows()}x${resultMask.cols()}")

            // Analyze the mask in detail
            val totalPixels = resultMask.rows() * resultMask.cols()
            var blackPixels = 0
            var whitePixels = 0
            var grayPixels = 0

            // Count different pixel values
            val blackMask = Mat()
            val whiteMask = Mat()

            Core.inRange(resultMask, Scalar(0.0), Scalar(10.0), blackMask)
            blackPixels = Core.countNonZero(blackMask)

            Core.inRange(resultMask, Scalar(245.0), Scalar(255.0), whiteMask)
            whitePixels = Core.countNonZero(whiteMask)

            grayPixels = totalPixels - blackPixels - whitePixels

            println("[DEBUG_LOG] Pixel analysis:")
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

            // Sample some pixel values
            val sampleData = ByteArray(100)
            resultMask.get(100, 100, sampleData)
            println(
                "[DEBUG_LOG] Sample pixel values at (100,100): ${
                    sampleData.take(10).joinToString(", ")
                }"
            )

            // Check car detection using the same logic as the test
            val blackPixelRatio = blackPixels.toDouble() / totalPixels.toDouble()
            val threshold = 0.0001 // 0.01%
            val carsDetected = blackPixelRatio > threshold

            println(
                "[DEBUG_LOG] Car detection result: $carsDetected (ratio: ${
                    String.format(
                        "%.6f",
                        blackPixelRatio
                    )
                }, threshold: $threshold)"
            )

            // Clean up
            blackMask.release()
            whiteMask.release()
            processedImage.originalMat.release()
            processedImage.transformedMat.release()
            resultMask.release()
            tempFile.delete()
            yoloInference.close()

            println("[DEBUG_LOG] Debug test completed")

        } catch (e: Exception) {
            println("[DEBUG_LOG] Error in debug test: ${e.message}")
            e.printStackTrace()
            fail("Debug test failed: ${e.message}")
        }
    }
}