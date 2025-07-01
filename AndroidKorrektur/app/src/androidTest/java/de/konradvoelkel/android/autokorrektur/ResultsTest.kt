package de.konradvoelkel.android.autokorrektur

import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.YoloInferenceTFLite
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Scalar
import java.io.File
import java.io.FileOutputStream

@Suppress("DEPRECATION")
@RunWith(AndroidJUnit4::class)
class ResultsTest {

    @Test
    fun testExample1Results() {
        println("[DEBUG_LOG] Starting results test for example1.jpeg")

        val appContext = InstrumentationRegistry.getInstrumentation().targetContext

        // Initialize OpenCV first
        if (!org.opencv.android.OpenCVLoader.initDebug()) {
            println("[DEBUG_LOG] OpenCV initialization failed")
            return
        }
        println("[DEBUG_LOG] OpenCV initialized successfully")

        try {
            // Initialize components
            val yoloInference = YoloInferenceTFLite(appContext)
            yoloInference.initialize("yolo11s")
            println("[DEBUG_LOG] YOLO model initialized successfully")

            val imageProcessor = ImageProcessor(appContext)
            println("[DEBUG_LOG] ImageProcessor created successfully")

            // Process example1.jpeg
            val mediaFile = "example1.jpeg"
            val mediaInputStream = appContext.assets.open(mediaFile)
            val tempFile = File(appContext.cacheDir, mediaFile)
            val outputStream = FileOutputStream(tempFile)

            mediaInputStream.use { input ->
                outputStream.use { output ->
                    input.copyTo(output)
                }
            }

            val fileUri = Uri.fromFile(tempFile)

            // Process the image
            val processedImage = imageProcessor.processInputImage(
                uri = fileUri,
                modelWidth = 640,
                modelHeight = 640,
                downscaleMp = null
            )

            println("[DEBUG_LOG] Image processed successfully")
            println("[DEBUG_LOG] Original: ${processedImage.originalMat.rows()}x${processedImage.originalMat.cols()}")
            println("[DEBUG_LOG] Transformed: ${processedImage.transformedMat.rows()}x${processedImage.transformedMat.cols()}, type: ${processedImage.transformedMat.type()}, channels: ${processedImage.transformedMat.channels()}")
            println("[DEBUG_LOG] Ratios - X: ${processedImage.xRatio}, Y: ${processedImage.yRatio}")

            // Check some pixel values
            val samplePixels = FloatArray(9) // 3 pixels * 3 channels
            processedImage.transformedMat.get(100, 100, samplePixels)
            println("[DEBUG_LOG] Sample pixel values: [${samplePixels.take(6).joinToString(", ")}]")

            // Run YOLO inference with very low threshold
            val resultMask = yoloInference.inferYolo(
                transformedMat = processedImage.transformedMat,
                xRatio = processedImage.xRatio,
                yRatio = processedImage.yRatio,
                modelWidth = 640,
                modelHeight = 640,
                upscaleFactor = 1.2f,
                scoreThreshold = 0.05f,
                downshiftFactor = 0.0f
            )
            println("[DEBUG_LOG] YOLO inference completed")

            // Analyze the result mask
            val totalPixels = resultMask.rows() * resultMask.cols()
            val blackMask = Mat()
            Core.inRange(resultMask, Scalar(0.0), Scalar(10.0), blackMask)
            val blackPixels = Core.countNonZero(blackMask)
            blackMask.release()

            val blackPixelRatio = blackPixels.toDouble() / totalPixels.toDouble()
            println("[DEBUG_LOG] RESULTS:")
            println("[DEBUG_LOG] - Total pixels: $totalPixels")
            println("[DEBUG_LOG] - Black pixels (cars): $blackPixels")
            println(
                "[DEBUG_LOG] - Black pixel ratio: ${
                    String.format(
                        "%.6f",
                        blackPixelRatio
                    )
                } (${String.format("%.4f", blackPixelRatio * 100)}%)"
            )
            println("[DEBUG_LOG] - Threshold: 0.001 (0.1%)")
            println("[DEBUG_LOG] - Cars detected: ${blackPixelRatio > 0.001}")

            // Try different thresholds
            for (threshold in listOf(0.0001, 0.0005, 0.001, 0.005, 0.01)) {
                println("[DEBUG_LOG] - At threshold $threshold: ${blackPixelRatio > threshold}")
            }

            // Clean up
            processedImage.originalMat.release()
            processedImage.transformedMat.release()
            resultMask.release()
            tempFile.delete()
            yoloInference.close()

            println("[DEBUG_LOG] Results test completed successfully")

        } catch (e: Exception) {
            println("[DEBUG_LOG] Results test failed: ${e.message}")
            e.printStackTrace()
        }
    }
}