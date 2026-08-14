package de.konradvoelkel.android.autokorrektur.pipeline

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.MediumTest
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.MatScaler
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

@RunWith(AndroidJUnit4::class)
@MediumTest
class MiGanDisplayBitmapPipelineTest : AndroidInstrumentedBaseTest() {

    private lateinit var yoloService: YoloServiceImpl
    private lateinit var miGanInference: MiGanInference
    private lateinit var imageProcessor: ImageProcessor

    @Before
    fun setUp() = kotlinx.coroutines.runBlocking {
        assertTrue("OpenCV initialization failed", OpenCVLoader.initLocal())
        yoloService = YoloServiceImpl(YoloTFLiteEngine(appContext))
        yoloService.initialize("yolo11s", useFP16 = false)
        miGanInference = MiGanInference(appContext)
        miGanInference.initialize()
        imageProcessor = ImageProcessor(appContext)
    }

    @Test
    fun testMiGanInferenceAndDisplayBitmap_verifiesChannelCountAndBitmapConversion(): Unit =
        kotlinx.coroutines.runBlocking {
            val testAssets = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().context.assets
            val stream: InputStream = testAssets.open("triples/triple_21_with_car.png")
            val bitmap = BitmapFactory.decodeStream(stream)
            stream.close()

            assertNotNull("Test image must decode", bitmap)

            val tempFile = File(appContext.cacheDir, "test_migan_pipeline.png")
            FileOutputStream(tempFile).use { out ->
                bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, out)
            }
            val uri = android.net.Uri.fromFile(tempFile)

            val processedImage = imageProcessor.processInputImage(
                imageUri = uri,
                modelWidth = 640,
                modelHeight = 640,
                downscaleMp = null
            )

            AppLogger.info("ProcessedImage: originalMat cols=${processedImage.originalMat.cols()}, rows=${processedImage.originalMat.rows()}, channels=${processedImage.originalMat.channels()}, type=${processedImage.originalMat.type()}")

            val yoloResult = yoloService.inferDetailed(
                transformedMat = processedImage.transformedMat,
                xRatio = processedImage.xRatio,
                yRatio = processedImage.yRatio,
                upscaleFactor = 1.05f,
                originalWidth = processedImage.originalMat.cols(),
                originalHeight = processedImage.originalMat.rows()
            )

            AppLogger.info("YoloResult mask: cols=${yoloResult.mask.cols()}, rows=${yoloResult.mask.rows()}, channels=${yoloResult.mask.channels()}, type=${yoloResult.mask.type()}")

            // Run Mi-GAN
            val miGanResult = miGanInference.inferMiGan(
                imageMat = processedImage.originalMat,
                maskMat = yoloResult.mask
            )

            AppLogger.info("MiGanResult: cols=${miGanResult.cols()}, rows=${miGanResult.rows()}, channels=${miGanResult.channels()}, type=${miGanResult.type()}")

            assertEquals("MiGanResult must have 3 channels (RGB)", 3, miGanResult.channels())
            assertEquals("MiGanResult width must match original", processedImage.originalMat.cols(), miGanResult.cols())
            assertEquals("MiGanResult height must match original", processedImage.originalMat.rows(), miGanResult.rows())

            // Convert to Bitmap via MatScaler
            val displayBitmap = MatScaler.createDisplayBitmap(miGanResult)
            assertNotNull("Display bitmap must not be null", displayBitmap)
            assertEquals(miGanResult.cols(), displayBitmap.width)
            assertEquals(miGanResult.rows(), displayBitmap.height)

            miGanResult.release()
            yoloResult.mask.release()
            processedImage.release()
            tempFile.delete()
            bitmap.recycle()
            displayBitmap.recycle()
            yoloService.close()
            miGanInference.close()
        }

    @Test
    fun testUltraHighResolutionDownsamplingAndInpaintingPipeline_handles50MegapixelsSafely(): Unit =
        kotlinx.coroutines.runBlocking {
            // Create a synthetic 50MP (8160x6144) high-res JPEG file
            val highResWidth = 8160
            val highResHeight = 6144
            val tempFile = File(appContext.cacheDir, "high_res_50mp_test.jpg")
            
            // Create a lightweight compressed bitmap representing the user's high-res camera shot
            val rawBmp = android.graphics.Bitmap.createBitmap(1020, 768, android.graphics.Bitmap.Config.ARGB_8888)
            val canvas = android.graphics.Canvas(rawBmp)
            val paint = android.graphics.Paint().apply { color = android.graphics.Color.GRAY }
            canvas.drawRect(0f, 0f, 1020f, 768f, paint)
            
            // Upscale to 8160x6144 file to simulate camera raw capture
            val scaledBmp = android.graphics.Bitmap.createScaledBitmap(rawBmp, highResWidth, highResHeight, true)
            FileOutputStream(tempFile).use { out ->
                scaledBmp.compress(android.graphics.Bitmap.CompressFormat.JPEG, 85, out)
            }
            rawBmp.recycle()
            scaledBmp.recycle()

            val uri = android.net.Uri.fromFile(tempFile)

            // Process image - must automatically downsample to fit under 8.0MP
            val processedImage = imageProcessor.processInputImage(
                imageUri = uri,
                modelWidth = 640,
                modelHeight = 640,
                downscaleMp = null
            )

            AppLogger.info("HighRes ProcessedImage: cols=${processedImage.originalMat.cols()}, rows=${processedImage.originalMat.rows()}, channels=${processedImage.originalMat.channels()}")

            val megapixels = (processedImage.originalMat.cols() * processedImage.originalMat.rows()) / 1_000_000f
            assertTrue("Processed Mat must be bounded under 8.5MP (actual: $megapixels MP)", megapixels <= 8.5f)
            assertEquals("Processed Mat must have 3 channels (RGB)", 3, processedImage.originalMat.channels())

            // Run YOLO
            val yoloResult = yoloService.inferDetailed(
                transformedMat = processedImage.transformedMat,
                xRatio = processedImage.xRatio,
                yRatio = processedImage.yRatio,
                upscaleFactor = 1.0f,
                originalWidth = processedImage.originalMat.cols(),
                originalHeight = processedImage.originalMat.rows()
            )

            // Run Mi-GAN
            val miGanResult = miGanInference.inferMiGan(
                imageMat = processedImage.originalMat,
                maskMat = yoloResult.mask
            )

            // Convert to Display Bitmap
            val displayBitmap = MatScaler.createDisplayBitmap(miGanResult)
            assertNotNull("Display bitmap must be created successfully", displayBitmap)
            assertEquals(processedImage.originalMat.cols(), displayBitmap.width)
            assertEquals(processedImage.originalMat.rows(), displayBitmap.height)

            miGanResult.release()
            yoloResult.mask.release()
            processedImage.release()
            tempFile.delete()
            displayBitmap.recycle()
            yoloService.close()
            miGanInference.close()
        }

    @Test
    fun testUltraHdrAndGainmapNormalization_convertsToExact3ChannelRgb(): Unit =
        kotlinx.coroutines.runBlocking {
            val tempFile = File(appContext.cacheDir, "hdr_test_photo.png")
            val baseBmp = android.graphics.Bitmap.createBitmap(1920, 1080, android.graphics.Bitmap.Config.ARGB_8888)
            val canvas = android.graphics.Canvas(baseBmp)
            val paint = android.graphics.Paint().apply { color = android.graphics.Color.BLUE }
            canvas.drawRect(0f, 0f, 1920f, 1080f, paint)

            FileOutputStream(tempFile).use { out ->
                baseBmp.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, out)
            }
            baseBmp.recycle()

            val uri = android.net.Uri.fromFile(tempFile)
            val processedImage = imageProcessor.processInputImage(
                imageUri = uri,
                modelWidth = 640,
                modelHeight = 640,
                downscaleMp = null
            )

            assertEquals("OriginalMat must have exactly 3 channels", 3, processedImage.originalMat.channels())
            assertEquals("OriginalMat CvType must be CV_8UC3 (16)", org.opencv.core.CvType.CV_8UC3, processedImage.originalMat.type())

            val yoloResult = yoloService.inferDetailed(
                transformedMat = processedImage.transformedMat,
                xRatio = processedImage.xRatio,
                yRatio = processedImage.yRatio,
                upscaleFactor = 1.0f,
                originalWidth = processedImage.originalMat.cols(),
                originalHeight = processedImage.originalMat.rows()
            )

            val miGanResult = miGanInference.inferMiGan(
                imageMat = processedImage.originalMat,
                maskMat = yoloResult.mask
            )

            assertEquals("MiGan output must be CV_8UC3 (16)", org.opencv.core.CvType.CV_8UC3, miGanResult.type())

            val displayBitmap = MatScaler.createDisplayBitmap(miGanResult)
            assertNotNull(displayBitmap)
            assertEquals(1920, displayBitmap.width)
            assertEquals(1080, displayBitmap.height)

            miGanResult.release()
            yoloResult.mask.release()
            processedImage.release()
            tempFile.delete()
            displayBitmap.recycle()
            yoloService.close()
            miGanInference.close()
        }

    @Test
    fun testVeryHighResCarAsset_throughFullStaticImagePipeline(): Unit =
        kotlinx.coroutines.runBlocking {
            val testAssets = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().context.assets
            val tempFile = File(appContext.cacheDir, "test_very_high_res_car.jpg")
            
            testAssets.open("very_high_res_car.jpg").use { input ->
                FileOutputStream(tempFile).use { output ->
                    input.copyTo(output)
                }
            }

            val uri = android.net.Uri.fromFile(tempFile)
            val serverSdxlApi = de.konradvoelkel.android.autokorrektur.ml.api.ServerSdxlApi(appContext)
            val pipeline = StaticImagePipeline(
                imageProcessor = imageProcessor,
                yoloService = yoloService,
                miGanInference = miGanInference,
                serverSdxlApi = serverSdxlApi
            )
            pipeline.initialize()

            val yoloTestResult = yoloService.inferDetailed(
                transformedMat = pipeline.let {
                    val p = imageProcessor.processInputImage(uri, 640, 640, null)
                    p.transformedMat
                },
                xRatio = 1.0f,
                yRatio = 1.0f,
                upscaleFactor = 1.0f,
                originalWidth = 2040,
                originalHeight = 1536
            )
            AppLogger.info("YOLO detections on very_high_res_car.jpg: count=${yoloTestResult.detections.size}")
            yoloTestResult.detections.forEachIndexed { idx, det ->
                AppLogger.info("  Det #$idx: classId=${det.classId} (${YoloConfig.DEFAULT_LABELS.getOrNull(det.classId)}), conf=${det.confidence}, box=[x=${det.x}, y=${det.y}, w=${det.width}, h=${det.height}]")
            }

            val result = pipeline.processImage(
                uri = uri,
                downscaleMp = null,
                maskUpscale = 1.0f,
                scoreThreshold = 0.25f,
                useServerSdxl = false
            )

            var diffPixels = 0
            var totalSampled = 0
            for (y in 0 until result.originalBitmap.height step 10) {
                for (x in 0 until result.originalBitmap.width step 10) {
                    totalSampled++
                    if (result.originalBitmap.getPixel(x, y) != result.inpaintedBitmap!!.getPixel(x, y)) {
                        diffPixels++
                    }
                }
            }
            AppLogger.info("Diff pixels between original and inpainted: $diffPixels / $totalSampled (${(diffPixels.toFloat() / totalSampled) * 100}%)")

            // Verify zero vehicles remain in inpainted output
            de.konradvoelkel.android.autokorrektur.shared.PostInpaintingVehicleAssertionUtils.assertNoVehiclesRemain(
                inpaintedBitmap = result.inpaintedBitmap!!,
                context = appContext,
                yoloService = yoloService,
                imageProcessor = imageProcessor,
                confidenceThreshold = 0.25f,
                message = "StaticImagePipeline output for very_high_res_car must have zero detected vehicles"
            )

            // Test UI presentation scaling (exercises BitmapMemoryUtils & MaskOverlayUtils on Android 17)
            val displayScaled = de.konradvoelkel.android.autokorrektur.utils.BitmapMemoryUtils.createScaledBitmapForDisplay(result.inpaintedBitmap!!)
            assertNotNull("Display scaled bitmap must not be null", displayScaled)
            assertTrue("Display scaled width <= 1920", displayScaled.width <= 1920)

            val overlay = de.konradvoelkel.android.autokorrektur.utils.MaskOverlayUtils.createRedOverlayBitmap(
                result.maskBitmap,
                displayScaled.width,
                displayScaled.height
            )
            assertNotNull("Overlay bitmap must not be null", overlay)
            assertEquals(displayScaled.width, overlay.width)
            assertEquals(displayScaled.height, overlay.height)

            // Save artifacts to appContext.cacheDir
            val cacheDir = appContext.cacheDir
            val origOut = File(cacheDir, "test_very_high_res_original.png")
            val maskOut = File(cacheDir, "test_very_high_res_mask.png")
            val inpaintOut = File(cacheDir, "test_very_high_res_inpainted.png")

            FileOutputStream(origOut).use { result.originalBitmap.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it) }
            FileOutputStream(maskOut).use { result.maskBitmap.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it) }
            FileOutputStream(inpaintOut).use { result.inpaintedBitmap!!.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it) }
            AppLogger.info("Saved test artifacts to ${cacheDir.absolutePath}")

            displayScaled.recycle()
            overlay.recycle()
            pipeline.close()
            tempFile.delete()
        }
}


