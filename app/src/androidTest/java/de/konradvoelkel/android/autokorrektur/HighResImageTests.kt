package de.konradvoelkel.android.autokorrektur

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import org.junit.After
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream

/**
 * Tests to verify that high-resolution images can be loaded and processed without crashing.
 */
@RunWith(AndroidJUnit4::class)
class HighResImageTests : de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest() {

    private lateinit var imageProcessor: ImageProcessor
    private val tempFiles = mutableListOf<File>()

    @Before
    fun setUp() {
        imageProcessor = ImageProcessor(appContext)
    }

    @After
    fun tearDown() {
        tempFiles.forEach { it.delete() }
    }

    /**
     * Helper method to create a synthetic high-resolution image for testing.
     * This creates an image larger than what might fit in memory if not handled properly.
     */
    private fun createHighResTestImage(width: Int, height: Int): File {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Fill with a simple pattern to make it realistic
        val canvas = android.graphics.Canvas(bitmap)
        val paint = android.graphics.Paint().apply {
            color = android.graphics.Color.BLUE
        }
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), paint)

        // Draw some shapes to make it more interesting
        paint.color = android.graphics.Color.RED
        canvas.drawCircle(width / 2f, height / 2f, width / 4f, paint)

        val file = File(appContext.cacheDir, "high_res_test_${width}x${height}.jpg")
        FileOutputStream(file).use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
        }
        bitmap.recycle()
        tempFiles.add(file)
        return file
    }

    @Test
    fun testLoadHighResImage_8MP_shouldNotCrash() {
        // Create a ~8MP image (typical modern smartphone resolution)
        val testFile = createHighResTestImage(3264, 2448) // 8MP
        val uri = Uri.fromFile(testFile)

        // This should not crash - the image processor should handle it gracefully
        val processedImage = imageProcessor.processInputImage(
            imageUri = uri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )

        assertNotNull("Processed image should not be null", processedImage)
        assertNotNull("Original bitmap should not be null", processedImage.originalBitmap)
        assertTrue(
            "Original bitmap width should be positive",
            processedImage.originalBitmap.width > 0
        )
        assertTrue(
            "Original bitmap height should be positive",
            processedImage.originalBitmap.height > 0
        )

        // Clean up
        processedImage.originalBitmap.recycle()
        processedImage.transformedBitmap.recycle()
    }

    @Test
    fun testLoadHighResImage_12MP_shouldNotCrash() {
        // Create a ~12MP image (common on modern devices)
        val testFile = createHighResTestImage(4000, 3000) // 12MP
        val uri = Uri.fromFile(testFile)

        // This should not crash
        val processedImage = imageProcessor.processInputImage(
            imageUri = uri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )

        assertNotNull("Processed image should not be null", processedImage)
        assertNotNull("Original bitmap should not be null", processedImage.originalBitmap)

        // Clean up
        processedImage.originalBitmap.recycle()
        processedImage.transformedBitmap.recycle()
    }

    @Test
    fun testLoadHighResImage_20MP_shouldNotCrash() {
        // Create a very high-res image (~20MP)
        val testFile = createHighResTestImage(5472, 3648) // ~20MP
        val uri = Uri.fromFile(testFile)

        // This should not crash even with very high resolution
        val processedImage = imageProcessor.processInputImage(
            imageUri = uri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = null
        )

        assertNotNull("Processed image should not be null", processedImage)
        assertNotNull("Original bitmap should not be null", processedImage.originalBitmap)

        // Clean up
        processedImage.originalBitmap.recycle()
        processedImage.transformedBitmap.recycle()
    }

    @Test
    fun testLoadHighResImageWithDownscaling_shouldRespectDownscaleLimit() {
        // Create a high-res image
        val testFile = createHighResTestImage(4000, 3000) // 12MP
        val uri = Uri.fromFile(testFile)

        // Process with 2MP downscaling
        val processedImage = imageProcessor.processInputImage(
            imageUri = uri,
            modelWidth = 640,
            modelHeight = 640,
            downscaleMp = 2.0f
        )

        assertNotNull("Processed image should not be null", processedImage)

        // Verify the image was downscaled
        val megapixels =
            (processedImage.originalMat.rows() * processedImage.originalMat.cols()) / 1000000f
        assertTrue(
            "Image should be downscaled to approximately 2MP or less, but was ${megapixels}MP",
            megapixels <= 2.1f // Allow small margin
        )

        // Clean up
        processedImage.originalBitmap.recycle()
        processedImage.transformedBitmap.recycle()
    }

    @Test
    fun testLoadExtremelyLargeImage_shouldHandleGracefully() {
        // Create an extremely large image that would definitely cause OOM if loaded fully
        // 40MP+ image
        val testFile = createHighResTestImage(7680, 5760) // ~44MP
        val uri = Uri.fromFile(testFile)

        try {
            // This should either succeed with downsampling or fail gracefully
            val processedImage = imageProcessor.processInputImage(
                imageUri = uri,
                modelWidth = 640,
                modelHeight = 640,
                downscaleMp = null
            )

            assertNotNull("Processed image should not be null", processedImage)

            // If we got here, it worked! Clean up
            processedImage.originalBitmap.recycle()
            processedImage.transformedBitmap.recycle()
        } catch (e: OutOfMemoryError) {
            // If we get an OOM, that's what we're trying to prevent, so this test should help us fix it
            fail("OutOfMemoryError occurred when loading extremely large image - this is the bug we need to fix: ${e.message}")
        } catch (e: Exception) {
            // Other exceptions might be okay (e.g., file too large), but OOM is not
            if (e.message?.contains("OutOfMemory") == true ||
                e.message?.contains("Failed to allocate") == true
            ) {
                fail("Memory allocation error when loading extremely large image: ${e.message}")
            }
            // For other exceptions, we can tolerate them as long as the app doesn't crash
            println("Exception loading extremely large image (this may be acceptable): ${e.message}")
        }
    }

    @Test
    fun testBitmapFactoryOptions_inSampleSize_shouldReduceMemoryUsage() {
        // This test verifies that using inSampleSize reduces memory usage
        val testFile = createHighResTestImage(4000, 3000) // 12MP

        // Load full resolution
        val options1 = BitmapFactory.Options()
        val fullBitmap = BitmapFactory.decodeFile(testFile.absolutePath, options1)
        assertNotNull("Full bitmap should load", fullBitmap)
        val fullSize = fullBitmap.byteCount
        fullBitmap.recycle()

        // Load with inSampleSize = 2 (1/4 the pixels, roughly 1/4 the memory)
        val options2 = BitmapFactory.Options().apply {
            inSampleSize = 2
        }
        val downsampledBitmap = BitmapFactory.decodeFile(testFile.absolutePath, options2)
        assertNotNull("Downsampled bitmap should load", downsampledBitmap)
        val downsampledSize = downsampledBitmap.byteCount
        downsampledBitmap.recycle()

        assertTrue(
            "Downsampled bitmap should use significantly less memory (full: $fullSize, downsampled: $downsampledSize)",
            downsampledSize < fullSize / 2
        )
    }
}
