package de.konradvoelkel.android.autokorrektur.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.InputStream
import java.nio.ByteBuffer
import java.util.Locale

/**
 * JVM Unit Test verifying all 50 image-triples (car image, mask image, carless ground truth).
 */
class FiftyImageTriplesUnitTest {

    data class PngInfo(
        val width: Int,
        val height: Int,
        val bitDepth: Int,
        val colorType: Int,
        val fileSizeBytes: Int
    )

    private fun parsePngHeader(stream: InputStream): PngInfo {
        val bytes = stream.readBytes()
        assertTrue("PNG file must have at least 30 bytes", bytes.size >= 30)

        // Verify PNG magic header: 0x89 0x50 0x4E 0x47 0x0D 0x0A 0x1A 0x0A
        val expectedHeader = intArrayOf(0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)
        for (i in 0 until 8) {
            assertEquals("PNG header byte $i mismatch", expectedHeader[i], bytes[i].toInt() and 0xFF)
        }

        // IHDR chunk starts at byte 12
        val width = ByteBuffer.wrap(bytes, 16, 4).int
        val height = ByteBuffer.wrap(bytes, 20, 4).int
        val bitDepth = bytes[24].toInt() and 0xFF
        val colorType = bytes[25].toInt() and 0xFF

        return PngInfo(
            width = width,
            height = height,
            bitDepth = bitDepth,
            colorType = colorType,
            fileSizeBytes = bytes.size
        )
    }

    @Test
    fun testAllFiftyImageTriplesStructureAndDimensions() {
        val totalTriples = 50

        for (i in 1..totalTriples) {
            val prefix = String.format(Locale.US, "triple_%02d", i)

            val carStream: InputStream? = javaClass.getResourceAsStream("/triples/${prefix}_with_car.png")
            val maskStream: InputStream? = javaClass.getResourceAsStream("/triples/${prefix}_mask.png")
            val carlessStream: InputStream? = javaClass.getResourceAsStream("/triples/${prefix}_without_car.png")

            assertNotNull("Triple $i with_car stream should exist", carStream)
            assertNotNull("Triple $i mask stream should exist", maskStream)
            assertNotNull("Triple $i without_car stream should exist", carlessStream)

            val carInfo = parsePngHeader(carStream!!)
            val maskInfo = parsePngHeader(maskStream!!)
            val carlessInfo = parsePngHeader(carlessStream!!)

            // 1. Verify width and height match across the triple
            assertTrue("Triple $i width (${carInfo.width}) should be positive", carInfo.width > 0)
            assertTrue("Triple $i height (${carInfo.height}) should be positive", carInfo.height > 0)

            assertEquals("Triple $i mask width should match car width", carInfo.width, maskInfo.width)
            assertEquals("Triple $i mask height should match car height", carInfo.height, maskInfo.height)
            assertEquals("Triple $i carless width should match car width", carInfo.width, carlessInfo.width)
            assertEquals("Triple $i carless height should match car height", carInfo.height, carlessInfo.height)

            // 2. Verify non-trivial file sizes
            val carMsg = "Triple $i with_car size (${carInfo.fileSizeBytes}) should be > 100B"
            assertTrue(carMsg, carInfo.fileSizeBytes > 100)

            val maskMsg = "Triple $i mask size (${maskInfo.fileSizeBytes}) should be > 100B"
            assertTrue(maskMsg, maskInfo.fileSizeBytes > 100)

            val carlessMsg = "Triple $i carless size (${carlessInfo.fileSizeBytes}) should be > 100B"
            assertTrue(carlessMsg, carlessInfo.fileSizeBytes > 100)

            carStream.close()
            maskStream.close()
            carlessStream.close()
        }
    }
}
