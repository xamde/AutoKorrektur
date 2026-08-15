package de.konradvoelkel.android.autokorrektur.video

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class VideoProcessorUnitTest {

    @Test
    fun testVideoProcessingResult_dataModel() {
        val testFile = File("/tmp/test.mp4")
        val result = VideoProcessingResult(
            outputFile = testFile,
            totalFrames = 150,
            durationMs = 5000L,
            width = 1080,
            height = 1920
        )

        assertEquals(testFile, result.outputFile)
        assertEquals(150, result.totalFrames)
        assertEquals(5000L, result.durationMs)
        assertEquals(1080, result.width)
        assertEquals(1920, result.height)
    }

    @Test
    fun testVideoEncoder_dimensionsEvenMath() {
        val testResolutions = listOf(
            Pair(1081, 1921),
            Pair(719, 1279),
            Pair(641, 641),
            Pair(1920, 1080)
        )

        for ((rawW, rawH) in testResolutions) {
            val targetW = (rawW / 2) * 2
            val targetH = (rawH / 2) * 2

            assertEquals(0, targetW % 2)
            assertEquals(0, targetH % 2)
            assertTrue(targetW <= rawW)
            assertTrue(targetH <= rawH)
        }
    }

    @Test
    fun testVideoEncoder_presentationTimestampsMath() {
        val fps = 30
        val totalFrames = 150 // 5 seconds at 30 fps
        val timestampsUs = (0 until totalFrames).map { frameIdx ->
            frameIdx * (1_000_000L / fps)
        }

        assertEquals(0L, timestampsUs.first())
        assertEquals(150, timestampsUs.size)
        // 149 * (1_000_000 / 30) = 149 * 33333 = 4966617 us (~4.966 seconds)
        assertTrue(timestampsUs.last() in 4_900_000L..5_000_000L)
    }
}
