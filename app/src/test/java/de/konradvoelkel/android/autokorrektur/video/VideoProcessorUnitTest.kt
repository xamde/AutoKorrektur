package de.konradvoelkel.android.autokorrektur.video

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
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
        val rawW = 1081
        val rawH = 1921
        val targetW = (rawW / 2) * 2
        val targetH = (rawH / 2) * 2

        assertEquals(1080, targetW)
        assertEquals(1920, targetH)
        assertEquals(0, targetW % 2)
        assertEquals(0, targetH % 2)
    }
}
