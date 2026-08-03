package de.konradvoelkel.android.autokorrektur.ml

import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized

@RunWith(Parameterized::class)
class ImageProcessingUtilsTest(
    private val stride: Int,
    private val width: Int,
    private val height: Int,
    private val expectedW: Int,
    private val expectedH: Int
) {
    companion object {
        @JvmStatic
        @Parameterized.Parameters(name = "stride={0}, {1}x{2} -> {3}x{4}")
        fun data(): Collection<Array<Int>> = listOf(
            arrayOf(32, 640, 480, 640, 480),      // already divisible
            arrayOf(32, 641, 479, 640, 480),      // round width down, height up
            arrayOf(16, 1023, 1025, 1024, 1024),  // symmetric around half stride
            arrayOf(8, 5, 10, 8, 8),              // both adjusted
            arrayOf(10, 14, 15, 10, 20),          // custom non-power-of-two stride
        )
    }

    @Test
    fun testDivStride() {
        val (w, h) = ImageProcessingUtils.divStride(stride, width, height)
        assertEquals("width", expectedW, w)
        assertEquals("height", expectedH, h)
    }
}
