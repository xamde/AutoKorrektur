package de.konradvoelkel.android.autokorrektur.shared

import org.junit.Assert.assertTrue
import org.junit.Test

class JvmTestUtilsUsageTest {
    @Test
    fun testDeterministicDoubles_areWithinRange() {
        val iter: Iterator<Double> = JvmTestUtils.deterministicDoubles(42L)
        repeat(5) {
            val x = iter.next()
            assertTrue("value within [0,1)", x >= 0.0 && x < 1.0)
        }
    }

    @Test
    fun testBuildIntArray_andApproxEquals() {
        val arr = JvmTestUtils.buildIntArray(3) { it * it }
        // 0^2 + 1^2 + 2^2 = 5
        val sum = arr.sum().toFloat()
        assertTrue(JvmTestUtils.approxEquals(sum, 5.0f))
    }
}
