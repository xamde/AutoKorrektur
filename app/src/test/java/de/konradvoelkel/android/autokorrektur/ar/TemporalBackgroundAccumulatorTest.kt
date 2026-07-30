package de.konradvoelkel.android.autokorrektur.ar

import org.junit.Assert.assertNotNull
import org.junit.Test

class TemporalBackgroundAccumulatorTest {

    @Test
    fun temporalBackgroundAccumulator_instantiatesSuccessfully() {
        val accumulator = TemporalBackgroundAccumulator()
        assertNotNull(accumulator)
        accumulator.reset()
    }
}
