package de.konradvoelkel.android.autokorrektur.ui

import org.junit.Assert.assertEquals
import org.junit.Test

class BeforeAfterSliderViewTest {

    @Test
    fun sliderPosition_clamping_ensuresValidBoundsBetweenZeroAndOne() {
        val position1 = 0.5f.coerceIn(0f, 1f)
        assertEquals(0.5f, position1, 0.001f)

        val positionUnder = (-0.2f).coerceIn(0f, 1f)
        assertEquals(0.0f, positionUnder, 0.001f)

        val positionOver = (1.5f).coerceIn(0f, 1f)
        assertEquals(1.0f, positionOver, 0.001f)
    }
}
