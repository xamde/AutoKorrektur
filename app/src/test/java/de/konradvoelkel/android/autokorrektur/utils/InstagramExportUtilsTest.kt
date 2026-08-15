package de.konradvoelkel.android.autokorrektur.utils

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class InstagramExportUtilsTest {

    @Test
    fun aspectRatio_squareOneToOne_hasExpectedResolution() {
        val ratio = InstagramExportUtils.AspectRatio.SQUARE_1_1
        assertEquals(1080, ratio.width)
        assertEquals(1080, ratio.height)
        assertEquals(1.0f, ratio.width.toFloat() / ratio.height.toFloat(), 0.001f)
    }

    @Test
    fun aspectRatio_portraitFourFive_hasExpectedResolution() {
        val ratio = InstagramExportUtils.AspectRatio.PORTRAIT_4_5
        assertEquals(1080, ratio.width)
        assertEquals(1350, ratio.height)
        assertEquals(0.8f, ratio.width.toFloat() / ratio.height.toFloat(), 0.001f)
    }

    @Test
    fun aspectRatio_storyNineSixteen_hasExpectedResolution() {
        val ratio = InstagramExportUtils.AspectRatio.STORY_9_16
        assertEquals(1080, ratio.width)
        assertEquals(1920, ratio.height)
        assertEquals(9.0f / 16.0f, ratio.width.toFloat() / ratio.height.toFloat(), 0.001f)
    }

    @Test
    fun layoutStyles_containsSideBySideAndStacked() {
        val styles = InstagramExportUtils.LayoutStyle.entries
        assertNotNull(styles)
        assertEquals(2, styles.size)
        assertTrue(styles.contains(InstagramExportUtils.LayoutStyle.SIDE_BY_SIDE))
        assertTrue(styles.contains(InstagramExportUtils.LayoutStyle.STACKED))
    }
}
