package de.konradvoelkel.android.autokorrektur.utils

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class InstagramExportUtilsTest {

    @Test
    fun aspectRatio_squareOneToOne_hasExpectedResolution() {
        val ratio = InstagramExportUtils.AspectRatio.SQUARE_1_1
        assertEquals(1080, ratio.width)
        assertEquals(1080, ratio.height)
    }

    @Test
    fun aspectRatio_portraitFourFive_hasExpectedResolution() {
        val ratio = InstagramExportUtils.AspectRatio.PORTRAIT_4_5
        assertEquals(1080, ratio.width)
        assertEquals(1350, ratio.height)
    }

    @Test
    fun aspectRatio_storyNineSixteen_hasExpectedResolution() {
        val ratio = InstagramExportUtils.AspectRatio.STORY_9_16
        assertEquals(1080, ratio.width)
        assertEquals(1920, ratio.height)
    }

    @Test
    fun layoutStyles_containsSideBySideAndStacked() {
        val styles = InstagramExportUtils.LayoutStyle.values()
        assertNotNull(styles)
        assertEquals(2, styles.size)
        assertEquals(InstagramExportUtils.LayoutStyle.SIDE_BY_SIDE, styles[0])
        assertEquals(InstagramExportUtils.LayoutStyle.STACKED, styles[1])
    }
}
