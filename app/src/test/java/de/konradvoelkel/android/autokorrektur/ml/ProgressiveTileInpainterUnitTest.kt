package de.konradvoelkel.android.autokorrektur.ml

import de.konradvoelkel.android.autokorrektur.ml.progressive.ProgressiveTileInpainter
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.core.Point
import org.opencv.core.Rect

class ProgressiveTileInpainterUnitTest {

    @Test
    fun testMergeBoundingBoxes_emptyList() {
        val merged = ProgressiveTileInpainter.mergeBoundingBoxes(
            emptyList(),
            paddingRatio = 0.2f,
            imgW = 1000,
            imgH = 1000
        )
        assertTrue(merged.isEmpty())
    }

    @Test
    fun testMergeBoundingBoxes_singleBox() {
        val box = Rect(100, 100, 200, 150)
        val merged = ProgressiveTileInpainter.mergeBoundingBoxes(
            listOf(box),
            paddingRatio = 0.2f,
            imgW = 1000,
            imgH = 1000
        )
        assertEquals(1, merged.size)
        val res = merged[0]
        assertTrue(res.x <= 100)
        assertTrue(res.y <= 100)
        assertTrue(res.width >= 200)
        assertTrue(res.height >= 150)
    }

    @Test
    fun testMergeBoundingBoxes_overlappingMergedIntoOne() {
        val box1 = Rect(100, 100, 200, 200)
        val box2 = Rect(150, 150, 200, 200)
        val merged = ProgressiveTileInpainter.mergeBoundingBoxes(
            listOf(box1, box2),
            paddingRatio = 0.1f,
            imgW = 1000,
            imgH = 1000
        )
        assertEquals(1, merged.size)
    }
}
