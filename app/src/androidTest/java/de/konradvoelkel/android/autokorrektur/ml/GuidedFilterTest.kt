package de.konradvoelkel.android.autokorrektur.ml

import androidx.test.ext.junit.runners.AndroidJUnit4
import de.konradvoelkel.android.autokorrektur.ml.mask.GuidedFilter
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

@RunWith(AndroidJUnit4::class)
class GuidedFilterTest : de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest() {

    @Test
    fun testDynamicRadiusScaling() {
        assertEquals(6, GuidedFilter.calculateDynamicRadius(640, 640))
        assertEquals(12, GuidedFilter.calculateDynamicRadius(1280, 720))
        assertEquals(37, GuidedFilter.calculateDynamicRadius(4000, 3000))
        assertTrue(GuidedFilter.calculateDynamicRadius(100, 100) >= 3)
    }

    @Test
    fun testGuidedFilterEdgeRefinementSnapsToGuidanceBoundary() {
        val width = 100
        val height = 100

        // 1. Create a guide image with a high-contrast vertical edge at x = 50
        val guide = Mat(height, width, CvType.CV_8UC3).apply { setTo(Scalar(30.0, 30.0, 30.0)) }
        val rightHalf = Rect(50, 0, 50, 100)
        val guideRight = Mat(guide, rightHalf)
        guideRight.setTo(Scalar(240.0, 240.0, 240.0))
        guideRight.release()

        // 2. Create an initial coarse/imprecise mask with edge at x = 54 (4 pixels displaced)
        // Background = 255 (left), Target = 0 (right)
        val coarseMask = Mat(height, width, CvType.CV_8UC1).apply { setTo(Scalar(255.0)) }
        val targetRect = Rect(54, 0, 46, 100)
        val coarseTarget = Mat(coarseMask, targetRect)
        coarseTarget.setTo(Scalar(0.0))
        coarseTarget.release()

        // 3. Apply Guided Filter with radius 8 and eps 0.001
        val refinedMask = GuidedFilter.filter(guide, coarseMask, radius = 8, eps = 0.001)

        assertFalse("Refined mask should not be empty", refinedMask.empty())
        assertEquals(width, refinedMask.cols())
        assertEquals(height, refinedMask.rows())

        // 4. Verify boundary shifted towards the guide edge (x = 50)
        // At x = 45 (left of edge): Should be background (255)
        val pixelLeft = refinedMask.get(50, 45)[0].toInt()
        assertEquals("Pixel left of guide edge should be background (255)", 255, pixelLeft)

        // At x = 52 (right of guide edge, between true edge and coarse edge):
        // Guided filter should have snapped it to target (0) based on the sharp transition in guide!
        val pixelRight = refinedMask.get(50, 52)[0].toInt()
        assertEquals("Pixel right of guide edge should snap to target (0)", 0, pixelRight)

        // Clean up
        guide.release()
        coarseMask.release()
        refinedMask.release()
    }
}
