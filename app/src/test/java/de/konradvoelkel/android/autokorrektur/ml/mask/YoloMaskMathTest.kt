package de.konradvoelkel.android.autokorrektur.ml.mask

import org.junit.Assert.assertEquals
import org.junit.Test

class YoloMaskMathTest {

    @Test
    fun testCalculateCropRect_simple() {
        // box at (0.25, 0.25) with size (0.5, 0.5) in a 160x160 proto grid
        val r = YoloMaskMath.calculateCropRect(0.25f, 0.25f, 0.5f, 0.5f, 160, 160)
        assertEquals(40, r.x)
        assertEquals(40, r.y)
        assertEquals(80, r.width)
        assertEquals(80, r.height)
    }

    @Test
    fun testCalculateCropRect_boundary() {
        // box that covers the whole grid
        val r = YoloMaskMath.calculateCropRect(0f, 0f, 1f, 1f, 160, 160)
        assertEquals(0, r.x)
        assertEquals(0, r.y)
        assertEquals(160, r.width)
        assertEquals(160, r.height)
    }

    @Test
    fun testCalculatePlacement_centered() {
        // 640x640 input, detection at center (0.5, 0.5) size (0.2, 0.2)
        // model coordinates: cx=320, cy=320, w=128, h=128
        // upscaled mask: 128x128
        // targetX = 320 + 64 - 64 = 320. Wait, YOLO x,y is center or top-left?
        // In Detection.kt: "coordinate convention is undocumented; x,y are top-left, normalized 0..1" (G4)
        // Let's assume top-left as per G4 note.

        val p = YoloMaskMath.calculatePlacement(
            boxX = 0.4f, boxY = 0.4f, boxW = 0.2f, boxH = 0.2f,
            maskW = 128, maskH = 128,
            inputW = 640, inputH = 640
        )

        // xModel = 0.4 * 640 = 256
        // wModel = 0.2 * 640 = 128
        // targetX = 256 + 64 - 64 = 256
        assertEquals(256, p.dst.x)
        assertEquals(256, p.dst.y)
        assertEquals(128, p.dst.width)
        assertEquals(128, p.dst.height)
    }

    @Test
    fun testCalculatePlacement_outOfBounds() {
        // detection at far right
        val p = YoloMaskMath.calculatePlacement(
            boxX = 0.9f, boxY = 0.4f, boxW = 0.2f, boxH = 0.2f,
            maskW = 128, maskH = 128,
            inputW = 640, inputH = 640
        )
        // xModel = 0.9 * 640 = 576
        // wModel = 0.2 * 640 = 128
        // targetX = 576 + 64 - 64 = 576
        // dstW = min(128, 640 - 576) = min(128, 64) = 64
        assertEquals(576, p.dst.x)
        assertEquals(64, p.dst.width)
        assertEquals(64, p.src.width)
    }
}
