package de.konradvoelkel.android.autokorrektur.ml

import de.konradvoelkel.android.autokorrektur.ml.factory.InpaintingEngineFactory
import de.konradvoelkel.android.autokorrektur.ml.factory.InpaintingModelType
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class LamaInferenceUnitTest {

    @Test
    fun testComputePaddedDimensions_roundsUpToMultipleOf8() {
        assertEquals(512, LamaInference.computePaddedDimension(512, 8))
        assertEquals(520, LamaInference.computePaddedDimension(513, 8))
        assertEquals(640, LamaInference.computePaddedDimension(637, 8))
        assertEquals(8, LamaInference.computePaddedDimension(1, 8))
    }

    @Test
    fun testInpaintingModelType_enumValuesAndTitles() {
        assertEquals("MI-GAN (Ultra-Fast)", InpaintingModelType.MIGAN.displayName)
        assertEquals("LaMa (High-Fidelity)", InpaintingModelType.LAMA.displayName)
        assertEquals("Stable Diffusion XL (Cloud)", InpaintingModelType.SDXL_CLOUD.displayName)
    }

    @Test
    fun testInpaintingModelType_fromString() {
        assertEquals(InpaintingModelType.LAMA, InpaintingModelType.fromString("lama"))
        assertEquals(InpaintingModelType.MIGAN, InpaintingModelType.fromString("migan"))
        assertEquals(InpaintingModelType.SDXL_CLOUD, InpaintingModelType.fromString("sdxl"))
        assertEquals(InpaintingModelType.MIGAN, InpaintingModelType.fromString("unknown_fallback"))
    }
}
