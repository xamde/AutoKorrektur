package de.konradvoelkel.android.autokorrektur.ml.model

import java.nio.ByteBuffer

/**
 * Raw outputs coming from the TFLite engine for a single inference pass.
 */
data class RawOutputs(
    val detections: ByteBuffer,
    val prototypes: ByteBuffer,
    val shapes: Shapes
)
