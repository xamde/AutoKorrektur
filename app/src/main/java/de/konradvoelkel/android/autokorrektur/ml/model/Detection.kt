package de.konradvoelkel.android.autokorrektur.ml.model

/**
 * Public detection data model used by refactored YOLO pipeline.
 * Note: A legacy inner class with the same name exists inside YoloInferenceTFLite.
 * Callers should import this package-qualified type explicitly when using the new APIs.
 */
data class Detection(
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
    val confidence: Float,
    val classId: Int,
    val maskCoefficients: FloatArray
) {
    // Ensure structural equality for array field
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Detection) return false
        return x == other.x &&
                y == other.y &&
                width == other.width &&
                height == other.height &&
                confidence == other.confidence &&
                classId == other.classId &&
                maskCoefficients.contentEquals(other.maskCoefficients)
    }

    override fun hashCode(): Int {
        var result = x.hashCode()
        result = 31 * result + y.hashCode()
        result = 31 * result + width.hashCode()
        result = 31 * result + height.hashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + classId
        result = 31 * result + maskCoefficients.contentHashCode()
        return result
    }
}
