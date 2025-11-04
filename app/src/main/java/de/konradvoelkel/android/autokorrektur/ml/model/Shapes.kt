package de.konradvoelkel.android.autokorrektur.ml.model

/**
 * Captures the discovered tensor shapes for input and outputs.
 */
data class Shapes(
    val inputH: Int,
    val inputW: Int,
    val inputC: Int,
    val detShape: IntArray,
    val protoShape: IntArray
) {
    // Ensure structural equality for array fields to avoid reference-equality pitfalls
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Shapes) return false
        return inputH == other.inputH &&
            inputW == other.inputW &&
            inputC == other.inputC &&
            detShape.contentEquals(other.detShape) &&
            protoShape.contentEquals(other.protoShape)
    }

    override fun hashCode(): Int {
        var result = inputH
        result = 31 * result + inputW
        result = 31 * result + inputC
        result = 31 * result + detShape.contentHashCode()
        result = 31 * result + protoShape.contentHashCode()
        return result
    }
}
