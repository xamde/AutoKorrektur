package de.konradvoelkel.android.autokorrektur.ml.config


/**
 * Configuration values for YOLO post-processing and filtering.
 * Defaults mirror the constants currently embedded in YoloInferenceTFLite.
 */
data class YoloConfig(
    val scoreThreshold: Float = 0.6f,
    val iouThreshold: Float = 0.9f,
    val topAmountPerClass: Int = 100,
    val vehicleClassIndices: IntArray = intArrayOf(2, 3, 5, 7),
    val labels: Array<String> = DEFAULT_LABELS
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is YoloConfig) return false
        return scoreThreshold == other.scoreThreshold &&
                iouThreshold == other.iouThreshold &&
                topAmountPerClass == other.topAmountPerClass &&
                vehicleClassIndices.contentEquals(other.vehicleClassIndices) &&
                labels.contentEquals(other.labels)
    }

    override fun hashCode(): Int {
        var result = scoreThreshold.hashCode()
        result = 31 * result + iouThreshold.hashCode()
        result = 31 * result + topAmountPerClass
        result = 31 * result + vehicleClassIndices.contentHashCode()
        result = 31 * result + labels.contentHashCode()
        return result
    }

    companion object {
        // COCO 80 classes
        val DEFAULT_LABELS = arrayOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        )
    }
}