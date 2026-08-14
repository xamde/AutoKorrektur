package de.konradvoelkel.android.autokorrektur.ml.errors

/** Base exception for YOLO object detection and segmentation failures. */
sealed class YoloException(message: String, cause: Throwable? = null) :
    RuntimeException(message, cause)

/** Thrown when model weights/assets fail to load or initialize. */
class ModelLoadException(message: String, cause: Throwable? = null) : YoloException(message, cause)

/** Thrown when tensor execution or inference pass fails. */
class InferenceException(message: String, cause: Throwable? = null) : YoloException(message, cause)

/** Thrown when input or output tensor dimensions do not match expected shapes. */
class ShapeMismatchException(message: String, cause: Throwable? = null) :
    YoloException(message, cause)

/** Thrown when an inference method is invoked prior to initializing the model engine. */
class ModelNotInitializedException(message: String) : YoloException(message)

/** Base exception for image inpainting failures. */
open class InpaintException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

/** Thrown when remote Cloud SDXL inpainting HTTP communication fails or returns an error. */
class CloudInferenceException(message: String, cause: Throwable? = null) : InpaintException(message, cause)

/** Thrown when the daily free cloud inpainting edit quota has been exceeded. */
class QuotaExceededException(message: String) : InpaintException(message)
